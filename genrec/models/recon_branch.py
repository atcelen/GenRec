"""Side-branch reconstruction network with sparse geometric cross-attention.

Gathers features from input view VAE latents using GeometricIndexMap correspondences,
then refines via FiLM-conditioned ResBlocks (timestep-aware). Produces a small x0
residual correction trained with L1 loss on observed regions only, completely
decoupled from diffusion MSE.
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FiLMResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv3x3 -> FiLM(temb) -> SiLU -> Conv3x3 + residual."""

    def __init__(self, channels: int, temb_dim: int = 1280, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, 2 * channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W)
            temb: (N, temb_dim)
        """
        h = self.act(self.norm(x))
        h = self.conv1(h)
        # FiLM conditioning after first conv
        film = self.temb_proj(temb)[:, :, None, None]  # (N, 2*C, 1, 1)
        scale, shift = film.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.dropout(self.act(h))
        h = self.conv2(h)
        return x + h


class SparseGeometricCrossAttention(nn.Module):
    """Multi-head cross-attention with K-nearest geometric gathering."""

    def __init__(self, dim: int, num_heads: int = 4, use_geo_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_geo_bias = use_geo_bias

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,        # (B*T_out, H*W, D)
        source: torch.Tensor,       # (B, T_in*H*W, D)
        indices: torch.Tensor,      # (B, T_out, H*W, K) — flat index into T_in*H*W
        valid_mask: torch.Tensor,   # (B, T_out, H*W, K) — bool
        geo_weights: torch.Tensor,  # (B, T_out, H*W, K) — normalized inv-dist weights
        T_out: int,
    ) -> torch.Tensor:
        """
        Returns: (B*T_out, H*W, D) — attention output.
        """
        B = source.shape[0]
        N = query.shape[1]  # H*W
        K = indices.shape[-1]

        # Flatten indices: (B, T_out, H*W, K) -> (B, T_out*H*W*K)
        flat_indices = indices.reshape(B, T_out * N * K)

        # Gather source features: (B, T_in*H*W, D) -> (B, T_out*H*W*K, D)
        flat_indices_expand = flat_indices.unsqueeze(-1).expand(-1, -1, self.dim)
        # Clamp indices to valid range for safe gather (invalid slots masked later)
        flat_indices_expand = flat_indices_expand.clamp(0, source.shape[1] - 1)
        gathered = torch.gather(source, 1, flat_indices_expand)

        # Reshape: (B, T_out*H*W*K, D) -> (B*T_out, N, K, D)
        gathered = gathered.reshape(B, T_out, N, K, self.dim)
        gathered = gathered.reshape(B * T_out, N, K, self.dim)

        # Reshape masks: (B, T_out, N, K) -> (B*T_out, N, K)
        valid = valid_mask.reshape(B * T_out, N, K)
        weights = geo_weights.reshape(B * T_out, N, K)

        # Layer norms
        q = self.norm_q(query)   # (B*T_out, N, D)
        kv = self.norm_kv(gathered)  # (B*T_out, N, K, D)

        # Q/K/V projections
        q = self.to_q(q)      # (B*T_out, N, D)
        k = self.to_k(kv)     # (B*T_out, N, K, D)
        v = self.to_v(kv)     # (B*T_out, N, K, D)

        # Multi-head reshape
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n k (h d) -> b h n k d', h=self.num_heads)
        v = rearrange(v, 'b n k (h d) -> b h n k d', h=self.num_heads)

        # Attention: q @ k^T
        attn = torch.einsum('bhnd, bhnkd -> bhnk', q, k) * self.scale

        # Mask invalid with -inf
        invalid = ~valid.unsqueeze(1)  # (B*T_out, 1, N, K)
        attn = attn.masked_fill(invalid, float('-inf'))

        # Add geometric weight bias
        if self.use_geo_bias:
            geo_bias = torch.log(weights.clamp(min=1e-8)).unsqueeze(1)  # (B*T_out, 1, N, K)
            attn = attn + geo_bias

        # Softmax + nan_to_num for all-invalid rows
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum of V
        out = torch.einsum('bhnk, bhnkd -> bhnd', attn, v)

        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class ReconSideBranch(nn.Module):
    """Side-branch network with pixel-resolution cross-attention and latent output.

    Stage 1: Pixel-resolution sparse geometric cross-attention gathers features
    from input view pixels (VAE-encoder-initialized source encoder) via
    geometric correspondences.
    Stage 2: Downsample to latent resolution via lossless PixelUnshuffle + 1x1.
    Stage 3: Latent-resolution output-to-output self-attention for inter-view
    consistency.
    Stage 4: Fuse with UNet x0 latent, refine via FiLM-conditioned ResBlocks.

    Produces a zero-initialized 4-channel latent residual correction.

    Args:
        bottleneck: Internal channel width.
        num_resblocks: Number of FiLMResBlocks (at latent resolution).
        num_attn_heads: Number of attention heads.
        temb_dim: Temporal embedding dimension from UNet (1280 for SD).
        use_geo_bias: Whether to add geometric weight bias in attention.
        clean_query: If True, query uses only pc_render + warp_mask; decoded_x0
            is fused after cross-attention (before downsample).
        dropout: Dropout rate for FiLMResBlocks.
        image_vae: Optional AutoencoderKL whose encoder's first down block is
            deep-copied to initialize the trainable source encoder. If None,
            falls back to a single 3x3 conv (used by standalone tests).
        vae_scale_factor: Spatial downsample factor from pixel to latent res.
            Also determines the expected resolution of `out_out_index_map`.
    """

    def __init__(
        self,
        bottleneck: int = 64,
        num_resblocks: int = 2,
        num_attn_heads: int = 4,
        temb_dim: int = 1280,
        use_geo_bias: bool = True,
        clean_query: bool = False,
        dropout: float = 0.0,
        image_vae=None,
        vae_scale_factor: int = 8,
    ):
        super().__init__()
        self.clean_query = clean_query
        self.vae_scale_factor = vae_scale_factor

        # --- Pixel-resolution attention stage ---

        if clean_query:
            # Query from clean signals only: pc_render[3] + warp_mask[1] = 4ch
            self.proj_query = nn.Conv2d(4, bottleneck, 3, padding=1)
            # Fuse decoded_x0 after cross-attention, before downsample
            self.proj_fuse_query = nn.Conv2d(bottleneck + 3, bottleneck, 3, padding=1)
        else:
            # Query: cat(decoded_x0[3], pc_render[3], warp_mask[1]) = 7ch
            self.proj_query = nn.Conv2d(7, bottleneck, 3, padding=1)

        # Source encoder: VAE-initialized first down block (conv_in + 2 resnets, no
        # downsampler), projected to bottleneck. Trainable. Falls back to a plain
        # 3x3 conv when image_vae is not provided (for standalone tests).
        if image_vae is not None:
            enc = image_vae.encoder
            self.source_conv_in = deepcopy(enc.conv_in)
            self.source_resnet1 = deepcopy(enc.down_blocks[0].resnets[0])
            self.source_resnet2 = deepcopy(enc.down_blocks[0].resnets[1])
            src_out_ch = self.source_resnet2.conv2.out_channels
            self.source_proj = nn.Conv2d(src_out_ch, bottleneck, 1)
            self._source_from_vae = True
            # Ensure all params are trainable
            for p in self.source_conv_in.parameters():
                p.requires_grad_(True)
            for p in self.source_resnet1.parameters():
                p.requires_grad_(True)
            for p in self.source_resnet2.parameters():
                p.requires_grad_(True)
        else:
            self.proj_source_input = nn.Conv2d(3, bottleneck, 3, padding=1)
            self._source_from_vae = False

        # Stage 1: input→output sparse cross-attention (pixel resolution)
        self.cross_attn = SparseGeometricCrossAttention(
            dim=bottleneck,
            num_heads=num_attn_heads,
            use_geo_bias=use_geo_bias,
        )
        nn.init.zeros_(self.cross_attn.to_out.weight)
        nn.init.zeros_(self.cross_attn.to_out.bias)

        # Stage 2: output→output sparse self-attention (runs at LATENT resolution,
        # after downsample). Same module class; only the resolution of its tokens
        # and the source index map change.
        self.view_self_attn = SparseGeometricCrossAttention(
            dim=bottleneck,
            num_heads=num_attn_heads,
            use_geo_bias=use_geo_bias,
        )
        nn.init.zeros_(self.view_self_attn.to_out.weight)
        nn.init.zeros_(self.view_self_attn.to_out.bias)

        # --- Downsample: pixel res → latent res via lossless PixelUnshuffle + 1x1 ---
        # Every pixel-res feature reaches the FiLM stack through a learned 64-way
        # linear combination per 8x8 block, instead of being discarded by strided convs.
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(vae_scale_factor),
            nn.Conv2d(bottleneck * (vae_scale_factor ** 2), bottleneck, 1),
        )

        # --- Latent fusion: cat(downsampled_features, x0_latent[4]) → bottleneck ---
        self.proj_fuse_latent = nn.Conv2d(bottleneck + 4, bottleneck, 3, padding=1)

        # --- FiLM ResBlocks at latent resolution ---
        self.resblocks = nn.ModuleList([
            FiLMResBlock(bottleneck, temb_dim=temb_dim, dropout=dropout)
            for _ in range(num_resblocks)
        ])

        # Output conv: ZERO-INITIALIZED so x0_delta starts at zero
        self.out_conv = nn.Conv2d(bottleneck, 4, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _encode_source(self, x: torch.Tensor) -> torch.Tensor:
        """Source encoder forward. (B*T_in, 3, H, W) -> (B*T_in, bottleneck, H, W)."""
        if self._source_from_vae:
            h = self.source_conv_in(x)
            h = self.source_resnet1(h, temb=None)
            h = self.source_resnet2(h, temb=None)
            h = self.source_proj(h)
            return h
        return self.proj_source_input(x)

    def forward(
        self,
        decoded_x0: torch.Tensor,       # (N, 3, H, W) — VAE-decoded UNet x0 prediction, [-1, 1]
        pc_render: torch.Tensor,         # (N, 3, H, W) — forward-warped source pixels, [0, 1]
        warp_mask: torch.Tensor,         # (N, 1, H, W) — warp coverage mask, [0, 1]
        input_images: torch.Tensor,      # (B*T_in, 3, H, W) — clean input view pixels, [-1, 1]
        x0_latent: torch.Tensor,         # (N, 4, H/8, W/8) — UNet x0 in latent space
        geo_index_map,                   # GeometricIndexMap at pixel resolution (in→out)
        out_out_index_map,               # GeometricIndexMap at LATENT resolution (out→out) or None
        temb: torch.Tensor,              # (N, 1280) — detached UNet temporal embedding
        T_out: int,
    ) -> torch.Tensor:
        """
        Returns:
            x0_delta: (N, 4, H/8, W/8) — latent residual correction.
        """
        N, _, H, W = decoded_x0.shape
        B = N // T_out

        # 1. Project query at pixel resolution
        if self.clean_query:
            query_2d = self.proj_query(torch.cat([pc_render, warp_mask], dim=1))
        else:
            query_2d = self.proj_query(torch.cat([decoded_x0, pc_render, warp_mask], dim=1))

        # Flatten to token format: (B*T_out, H*W, D)
        query = rearrange(query_2d, 'n d h w -> n (h w) d')

        # 2. Encode source input view pixels: (B*T_in, 3, H, W) -> (B, T_in*H*W, D)
        B_T_in = input_images.shape[0]
        T_in = B_T_in // B
        src_2d = self._encode_source(input_images)  # (B*T_in, D, H, W)
        src = rearrange(src_2d, '(b t) d h w -> b (t h w) d', b=B, t=T_in)

        # 3. Stage 1: input→output sparse cross-attention (pixel resolution)
        attn_out = self.cross_attn(
            query=query,
            source=src,
            indices=geo_index_map.indices,
            valid_mask=geo_index_map.valid_mask,
            geo_weights=geo_index_map.weights,
            T_out=T_out,
        )

        # Residual add
        h = query + attn_out

        # Reshape to spatial (pixel res): (B*T_out, D, H, W)
        h = rearrange(h, 'n (h w) d -> n d h w', h=H, w=W)

        # Fuse decoded_x0 before downsample (clean_query mode only)
        if self.clean_query:
            h = self.proj_fuse_query(torch.cat([h, decoded_x0], dim=1))

        # 4. Downsample: pixel res → latent res (PixelUnshuffle + 1x1)
        h = self.downsample(h)
        h_lat, w_lat = h.shape[-2], h.shape[-1]

        # 5. Stage 2: output→output sparse self-attention AT LATENT RESOLUTION
        if T_out > 1 and out_out_index_map is not None:
            h_tokens = rearrange(h, 'n d hh ww -> n (hh ww) d')
            h_global = rearrange(h_tokens, '(b t) n d -> b (t n) d', b=B, t=T_out)
            self_attn_out = self.view_self_attn(
                query=h_tokens,
                source=h_global,
                indices=out_out_index_map.indices,
                valid_mask=out_out_index_map.valid_mask,
                geo_weights=out_out_index_map.weights,
                T_out=T_out,
            )
            h_tokens = h_tokens + self_attn_out
            h = rearrange(h_tokens, 'n (hh ww) d -> n d hh ww', hh=h_lat, ww=w_lat)

        # 6. Fuse with UNet x0 latent
        h = self.proj_fuse_latent(torch.cat([h, x0_latent], dim=1))

        # 7. FiLM ResBlocks at latent resolution
        for resblock in self.resblocks:
            h = resblock(h, temb)

        # 8. Zero-init output conv -> 4ch latent delta
        return self.out_conv(h)


class PixelReconSideBranch(nn.Module):
    """Pixel-space reconstruction side-branch with two-stage sparse 3D attention.

    Operates at full image resolution (H×W) instead of latent resolution (H/8×W/8).
    Stage 1: input→output sparse cross-attention using source pixel colors.
    Stage 2: output→output sparse self-attention for inter-view consistency.
    Produces a zero-initialized pixel-space delta correction.

    Args:
        bottleneck: Internal channel width (D=32 for pixel-space).
        num_resblocks: Number of FiLMResBlocks.
        num_attn_heads: Number of attention heads.
        temb_dim: Temporal embedding dimension from UNet (1280 for SD).
        use_geo_bias: Whether to add geometric weight bias in attention.
    """

    def __init__(
        self,
        bottleneck: int = 32,
        num_resblocks: int = 2,
        num_attn_heads: int = 4,
        temb_dim: int = 1280,
        use_geo_bias: bool = True,
        clean_query: bool = False,
        dropout: float = 0.0,
        image_vae=None,
        use_plucker: bool = False,
        plucker_dim: int = 6,
    ):
        super().__init__()
        self.clean_query = clean_query
        self.use_plucker = use_plucker
        self.plucker_dim = plucker_dim

        if clean_query:
            # Query from clean signals only: pc_render[3] + warp_mask[1] = 4ch
            self.proj_query = nn.Conv2d(4, bottleneck, 3, padding=1)
            # Fuse decoded_x0 after cross-attention, before ResBlocks
            self.proj_fuse = nn.Conv2d(bottleneck + 3, bottleneck, 3, padding=1)
        else:
            # Original: cat(decoded_x0[3], pc_render[3], warp_mask[1]) = 7ch
            self.proj_query = nn.Conv2d(7, bottleneck, 3, padding=1)

        # Source encoder: optionally VAE-initialized (conv_in + 2 resnets, no
        # downsampler — stays at pixel resolution) + 1x1 projection to bottleneck.
        # Trainable. Falls back to a plain 3x3 conv when image_vae is not provided.
        if image_vae is not None:
            enc = image_vae.encoder
            self.source_conv_in = deepcopy(enc.conv_in)
            self.source_resnet1 = deepcopy(enc.down_blocks[0].resnets[0])
            self.source_resnet2 = deepcopy(enc.down_blocks[0].resnets[1])
            src_out_ch = self.source_resnet2.conv2.out_channels
            self.source_proj = nn.Conv2d(src_out_ch, bottleneck, 1)
            self._source_from_vae = True
            for p in self.source_conv_in.parameters():
                p.requires_grad_(True)
            for p in self.source_resnet1.parameters():
                p.requires_grad_(True)
            for p in self.source_resnet2.parameters():
                p.requires_grad_(True)
        else:
            self.proj_source_input = nn.Conv2d(3, bottleneck, 3, padding=1)
            self._source_from_vae = False

        if use_plucker:
            # Zero-init residual projections so init is bit-identical to use_plucker=False.
            # Query side: adds to proj_query output (bottleneck channels).
            self.proj_query_plucker = nn.Conv2d(plucker_dim, bottleneck, 3, padding=1)
            nn.init.zeros_(self.proj_query_plucker.weight)
            nn.init.zeros_(self.proj_query_plucker.bias)
            # Source side: adds to source_conv_in / proj_source_input output.
            if self._source_from_vae:
                src_in_ch = self.source_conv_in.out_channels
            else:
                src_in_ch = bottleneck
            self.proj_source_plucker = nn.Conv2d(plucker_dim, src_in_ch, 3, padding=1)
            nn.init.zeros_(self.proj_source_plucker.weight)
            nn.init.zeros_(self.proj_source_plucker.bias)

        # Stage 1: input→output sparse cross-attention
        self.cross_attn = SparseGeometricCrossAttention(
            dim=bottleneck,
            num_heads=num_attn_heads,
            use_geo_bias=use_geo_bias,
        )
        nn.init.zeros_(self.cross_attn.to_out.weight)
        nn.init.zeros_(self.cross_attn.to_out.bias)

        # Stage 2: output→output sparse self-attention
        self.view_self_attn = SparseGeometricCrossAttention(
            dim=bottleneck,
            num_heads=num_attn_heads,
            use_geo_bias=use_geo_bias,
        )
        nn.init.zeros_(self.view_self_attn.to_out.weight)
        nn.init.zeros_(self.view_self_attn.to_out.bias)

        # FiLM ResBlocks
        self.resblocks = nn.ModuleList([
            FiLMResBlock(bottleneck, temb_dim=temb_dim, dropout=dropout)
            for _ in range(num_resblocks)
        ])

        # Output conv: ZERO-INITIALIZED so pixel_delta starts at zero
        self.out_conv = nn.Conv2d(bottleneck, 3, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _encode_source(self, x: torch.Tensor, plucker: torch.Tensor = None) -> torch.Tensor:
        """Source encoder forward. (B*T_in, 3, H, W) -> (B*T_in, bottleneck, H, W).

        If `plucker` is provided and `use_plucker` is on, a zero-initialized residual
        projection from Plucker rays is added to the first conv's output.
        """
        if self._source_from_vae:
            h = self.source_conv_in(x)
            if self.use_plucker and plucker is not None:
                h = h + self.proj_source_plucker(plucker)
            h = self.source_resnet1(h, temb=None)
            h = self.source_resnet2(h, temb=None)
            h = self.source_proj(h)
            return h
        h = self.proj_source_input(x)
        if self.use_plucker and plucker is not None:
            h = h + self.proj_source_plucker(plucker)
        return h

    def forward(
        self,
        decoded_x0: torch.Tensor,       # (N, 3, H, W) — decoded UNet x0 prediction, [-1, 1]
        pc_render: torch.Tensor,         # (N, 3, H, W) — forward-warped source pixels, [0, 1]
        warp_mask: torch.Tensor,         # (N, 1, H, W) — warp coverage mask, [0, 1]
        input_images: torch.Tensor,      # (B*T_in, 3, H, W) — clean input view pixels, [-1, 1]
        geo_index_map,                   # GeometricIndexMap at pixel resolution
        out_out_index_map,               # GeometricIndexMap for output→output (or None)
        temb: torch.Tensor,             # (N, 1280) — detached UNet temporal embedding
        T_out: int,
        target_plucker: torch.Tensor = None,  # (N, 6, H, W) — per-pixel Plucker rays for target views
        source_plucker: torch.Tensor = None,  # (B*T_in, 6, H, W) — per-pixel Plucker rays for source views
    ) -> torch.Tensor:
        """
        Returns:
            pixel_delta: (N, 3, H, W) — residual correction to add to decoded_x0.
        """
        N, _, H, W = decoded_x0.shape
        B = N // T_out

        # 1. Project query
        if self.clean_query:
            query_2d = self.proj_query(torch.cat([pc_render, warp_mask], dim=1))
        else:
            query_2d = self.proj_query(torch.cat([decoded_x0, pc_render, warp_mask], dim=1))

        if self.use_plucker and target_plucker is not None:
            query_2d = query_2d + self.proj_query_plucker(target_plucker)

        # Flatten to token format: (B*T_out, H*W, D)
        query = rearrange(query_2d, 'n d h w -> n (h w) d')

        # 2. Encode source input view pixels: (B*T_in, 3, H, W) -> (B, T_in*H*W, D)
        B_T_in = input_images.shape[0]
        T_in = B_T_in // B
        src_2d = self._encode_source(input_images, plucker=source_plucker)  # (B*T_in, D, H, W)
        src = rearrange(src_2d, '(b t) d h w -> b (t h w) d', b=B, t=T_in)

        # 3. Stage 1: input→output sparse cross-attention
        attn_out = self.cross_attn(
            query=query,
            source=src,
            indices=geo_index_map.indices,
            valid_mask=geo_index_map.valid_mask,
            geo_weights=geo_index_map.weights,
            T_out=T_out,
        )

        # Residual add
        h = query + attn_out

        # 4. Stage 2: output→output sparse self-attention
        if T_out > 1 and out_out_index_map is not None:
            h_global = rearrange(h, '(b t) n d -> b (t n) d', b=B, t=T_out)
            self_attn_out = self.view_self_attn(
                query=h,
                source=h_global,
                indices=out_out_index_map.indices,
                valid_mask=out_out_index_map.valid_mask,
                geo_weights=out_out_index_map.weights,
                T_out=T_out,
            )
            h = h + self_attn_out

        # Reshape to spatial: (B*T_out, D, H, W)
        h = rearrange(h, 'n (h w) d -> n d h w', h=H, w=W)

        # Fuse decoded_x0 before ResBlocks (clean_query mode only)
        if self.clean_query:
            h = self.proj_fuse(torch.cat([h, decoded_x0], dim=1))

        # 5. FiLM ResBlocks with temb
        for resblock in self.resblocks:
            h = resblock(h, temb)

        # 6. Output correction
        return self.out_conv(h)
