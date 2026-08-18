"""
Post-hoc DINOv2 + DPT confidence prediction network.

Operates on the final decoded output (same distribution in training and eval)
rather than intermediate denoising steps. Uses frozen DINOv2 features from both
the generated image and the PC render, fused via a lightweight DPT decoder to
produce per-pixel confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for DPT fusion."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block with optional lateral connection and upsampling."""

    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True)

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(xs[1])
        y = self.resConfUnit2(y)
        if size is not None:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=True)
        else:
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = self.out_conv(y)
        return y


class ConfidencePredictor(nn.Module):
    """
    Post-hoc confidence predictor using frozen DINOv2 + trainable DPT decoder.

    Takes a generated image and the corresponding PC render, extracts multi-scale
    DINOv2 features from both, concatenates them per stage, and decodes via a
    DPT-style top-down fusion pyramid to produce per-pixel confidence in [0, 1].

    Args:
        dino_model_name: DINOv2 hub model name (default: ViT-B/14 with registers).
        extract_layers: Which transformer block indices to hook for multi-scale features.
        features: DPT decoder channel width.
    """

    def __init__(
        self,
        dino_model_name: str = "dinov2_vitb14_reg",
        extract_layers: List[int] = None,
        features: int = 256,
        use_pixel_branch: bool = False,
        pixel_features: int = 64,
    ) -> None:
        super().__init__()

        if extract_layers is None:
            extract_layers = [2, 5, 8, 11]
        self.extract_layers = extract_layers
        self.num_stages = len(extract_layers)

        # Frozen DINOv2 backbone (shared for both image and PC render)
        self.dino = torch.hub.load(
            "facebookresearch/dinov2", dino_model_name, pretrained=True
        )
        self.dino.requires_grad_(False)
        self.dino.eval()

        # DINOv2 ViT-B/14: embed_dim=768, patch_size=14
        dino_dim = self.dino.embed_dim
        self.patch_size = self.dino.patch_size
        self.num_register_tokens = getattr(self.dino, "num_register_tokens", 0)

        # Concatenated dim: image features + PC render features
        cat_dim = dino_dim * 2

        # Per-stage 1x1 projection from concatenated features to DPT width
        self.stage_projects = nn.ModuleList([
            nn.Conv2d(cat_dim, features, kernel_size=1) for _ in range(self.num_stages)
        ])

        # Spatial resize layers — build dynamically based on num_stages
        # Pattern: deepest stage gets /2, shallowest gets the largest upsample
        # For 4 stages: x4, x2, x1, /2  (classic DPT)
        # For 2 stages: x2, /2  (lighter)
        # For 3 stages: x4, x1, /2
        resize_list = []
        for i in range(self.num_stages):
            if i == self.num_stages - 1:
                # Deepest stage: downsample /2
                resize_list.append(nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1))
            elif i == self.num_stages - 2:
                # Second deepest: identity
                resize_list.append(nn.Identity())
            else:
                # Shallower stages: upsample with increasing factor
                scale = 2 ** (self.num_stages - 2 - i)
                resize_list.append(nn.ConvTranspose2d(features, features, kernel_size=scale, stride=scale, padding=0))
        self.resize_layers = nn.ModuleList(resize_list)

        # Pixel-level branch: lightweight CNN on |gen - pc| + obs_mask
        self.use_pixel_branch = use_pixel_branch
        self.pixel_features = pixel_features
        if use_pixel_branch:
            self.pixel_encoder = nn.Sequential(
                nn.Conv2d(4, pixel_features, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(pixel_features, pixel_features, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(pixel_features, pixel_features, 3, 1, 1),
                nn.ReLU(inplace=True),
            )

        # Stage adapter convolutions
        self.layer_rns = nn.ModuleList([
            nn.Conv2d(features, features, 3, 1, 1, bias=False) for _ in range(self.num_stages)
        ])

        # Top-down fusion chain: deepest has no residual, rest have lateral connections
        refinenets = []
        for i in range(self.num_stages):
            # Last (deepest) stage has no lateral residual
            refinenets.append(FeatureFusionBlock(features, has_residual=(i < self.num_stages - 1)))
        self.refinenets = nn.ModuleList(refinenets)

        # Confidence head: features (+ pixel_features) -> 1 channel
        head_in = features + pixel_features if use_pixel_branch else features
        head_mid = features // 2
        self.head = nn.Sequential(
            nn.Conv2d(head_in, head_mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, 1, 0),
        )

        # ImageNet normalization constants for DINOv2
        self.register_buffer(
            "dino_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "dino_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize_for_dino(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization. Input H, W must be divisible by patch_size."""
        return (x - self.dino_mean.to(x)) / self.dino_std.to(x)

    @torch.no_grad()
    def _extract_multiscale_features(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale patch features from DINOv2 via forward hooks.

        Args:
            img: (B, 3, H, W) normalized input in [0, 1].

        Returns:
            List of N feature maps (one per extract_layer), each (B, embed_dim, ph, pw).
        """
        x = self._normalize_for_dino(img)

        # Pad to nearest multiple of patch_size (DINOv2 requires this)
        _, _, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        features = {}
        hooks = []

        try:
            for layer_idx in self.extract_layers:
                def _hook_fn(module, input, output, idx=layer_idx):
                    features[idx] = output
                hook = self.dino.blocks[layer_idx].register_forward_hook(_hook_fn)
                hooks.append(hook)

            self.dino.forward_features(x)

            start_idx = 1 + self.num_register_tokens
            ph = x.shape[2] // self.patch_size
            pw = x.shape[3] // self.patch_size

            result = []
            for layer_idx in self.extract_layers:
                tokens = features[layer_idx][:, start_idx:, :]  # (B, N_patches, C)
                spatial = tokens.permute(0, 2, 1).reshape(-1, self.dino.embed_dim, ph, pw)
                result.append(spatial)
            return result
        finally:
            for hook in hooks:
                hook.remove()

    def forward(
        self, generated_img: torch.Tensor, pc_render: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict per-pixel log-variance (training) or confidence (eval).

        During training, returns raw log-variance s = log(sigma^2) for use with
        Gaussian NLL: loss = 0.5 * exp(-s) * (gen - gt)^2 + 0.5 * s.
        During eval, returns confidence = sigmoid(-s) in [0, 1].

        Args:
            generated_img: (B, 3, H, W) generated image in [0, 1].
            pc_render: (B, 3, H, W) point cloud render in [0, 1].
            output_size: Optional (H_out, W_out) for the output map.
                         Defaults to generated_img spatial dims.
            obs_mask: Optional (B, 1, H, W) observation mask from forward-warp
                      coverage. Used by the pixel branch if enabled.

        Returns:
            Training: log_var (B, 1, H_out, W_out) unbounded log-variance.
            Eval: confidence (B, 1, H_out, W_out) in [0, 1].
        """
        if output_size is None:
            output_size = generated_img.shape[-2:]

        # Extract multi-scale DINOv2 features from both inputs
        img_feats = self._extract_multiscale_features(generated_img)
        pc_feats = self._extract_multiscale_features(pc_render)

        # Concatenate per stage, project, and resize
        resized_feats = []
        for i in range(self.num_stages):
            cat = torch.cat([img_feats[i], pc_feats[i]], dim=1)  # (B, 2*C, ph, pw)
            proj = self.stage_projects[i](cat.to(self.stage_projects[i].weight.dtype))
            resized = self.resize_layers[i](proj)
            resized_feats.append(resized)

        # Stage adapter convolutions
        adapted = [self.layer_rns[i](resized_feats[i]) for i in range(self.num_stages)]

        # Top-down fusion: from deepest to shallowest
        # deepest stage (last index) has no lateral, rest fuse with lateral
        out = self.refinenets[-1](adapted[-1], size=adapted[-2].shape[2:] if self.num_stages > 1 else None)
        for i in range(self.num_stages - 2, 0, -1):
            out = self.refinenets[i](out, adapted[i], size=adapted[i - 1].shape[2:])
        if self.num_stages > 1:
            out = self.refinenets[0](out, adapted[0])

        # Upsample to output resolution
        out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=True)

        # Fuse pixel-level branch if enabled
        if self.use_pixel_branch:
            pixel_diff = (generated_img - pc_render).abs()  # (B, 3, H, W)
            if pixel_diff.shape[-2:] != output_size:
                pixel_diff = F.interpolate(pixel_diff, size=output_size, mode="bilinear", align_corners=False)
            if obs_mask is not None:
                if obs_mask.shape[-2:] != output_size:
                    obs_mask = F.interpolate(obs_mask, size=output_size, mode="bilinear", align_corners=False)
            else:
                obs_mask = torch.zeros(pixel_diff.shape[0], 1, *output_size,
                                       device=pixel_diff.device, dtype=pixel_diff.dtype)
            pixel_input = torch.cat([pixel_diff, obs_mask], dim=1)  # (B, 4, H, W)
            pixel_feat = self.pixel_encoder(pixel_input.to(out.dtype))
            out = torch.cat([out, pixel_feat], dim=1)

        log_var = self.head(out)  # raw log-variance, unbounded

        if self.training:
            return log_var
        else:
            return torch.exp(-log_var)
