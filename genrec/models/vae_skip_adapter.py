import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from peft import LoraConfig


class VAESkipAdapter(nn.Module):
    """
    Difix3D-style skip connections from VAE encoder (PC renders) to VAE decoder
    (UNet predictions). Captures features before each encoder down_block and
    injects them (via 1x1 projection) before each decoder up_block.

    Following Difix3D:
    - 4 skip levels (H, H/2, H/4, H/8)
    - Pre-block additive injection (add before up_block, not after)
    - 1x1 convs with no bias, near-zero init (1e-5)
    - Optional LoRA on all decoder convs + attention + skip convs (gaussian init)
    """

    def __init__(self, vae, lora_rank=0):
        super().__init__()

        block_out_channels = list(vae.config.block_out_channels)
        reversed_channels = list(reversed(block_out_channels))

        # Encoder features captured BEFORE each down_block:
        #   [0] before down_block[0]: conv_in output,      block_out_channels[0] at H
        #   [1] before down_block[1]: down_block[0] output, block_out_channels[0] at H/2
        #   [2] before down_block[2]: down_block[1] output, block_out_channels[1] at H/4
        #   [3] before down_block[3]: down_block[2] output, block_out_channels[2] at H/8
        enc_channels = [
            block_out_channels[0],   # conv_in output (128)
            block_out_channels[0],   # down_block[0] output (128)
            block_out_channels[1],   # down_block[1] output (256)
            block_out_channels[2],   # down_block[2] output (512)
        ]

        # Hidden state channels BEFORE each decoder up_block (pre-block injection target).
        # up_block[i] outputs reversed_channels[i]; mid_block outputs reversed_channels[0].
        # So the input to up_block[i] is reversed_channels[max(0, i-1)].
        # For SD2.1 [128,256,512,512] → reversed [512,512,256,128]:
        #   before up_block[0]: 512 (mid),  before up_block[1]: 512,
        #   before up_block[2]: 512,        before up_block[3]: 256
        dec_channels = [reversed_channels[max(0, i - 1)] for i in range(len(reversed_channels))]

        # Register skip convs on the decoder so LoRA can target them by name.
        # Reversed enc_channels to match the reversed usage order in encode_with_features:
        #   skip_conv_0 ← enc_feats[3] (deepest), skip_conv_3 ← enc_feats[0] (shallowest)
        # +1 input channel for the warp mask (valid PC coverage signal).
        reversed_enc_channels = list(reversed(enc_channels))
        for i, (enc_ch, dec_ch) in enumerate(zip(reversed_enc_channels, dec_channels)):
            conv = nn.Conv2d(enc_ch + 1, dec_ch, kernel_size=1, bias=False)
            nn.init.constant_(conv.weight, 1e-5)
            setattr(vae.decoder, f"skip_conv_{i}", conv)

        self.num_skips = len(enc_channels)
        self.gradient_checkpointing = False

        # Optional LoRA on the full decoder (convs + attention + skip convs)
        if lora_rank > 0:
            # Build explicit target list: all decoder convs, attention projections, and skip convs
            target_names = [
                "conv1", "conv2", "conv_in", "conv_out", "conv_shortcut", "conv",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            for i in range(self.num_skips):
                target_names.append(f"skip_conv_{i}")

            target_modules = []
            for name, _ in vae.decoder.named_modules():
                if any(name.endswith(t) for t in target_names):
                    target_modules.append(name)

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
                bias="none",
            )
            vae.add_adapter(lora_config, adapter_name="vae_skip")

    def encode_with_features(self, vae, x, warp_mask):
        """
        Run VAE encoder step-by-step, capturing features before each down_block.
        The encoder runs under no_grad (frozen).

        Args:
            vae: AutoencoderKL instance
            x: PC render images (N, 3, H, W) in [-1, 1]
            warp_mask: Binary warp mask (N, 1, H, W) — concatenated with encoder features

        Returns:
            (latents, skip_features)
            - latents: (N, 4, H/8, W/8) scaled by vae.config.scaling_factor
            - skip_features: list of 4 projected tensors matching decoder dims
        """
        encoder = vae.encoder
        decoder = vae.decoder

        with torch.no_grad():
            h = encoder.conv_in(x)

            enc_feats = []
            for down_block in encoder.down_blocks:
                enc_feats.append(h.detach())
                h = down_block(h)

            h = encoder.mid_block(h)
            h = encoder.conv_norm_out(h)
            h = encoder.conv_act(h)
            h = encoder.conv_out(h)

            moments = vae.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            latents = posterior.mode() * vae.config.scaling_factor

        # Project encoder features + warp mask through skip convs (WITH gradient)
        # Reversed order: enc_feats[3] (deepest) → skip_conv_0 (first decoder block)
        skip_features = []
        for i in range(self.num_skips):
            conv = getattr(decoder, f"skip_conv_{i}")
            feat = enc_feats[self.num_skips - 1 - i]  # reverse order
            mask_resized = F.interpolate(warp_mask, size=feat.shape[-2:], mode='nearest')
            feat_with_mask = torch.cat([feat, mask_resized], dim=1)
            skip_features.append(conv(feat_with_mask))

        return latents, skip_features

    def decode_with_skip(self, vae, z, skip_features):
        """
        Run VAE decoder step-by-step, injecting skip features BEFORE each
        up_block via element-wise addition (Difix3D-style pre-block injection).

        Args:
            vae: AutoencoderKL instance
            z: Latent tensor (N, 4, H/8, W/8) — NOT pre-divided by scaling_factor
            skip_features: list of 4 tensors from encode_with_features

        Returns:
            Decoded pixel tensor (N, 3, H, W) in [-1, 1]
        """
        decoder = vae.decoder

        if vae.post_quant_conv is not None:
            z = vae.post_quant_conv(z)

        h = decoder.conv_in(z)

        # mid (contains attention — skip checkpointing to avoid SDPA recompute mismatch)
        h = decoder.mid_block(h)
        upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype
        h = h.to(upscale_dtype)

        # Pre-block injection: add skip features BEFORE each up_block
        use_ckpt = torch.is_grad_enabled() and self.gradient_checkpointing
        for i, up_block in enumerate(decoder.up_blocks):
            if i < len(skip_features):
                h = h + skip_features[i]
            if use_ckpt:
                h = checkpoint(up_block, h, use_reentrant=False)
            else:
                h = up_block(h)

        # post-process
        h = decoder.conv_norm_out(h)
        h = decoder.conv_act(h)
        h = decoder.conv_out(h)

        return h
