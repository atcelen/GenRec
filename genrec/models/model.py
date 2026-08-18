import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import einops
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from typing import Optional, List, Union, Dict, Any, Tuple

from genrec.models.base_model import GenerativeReconstructionBaseModel
from genrec.option.options import Options
from genrec.utils.gram_loss import gram_loss
from genrec.utils.logger import setup_logger

logging = setup_logger(__name__)


def fill_holes_adaptive(rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Fill holes with per-image mean color of valid pixels.

    Args:
        rgb: (N, C, H, W) — RGB image.
        alpha: (N, 1, H, W) — binary/soft mask (1 = valid).

    Returns:
        Hole-filled RGB: (N, C, H, W).
    """
    valid_sum = (rgb * alpha).sum(dim=(2, 3), keepdim=True)
    valid_count = alpha.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    mean_color = valid_sum / valid_count
    return rgb * alpha + mean_color * (1 - alpha)


def compute_patch_trust_mask(alpha_mask: torch.Tensor, sigma: float = 3.0, vae_scale: int = 8) -> torch.Tensor:
    """Gaussian-blur binary alpha, then avg_pool to latent resolution.

    Args:
        alpha_mask: (N, 1, H, W) — binary/soft warp mask at image resolution.
        sigma: Gaussian blur sigma.
        vae_scale: VAE spatial downscale factor.

    Returns:
        Soft trust mask at latent resolution: (N, 1, H_lat, W_lat).
    """
    ks = 2 * math.ceil(3 * sigma) + 1
    blurred = TF.gaussian_blur(alpha_mask, kernel_size=ks, sigma=sigma)
    return F.avg_pool2d(blurred, kernel_size=vae_scale, stride=vae_scale)

class GenerativeReconstructionDiffusionModel(GenerativeReconstructionBaseModel):
    def __init__(self, opt: Options, mp_mode: str = "no"):
        super().__init__(opt, mp_mode)

        # Diffusion-specific: SNR config
        self.register_buffer('snr_gamma', torch.tensor(float(getattr(opt, "snr_gamma", 5.0))))
        self.register_buffer('snr_mid', torch.tensor(float(getattr(opt, "snr_mid", 5.0))))
        self.snr_weighting_mode = getattr(opt, "snr_weighting_mode", None)

        self.latent_channels = 4

        # Post-hoc confidence network (loaded separately, not part of diffusion training)
        self._posthoc_confidence_model = None
        self.use_posthoc_confidence = getattr(opt, "use_posthoc_confidence", False)

        # V-Prediction as flow config
        self.v_as_flow_prediction = getattr(opt, "v_as_flow_prediction", False)

        # Non-Gaussian flow source: use PC render latents as x_0 instead of noise
        self.init_from_pc_renders = getattr(opt, "init_from_pc_renders", False)

        # PC latent masking config
        self.mask_pc_latents = getattr(opt, "mask_pc_latents", True)

        # Observation mask conditioning: replace binary input/output mask with soft warp coverage
        self.use_observation_mask = getattr(opt, "use_observation_mask", False)

        # CFG conditioning dropout
        self.cfg_dropout_prob = getattr(opt, "cfg_dropout_prob", 0.0)

        # Dual FFN
        self.dual_ffn = getattr(opt, "dual_ffn", False)

        # Dual Reconstruction Pathway (side-branch with sparse geometric attention)
        self.dual_recon = getattr(opt, "dual_recon", False)
        self.recon_pixel_space = getattr(opt, "recon_pixel_space", False)
        self.recon_t_threshold = int(getattr(opt, "recon_t_threshold", 0))
        self.recon_last_step_only = bool(getattr(opt, "recon_last_step_only", False))
        self.train_t_clamp_ratio = getattr(opt, "train_t_clamp_ratio", None)
        self.recon_unrolled = bool(getattr(opt, "recon_unrolled", False))
        self.recon_unrolled_steps = int(getattr(opt, "recon_unrolled_steps", 15))
        self.recon_pixel_lpips_full = bool(getattr(opt, "recon_pixel_lpips_full", False))
        if self.dual_recon:
            self.recon_l1_weight = getattr(opt, "recon_l1_weight", 1.0)
            self.recon_K = getattr(opt, "recon_K", 8)
            self.recon_search_radius = getattr(opt, "recon_search_radius", 1.5)
            self.recon_occlusion_eps = getattr(opt, "recon_occlusion_eps", 0.02)
            self.recon_trust_sigma = getattr(opt, "recon_trust_sigma", 3.0)
            # Gaussian kernel for compute_observation_mask
            self.mask_blur_kernel = getattr(opt, "mask_blur_kernel", 21)
            self.mask_blur_sigma = getattr(opt, "mask_blur_sigma", 5.0)
            gauss_kernel = self._make_gaussian_kernel(self.mask_blur_kernel, self.mask_blur_sigma)
            self.register_buffer('_gauss_kernel', gauss_kernel, persistent=False)

            # Recon-branch-specific mask overrides (pixel branch loss path only).
            # None-defaults preserve backward compatibility: callers that don't set
            # these fall back to the global mask_blur_kernel / mask_blur_sigma above.
            _recon_mk = getattr(opt, "recon_mask_blur_kernel", None)
            _recon_ms = getattr(opt, "recon_mask_blur_sigma", None)
            self.recon_grad_weight = float(getattr(opt, "recon_grad_weight", 0.0))
            self.recon_delta_sparsity_weight = float(getattr(opt, "recon_delta_sparsity_weight", 0.0))
            self.recon_delta_sparsity_evidence_tau = float(getattr(opt, "recon_delta_sparsity_evidence_tau", 0.0))
            self.recon_gram_weight = float(getattr(opt, "recon_gram_weight", 0.0))
            if _recon_mk is not None or _recon_ms is not None:
                self.recon_mask_blur_kernel = int(_recon_mk) if _recon_mk is not None else self.mask_blur_kernel
                self.recon_mask_blur_sigma = float(_recon_ms) if _recon_ms is not None else self.mask_blur_sigma
                self.register_buffer(
                    '_gauss_kernel_recon',
                    self._make_gaussian_kernel(self.recon_mask_blur_kernel, self.recon_mask_blur_sigma),
                    persistent=False,
                )
            else:
                self.recon_mask_blur_kernel = self.mask_blur_kernel
                self.recon_mask_blur_sigma = self.mask_blur_sigma
                # No separate kernel buffer; _compute_pixel_obs_mask will fall through to _gauss_kernel.

            if self.recon_pixel_space:
                from genrec.models.recon_branch import PixelReconSideBranch
                recon_bottleneck = getattr(opt, "recon_bottleneck", 32)
                recon_branch_depth = getattr(opt, "recon_branch_depth", 2)
                recon_num_attn_heads = getattr(opt, "recon_num_attn_heads", 4)
                recon_use_geo_bias = getattr(opt, "recon_use_geo_bias", True)
                self.recon_use_plucker = bool(getattr(opt, "recon_use_plucker", False))
                self.recon_branch = PixelReconSideBranch(
                    bottleneck=recon_bottleneck,
                    num_resblocks=recon_branch_depth,
                    num_attn_heads=recon_num_attn_heads,
                    temb_dim=1280,
                    use_geo_bias=recon_use_geo_bias,
                    clean_query=getattr(opt, "recon_clean_query", False),
                    dropout=getattr(opt, "recon_dropout", 0.0),
                    image_vae=self.image_vae if getattr(opt, "recon_pixel_source_from_vae", False) else None,
                    use_plucker=self.recon_use_plucker,
                    plucker_dim=6,
                )
                self.recon_lpips_weight = getattr(opt, "recon_lpips_weight", 0.5)
                self._recon_lpips_net = None
                self._recon_vgg_net = None
                self.register_buffer(
                    '_imagenet_mean',
                    torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    '_imagenet_std',
                    torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.lpips_resize = getattr(opt, "lpips_resize", 256)
                logging.info(f"Pixel-space recon branch: D={recon_bottleneck}, K={self.recon_K}, heads={recon_num_attn_heads}")
            else:
                from genrec.models.recon_branch import ReconSideBranch
                recon_bottleneck = getattr(opt, "recon_bottleneck", 64)
                recon_branch_depth = getattr(opt, "recon_branch_depth", 2)
                recon_num_attn_heads = getattr(opt, "recon_num_attn_heads", 4)
                recon_use_geo_bias = getattr(opt, "recon_use_geo_bias", True)
                self.recon_use_plucker = False  # latent branch does not consume Plucker rays
                self.recon_branch = ReconSideBranch(
                    bottleneck=recon_bottleneck,
                    num_resblocks=recon_branch_depth,
                    num_attn_heads=recon_num_attn_heads,
                    temb_dim=1280,
                    use_geo_bias=recon_use_geo_bias,
                    clean_query=getattr(opt, "recon_clean_query", False),
                    dropout=getattr(opt, "recon_dropout", 0.0),
                    image_vae=self.image_vae,
                    vae_scale_factor=8,
                )
                self.recon_lpips_weight = getattr(opt, "recon_lpips_weight", 0.5)
                self._recon_lpips_net = None
                self.lpips_resize = getattr(opt, "lpips_resize", 256)
                logging.info(f"Latent-space recon branch: D={recon_bottleneck}, K={self.recon_K}, heads={recon_num_attn_heads}")

        # Region-Specific Loss config
        self.use_region_loss = getattr(opt, "use_region_loss", False)
        self.use_dino_loss = getattr(opt, "use_dino_loss", True)
        if self.use_region_loss:
            self.mask_blur_kernel = getattr(opt, "mask_blur_kernel", 21)
            self.mask_blur_sigma = getattr(opt, "mask_blur_sigma", 5.0)
            self.region_obs_l1_weight = getattr(opt, "region_obs_l1_weight", 1.0)
            self.region_obs_lpips_weight = getattr(opt, "region_obs_lpips_weight", 0.5)
            self.region_dino_weight = getattr(opt, "region_dino_weight", 0.1)
            self.dino_extract_layers = getattr(opt, "dino_extract_layers", [5, 8, 11])
            self.dino_pool_sizes = getattr(opt, "dino_pool_sizes", [1, 2, 4])
            self.dino_unobs_weight = getattr(opt, "dino_unobs_weight", 0.5)
            self.region_boundary_tv_weight = getattr(opt, "region_boundary_tv_weight", 0.01)
            self.region_loss_warmup_steps = getattr(opt, "region_loss_warmup_steps", 1000)
            # Soft gate params (backward compat: map old region_loss_t_threshold to center)
            old_threshold = getattr(opt, "region_loss_t_threshold", None)
            if old_threshold is not None and not hasattr(opt, "region_gate_center"):
                self.region_gate_center = old_threshold
            else:
                self.region_gate_center = getattr(opt, "region_gate_center", 0.4)
            self.region_gate_temperature = getattr(opt, "region_gate_temperature", 0.05)
            self.region_gate_compute_eps = getattr(opt, "region_gate_compute_eps", 0.01)
            self.lpips_resize = getattr(opt, "lpips_resize", 256)
            self.region_diffusion_downweight = getattr(opt, "region_diffusion_downweight", 0.0)
            self.use_latent_l1_loss = getattr(opt, "use_latent_l1_loss", False)
            self.latent_l1_weight = getattr(opt, "latent_l1_weight", 1.0)
            self.use_reliability_weighting = getattr(opt, "use_reliability_weighting", False)
            self.region_coord_l1_weight = getattr(opt, "region_coord_l1_weight", 0.0)

            # Precomputed Gaussian kernel for mask blurring (non-persistent buffer)
            gauss_kernel = self._make_gaussian_kernel(self.mask_blur_kernel, self.mask_blur_sigma)
            self.register_buffer('_gauss_kernel', gauss_kernel, persistent=False)
            self.register_buffer('_region_loss_step', torch.tensor(0, dtype=torch.long))

            # Lazy-init placeholders for LPIPS and DINO (initialized on first use)
            self._lpips_net = None
            self._dino_model = None

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel for convolution-based blurring."""
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        # Shape: (1, 1, K, K) for use with F.conv2d on single-channel input
        return kernel.unsqueeze(0).unsqueeze(0)

    def _compute_soft_gate(self, t_ratio: torch.Tensor) -> torch.Tensor:
        """Compute per-sample sigmoid soft gate: high at low noise, low at high noise."""
        return torch.sigmoid((self.region_gate_center - t_ratio) / self.region_gate_temperature)

    def _get_lpips_net(self):
        """Lazily initialize LPIPS network on first use."""
        if self._lpips_net is None:
            import kiui.lpips
            self._lpips_net = kiui.lpips.LPIPS(net='vgg').to(self.device)
            self._lpips_net.requires_grad_(False)
            self._lpips_net.eval()
        return self._lpips_net

    def _get_dino_model(self):
        """Lazily initialize DINOv2 ViT-B/14 feature extractor on first use."""
        if self._dino_model is None:
            self._dino_model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True
            ).to(self.device)
            self._dino_model.requires_grad_(False)
            self._dino_model.eval()
        return self._dino_model

    def load_posthoc_confidence_model(self, ckpt_path: str):
        """Load a trained post-hoc confidence predictor from checkpoint.

        Args:
            ckpt_path: Path to ConfidencePredictor checkpoint (.pth).
        """
        from genrec.models.confidence_model import ConfidencePredictor

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get("config", {})

        model = ConfidencePredictor(
            dino_model_name=config.get("dino_model", "dinov2_vitb14_reg"),
            extract_layers=config.get("extract_layers", [2, 5, 8, 11]),
            features=config.get("dpt_features", 256),
            use_pixel_branch=config.get("use_pixel_branch", False),
            pixel_features=config.get("pixel_features", 64),
        )
        model.to(self.device)

        # Load trainable weights (checkpoint excludes frozen dino.* keys)
        trainable_sd = ckpt["confidence_model"]
        missing, unexpected = model.load_state_dict(trainable_sd, strict=False)
        # Missing keys are expected for the frozen DINOv2 backbone (loaded from hub)
        logging.info(f"Loaded posthoc confidence model from {ckpt_path} "
                     f"(missing={len(missing)}, unexpected={len(unexpected)})")

        model.eval()
        model.requires_grad_(False)
        self._posthoc_confidence_model = model
        self.use_posthoc_confidence = True

    def compute_observation_mask(self, output_warp_masks: torch.Tensor) -> torch.Tensor:
        """Compute soft observation mask from binary warp masks.

        Args:
            output_warp_masks: (B, T_out, 1, H, W) binary forward-warp coverage masks.

        Returns:
            Soft mask at latent resolution: (B*T_out, 1, H_lat, W_lat).
        """
        N = output_warp_masks.shape[0] * output_warp_masks.shape[1]
        mask = output_warp_masks.reshape(N, *output_warp_masks.shape[2:]).float()

        # Gaussian blur for soft transitions (needed by compute_boundary_mask)
        # Use replicate padding so image borders aren't treated as unobserved
        gauss = self._gauss_kernel.to(mask.device, mask.dtype)
        blur_pad = self.mask_blur_kernel // 2
        mask = F.pad(mask, [blur_pad, blur_pad, blur_pad, blur_pad], mode='replicate')
        mask = F.conv2d(mask, gauss)

        # Downsample to latent resolution
        latent_h = mask.shape[-2] // 8
        latent_w = mask.shape[-1] // 8
        mask = F.interpolate(mask, size=(latent_h, latent_w), mode='bilinear', align_corners=False)

        return mask.clamp(0.0, 1.0)

    def _compute_pixel_obs_mask(self, output_warp_masks: torch.Tensor, use_recon_kernel: bool = False) -> torch.Tensor:
        """Compute soft observation mask at PIXEL resolution (no downsampling).

        Args:
            output_warp_masks: (B, T_out, 1, H, W) binary forward-warp coverage masks.
            use_recon_kernel: If True and a recon-specific Gaussian kernel exists
                (i.e. config sets recon_mask_blur_kernel/sigma), use it instead of
                the global kernel. Leaves non-recon call sites unchanged.

        Returns:
            Soft mask at pixel resolution: (B*T_out, 1, H, W).
        """
        N = output_warp_masks.shape[0] * output_warp_masks.shape[1]
        mask = output_warp_masks.reshape(N, *output_warp_masks.shape[2:]).float()

        _recon_gauss = getattr(self, '_gauss_kernel_recon', None)
        if use_recon_kernel and _recon_gauss is not None:
            gauss = _recon_gauss.to(mask.device, mask.dtype)
            kernel_size = self.recon_mask_blur_kernel
        else:
            gauss = self._gauss_kernel.to(mask.device, mask.dtype)
            kernel_size = self.mask_blur_kernel
        blur_pad = kernel_size // 2
        mask = F.pad(mask, [blur_pad, blur_pad, blur_pad, blur_pad], mode='replicate')
        mask = F.conv2d(mask, gauss)

        return mask.clamp(0.0, 1.0)

    def _get_recon_lpips_net(self):
        """Lazily initialize LPIPS network for pixel-space recon loss."""
        if self._recon_lpips_net is None:
            import kiui.lpips
            self._recon_lpips_net = kiui.lpips.LPIPS(net='vgg').to(self.device)
            self._recon_lpips_net.requires_grad_(False)
            self._recon_lpips_net.eval()
        return self._recon_lpips_net

    def _get_recon_vgg_net(self):
        """Lazily initialize frozen VGG19 feature extractor for Gram-matrix loss."""
        if self._recon_vgg_net is None:
            from torchvision.models import vgg19, VGG19_Weights
            net = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._recon_vgg_net = net.to(self.device)
        return self._recon_vgg_net

    def compute_reliability_mask(self, reliability_maps: torch.Tensor) -> torch.Tensor:
        """Compute soft reliability mask at latent resolution.

        Same blur + downsample pipeline as compute_observation_mask.

        Args:
            reliability_maps: (B, T_out, 1, H, W) per-pixel viewing-angle reliability [0, 1].

        Returns:
            Soft reliability mask at latent resolution: (B*T_out, 1, H_lat, W_lat).
        """
        N = reliability_maps.shape[0] * reliability_maps.shape[1]
        mask = reliability_maps.reshape(N, *reliability_maps.shape[2:]).float()

        gauss = self._gauss_kernel.to(mask.device, mask.dtype)
        blur_pad = self.mask_blur_kernel // 2
        mask = F.pad(mask, [blur_pad, blur_pad, blur_pad, blur_pad], mode='replicate')
        mask = F.conv2d(mask, gauss)

        latent_h = mask.shape[-2] // 8
        latent_w = mask.shape[-1] // 8
        mask = F.interpolate(mask, size=(latent_h, latent_w), mode='bilinear', align_corners=False)

        return mask.clamp(0.0, 1.0)

    def compute_boundary_mask(self, obs_mask: torch.Tensor, width: float = 0.15) -> torch.Tensor:
        """
        Compute boundary mask peaked at the observation/unobservation transition.

        Args:
            obs_mask: (N, 1, H, W) soft observation mask
            width: Controls the width of the boundary band

        Returns:
            boundary_mask: (N, 1, H, W) peaked at obs_mask=0.5
        """
        return torch.exp(-((obs_mask - 0.5) ** 2) / (2 * width ** 2))

    def _reconstruct_x0(self, pred: torch.Tensor,
                         noisy_targets: torch.Tensor, target_timesteps: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct predicted x0 latents from the model prediction.

        Works for both flow matching and standard v-prediction modes.
        The result is differentiable w.r.t. the UNet output.
        """
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")

        if pred_type == "v_prediction" and self.v_as_flow_prediction:
            # Flow prediction: pred ≈ x0 - noise, noisy = alpha_t * x0 + sigma_t * noise
            # => noise = x0 - pred, substituting:
            # noisy = (alpha_t + sigma_t) * x0 - sigma_t * pred
            # => x0 = (noisy + sigma_t * pred) / (alpha_t + sigma_t)
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=pred.dtype)
            alpha_bar = alphas_cumprod.gather(0, target_timesteps)
            alpha_t = alpha_bar.sqrt().view(-1, 1, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
            x0_pred = (noisy_targets + sigma_t * pred) / (alpha_t + sigma_t)
        elif pred_type == "v_prediction":
            # v-prediction: pred = alpha_t * noise - sigma_t * x0
            # noisy_targets = alpha_t * x0 + sigma_t * noise
            # x0 = alpha_t * noisy - sigma_t * pred
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=pred.dtype)
            alpha_bar = alphas_cumprod.gather(0, target_timesteps)
            alpha_t = alpha_bar.sqrt().view(-1, 1, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
            x0_pred = alpha_t * noisy_targets - sigma_t * pred
        elif pred_type == "epsilon":
            # epsilon prediction: x0 = (noisy - sigma_t * eps) / alpha_t
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=pred.dtype)
            alpha_bar = alphas_cumprod.gather(0, target_timesteps)
            alpha_t = alpha_bar.sqrt().view(-1, 1, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
            x0_pred = (noisy_targets - sigma_t * pred) / alpha_t.clamp(min=1e-8)
        else:
            # x0 prediction: pred is already x0
            x0_pred = pred

        return x0_pred

    @torch.no_grad()
    def _unrolled_denoise_for_recon(
        self,
        aligned_input_latents: torch.Tensor,
        aligned_input_rays: torch.Tensor,
        aligned_output_rays: torch.Tensor,
        aligned_input_pc: torch.Tensor,
        aligned_output_pc: torch.Tensor,
        obs_mask_channel: torch.Tensor,
        prompt_embeds: torch.Tensor,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        cross_attention_kwargs,
        real_bsz: int,
        T_in: int,
        T_out: int,
    ):
        """Partial N-step denoising to produce realistic x0 for recon branch.

        Runs the denoising loop from noise/PC-renders up to a randomly sampled
        stopping step (whose discrete timestep < recon_t_threshold), then
        reconstructs x0 from the UNet prediction at that point.

        Returns:
            x0_task0: (B*T_out, 4, H_lat, W_lat) — x0 for task-0 output views.
            temb: (B*T_out, 1280) — temporal embedding from the stopping step.
        """
        from copy import deepcopy
        import random

        _dtype = self.mixed_dtype
        num_steps = self.recon_unrolled_steps
        total_scenes = len(input_indices) + len(output_indices)
        num_targets = len(output_indices)
        lat_shape = aligned_input_latents.shape[2:]  # (H_lat, W_lat)

        # --- Scheduler setup (deepcopy to avoid mutating shared state) ---
        scheduler = deepcopy(self.val_scheduler)
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        # --- Determine stopping step ---
        # Map each scheduler step to its discrete timestep, find eligible
        # steps where t_discrete < recon_t_threshold.
        threshold = self.recon_t_threshold
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=self.device, dtype=_dtype,
        )
        alpha_t_all = alphas_cumprod.sqrt()
        sigma_t_all = (1.0 - alphas_cumprod).sqrt()
        dm_timesteps = self.noise_scheduler.timesteps

        if self.v_as_flow_prediction:
            dm_timesteps_continuous = alpha_t_all[dm_timesteps] / (
                alpha_t_all[dm_timesteps] + sigma_t_all[dm_timesteps]
            )
            step_t_discrete = []
            for si in range(len(timesteps)):
                t_fm = (1.0 / num_steps) * si
                td = torch.searchsorted(dm_timesteps_continuous, t_fm)
                td = td.clamp(0, len(dm_timesteps) - 1)
                step_t_discrete.append(dm_timesteps[td].item())
        else:
            step_t_discrete = [t.item() for t in timesteps]

        if threshold > 0:
            eligible = [si for si, td in enumerate(step_t_discrete) if td < threshold]
        else:
            eligible = []
        if not eligible:
            eligible = [len(timesteps) - 1]
        stop_step = random.choice(eligible)

        # --- Latent initialization (mirrors evaluate lines 2123-2133) ---
        if self.v_as_flow_prediction:
            if self.init_from_pc_renders:
                latents = aligned_output_pc[:, :self.latent_channels].clone().to(dtype=_dtype)
            else:
                latents = torch.randn(
                    (num_targets, self.latent_channels, *lat_shape),
                    device=self.device, dtype=_dtype,
                )
        else:
            latents = torch.randn(
                (num_targets, self.latent_channels, *lat_shape),
                device=self.device, dtype=_dtype,
            ) * scheduler.init_noise_sigma

        # --- Pre-build static conditioning (constant across steps) ---
        aligned_all_rays = torch.zeros(
            (total_scenes, aligned_input_rays.shape[1], *lat_shape),
            device=self.device, dtype=_dtype,
        )
        aligned_all_rays[input_indices] = aligned_input_rays.to(dtype=_dtype)
        aligned_all_rays[output_indices] = aligned_output_rays.to(dtype=_dtype)

        aligned_all_pc = torch.zeros(
            (total_scenes, aligned_input_pc.shape[1], *lat_shape),
            device=self.device, dtype=_dtype,
        )
        aligned_all_pc[input_indices] = aligned_input_pc.to(dtype=_dtype)
        aligned_all_pc[output_indices] = aligned_output_pc.to(dtype=_dtype)

        obs_mask = obs_mask_channel.to(dtype=_dtype)

        # --- Denoising loop ---
        _last_pred = None
        _last_scaled = None
        _last_t_discrete = None

        for step_idx, t in enumerate(timesteps):
            if step_idx > stop_step:
                break

            # Timestep mapping
            if self.v_as_flow_prediction:
                t_fm = (1.0 / num_steps) * step_idx
                td = torch.searchsorted(dm_timesteps_continuous, t_fm)
                td = td.clamp(0, len(dm_timesteps) - 1)
                t_discrete = dm_timesteps[td].to(device=self.device, dtype=torch.long)

                t_in = torch.full((total_scenes,), t_discrete.item(),
                                  device=self.device, dtype=torch.long)
                t_in[input_indices] = 0
            else:
                t_discrete = t
                t_in = torch.full((total_scenes,), t.item(),
                                  device=self.device, dtype=torch.long)
                t_in[input_indices] = 0

            # Build model input
            current_full_latents = torch.zeros(
                (total_scenes, self.latent_channels, *lat_shape),
                device=self.device, dtype=_dtype,
            )
            current_full_latents[input_indices] = aligned_input_latents.to(dtype=_dtype)

            if self.v_as_flow_prediction:
                at_val = alphas_cumprod[t_discrete].sqrt()
                st_val = (1.0 - alphas_cumprod[t_discrete]).sqrt()
                scaled_latents = (at_val + st_val) * latents
            else:
                scaled_latents = latents

            current_full_latents[output_indices] = scaled_latents.to(dtype=_dtype)

            model_in = torch.cat([
                current_full_latents, aligned_all_rays, obs_mask, aligned_all_pc,
            ], dim=1)

            # UNet forward
            unet_output = self.unet(
                model_in, timestep=t_in,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            model_pred = unet_output[output_indices, :self.latent_channels]

            # Save state from this step
            _last_pred = model_pred
            _last_scaled = scaled_latents
            _last_t_discrete = t_discrete

            # Scheduler step (skip on the stopping step)
            if step_idx < stop_step:
                if self.v_as_flow_prediction:
                    dt = 1.0 / num_steps
                    at = at_val.view(1, 1, 1, 1)
                    st = st_val.view(1, 1, 1, 1)
                    velocity = at * (scaled_latents - model_pred) - st * (scaled_latents + model_pred)
                    latents = scaled_latents / (at + st)
                    latents = latents + dt * velocity
                else:
                    latents = scheduler.step(model_pred, t, latents).prev_sample

        # --- Extract task-0 x0 from the stopping step ---
        task0_local = torch.arange(num_targets, device=self.device).view(
            real_bsz, self.num_tasks, T_out,
        )[:, 0, :].reshape(-1)

        pred_task0 = _last_pred[task0_local]
        lat_task0 = _last_scaled[task0_local]

        ab = alphas_cumprod[_last_t_discrete]
        at = ab.sqrt().view(1, 1, 1, 1)
        st = (1.0 - ab).sqrt().view(1, 1, 1, 1)

        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
        if pred_type == "v_prediction":
            x0_task0 = at * lat_task0 - st * pred_task0
        elif pred_type == "epsilon":
            x0_task0 = (lat_task0 - st * pred_task0) / at.clamp(min=1e-8)
        else:
            x0_task0 = pred_task0

        # Temporal embedding from the last UNet call
        temb = self.unet._cached_temb.detach()[output_indices][task0_local]

        return x0_task0, temb

    @staticmethod
    def _filter_geo_index_map(geo_index_map, keep_mask):
        """Filter a batched GeometricIndexMap by a boolean mask over the batch dim."""
        from genrec.utils.geometric_router import GeometricIndexMap
        return GeometricIndexMap(
            indices=geo_index_map.indices[keep_mask],
            distances=geo_index_map.distances[keep_mask],
            weights=geo_index_map.weights[keep_mask],
            valid_mask=geo_index_map.valid_mask[keep_mask],
            proj_xy=geo_index_map.proj_xy[keep_mask] if geo_index_map.proj_xy is not None else None,
            proj_depth=geo_index_map.proj_depth[keep_mask] if geo_index_map.proj_depth is not None else None,
        )

    def _compute_obs_l1(self, pred_pixels: torch.Tensor, gt_pixels: torch.Tensor,
                         pixel_obs_mask: torch.Tensor) -> torch.Tensor:
        """Observed region L1 loss in pixel space."""
        l1 = (pred_pixels - gt_pixels).abs() * pixel_obs_mask
        return l1.sum() / pixel_obs_mask.sum().clamp(min=1.0)

    def _compute_obs_lpips(self, pred_pixels: torch.Tensor, gt_pixels: torch.Tensor,
                            pixel_obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Observed region LPIPS loss in pixel space (image task only).

        Takes pre-decoded pixels and pixel-resolution mask. Masks unobserved regions
        to neutral gray, then computes LPIPS on the observed content.
        """
        lpips_net = self._get_lpips_net()

        # Fill unobserved regions with gray (0.5) so LPIPS only measures observed diffs
        pred_masked = pred_pixels * pixel_obs_mask + 0.5 * (1 - pixel_obs_mask)
        gt_masked = gt_pixels * pixel_obs_mask + 0.5 * (1 - pixel_obs_mask)

        # Resize for LPIPS efficiency
        if self.lpips_resize > 0:
            pred_masked = F.interpolate(pred_masked, size=(self.lpips_resize, self.lpips_resize),
                                         mode='bilinear', align_corners=False)
            gt_masked = F.interpolate(gt_masked, size=(self.lpips_resize, self.lpips_resize),
                                       mode='bilinear', align_corners=False)

        # LPIPS expects [-1, 1]
        loss = lpips_net(pred_masked * 2 - 1, gt_masked * 2 - 1).mean()
        return loss

    def _extract_dino_features(self, x: torch.Tensor, extract_layers: List[int]) -> List[torch.Tensor]:
        """
        Extract intermediate block outputs from DINOv2.

        Registers forward hooks on specified blocks, runs forward_features,
        and returns patch tokens (CLS + register tokens stripped).
        """
        dino = self._get_dino_model()
        features = {}
        hooks = []

        try:
            for layer_idx in extract_layers:
                def _hook_fn(module, input, output, idx=layer_idx):
                    features[idx] = output
                hook = dino.blocks[layer_idx].register_forward_hook(_hook_fn)
                hooks.append(hook)

            dino.forward_features(x)

            # Strip CLS + register tokens, return patch tokens only
            start_idx = 1 + dino.num_register_tokens
            result = []
            for layer_idx in extract_layers:
                result.append(features[layer_idx][:, start_idx:, :])
            return result
        finally:
            for hook in hooks:
                hook.remove()

    def _compute_dino_loss(self, pred_pixels: torch.Tensor, gt_pixels: torch.Tensor) -> torch.Tensor:
        """Per-patch DINO cosine similarity loss across multiple layers."""
        dino_size = (518, 518)
        pred_resized = F.interpolate(pred_pixels, size=dino_size, mode='bilinear', align_corners=False)
        gt_resized = F.interpolate(gt_pixels, size=dino_size, mode='bilinear', align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=pred_resized.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=pred_resized.dtype).view(1, 3, 1, 1)
        pred_normed = (pred_resized - mean) / std
        gt_normed = (gt_resized - mean) / std

        pred_features = self._extract_dino_features(pred_normed, self.dino_extract_layers)
        with torch.no_grad():
            gt_features = self._extract_dino_features(gt_normed, self.dino_extract_layers)

        grid_size = 37  # 518 / 14

        total_loss = 0.0
        for pred_feat, gt_feat in zip(pred_features, gt_features):
            # pred_feat, gt_feat: (B, N, C)
            cos_sim = F.cosine_similarity(pred_feat, gt_feat, dim=-1)  # (B, N)
            total_loss = total_loss + (1.0 - cos_sim).mean()

        return total_loss / len(self.dino_extract_layers)

    def _compute_boundary_tv(self, pred: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        """
        Boundary total variation loss in pixel space.

        Encourages smooth transitions at the observed/unobserved boundary.
        """
        # Horizontal differences
        diff_h = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
        boundary_h = boundary_mask[:, :, 1:, :]
        tv_h = (diff_h * boundary_h).mean()

        # Vertical differences
        diff_v = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
        boundary_v = boundary_mask[:, :, :, 1:]
        tv_v = (diff_v * boundary_v).mean()

        return tv_h + tv_v

    @torch.no_grad()
    def _debug_visualize_region_masks(
        self,
        output_pc_renders: torch.Tensor,
        output_coord_map: torch.Tensor,
        obs_mask: torch.Tensor,
        pixel_obs_mask: torch.Tensor,
        gt_pixels: torch.Tensor,
        pred_pixels: torch.Tensor,
        pixel_boundary_mask: torch.Tensor,
    ):
        """Save debug grid: pc_render | coord_map | obs_mask | boundary | gt | pred | masked_pred."""
        import torchvision.utils as vutils
        import os

        step = self._region_loss_step.item()
        out_dir = os.path.join("experiments", "debug_region_loss")
        os.makedirs(out_dir, exist_ok=True)

        # Flatten pc_renders: (B, T_out, 3, H, W) -> (N, 3, H, W)
        N = output_pc_renders.shape[0] * output_pc_renders.shape[1]
        pc_flat = output_pc_renders.reshape(N, *output_pc_renders.shape[2:])
        # Normalize pc renders to [0,1] for visualization
        pc_vis = pc_flat.abs()
        pc_vis = pc_vis / pc_vis.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)

        # Flatten coord maps: (B, T_out, 3, H, W) -> (N, 3, H, W)
        coord_flat = output_coord_map.reshape(N, *output_coord_map.shape[2:])
        # Take first 3 channels (XYZ) and normalize per-sample to [0,1]
        coord_vis = coord_flat[:, :3]
        coord_min = coord_vis.amin(dim=(1, 2, 3), keepdim=True)
        coord_max = coord_vis.amax(dim=(1, 2, 3), keepdim=True)
        coord_vis = (coord_vis - coord_min) / (coord_max - coord_min).clamp(min=1e-6)

        H, W = gt_pixels.shape[-2:]

        # Upsample obs_mask (latent res) to pixel res, expand to 3ch for grid
        obs_vis = F.interpolate(obs_mask, size=(H, W), mode='bilinear', align_corners=False)
        obs_vis = obs_vis.expand(-1, 3, -1, -1)  # (N, 3, H, W)

        # Boundary mask to 3ch
        bnd_vis = pixel_boundary_mask.expand(-1, 3, -1, -1)

        # Resize pc_vis and coord_vis to match pixel resolution
        pc_vis = F.interpolate(pc_vis, size=(H, W), mode='bilinear', align_corners=False)
        coord_vis = F.interpolate(coord_vis, size=(H, W), mode='bilinear', align_corners=False)

        # Masked prediction: obs regions normal, unobs regions red-tinted
        unobs_mask = 1.0 - pixel_obs_mask
        masked_pred = pred_pixels.clone()
        masked_pred[:, 0:1] = masked_pred[:, 0:1] + 0.3 * unobs_mask  # red tint on unobs
        masked_pred = masked_pred.clamp(0, 1)

        # Build rows: each sample gets one row
        # Columns: pc_render | coord_map | obs_mask | boundary | gt | pred | masked_pred
        num_samples = min(N, 4)  # Cap at 4 rows
        rows = []
        for i in range(num_samples):
            rows.extend([pc_vis[i], coord_vis[i], obs_vis[i], bnd_vis[i],
                         gt_pixels[i], pred_pixels[i], masked_pred[i]])

        grid = vutils.make_grid(rows, nrow=7, padding=2, normalize=False)
        out_path = os.path.join(out_dir, f"step_{step:06d}.png")
        vutils.save_image(grid, out_path)
        logging.info(f"[DEBUG] Region loss visualization saved to {out_path}")

    def _assemble_region_loss(self, diffusion_mse: torch.Tensor,
                                obs_l1: torch.Tensor, obs_lpips: torch.Tensor,
                                dino_loss: torch.Tensor, boundary_tv: torch.Tensor,
                                coord_diffusion_mse: Optional[torch.Tensor] = None,
                                coord_obs_l1: Optional[torch.Tensor] = None,
                                gate: float = 1.0,
                                diffusion_weight: float = 1.0) -> dict:
        """
        Assemble total loss with warmup and soft gate scaling.

        Returns dict with individual loss components and total.
        """
        # Linear warmup
        step = self._region_loss_step.float()
        warmup = (step / max(self.region_loss_warmup_steps, 1)).clamp(0.0, 1.0)

        scale = warmup * gate

        weighted_obs_l1 = self.region_obs_l1_weight * obs_l1
        weighted_obs_lpips = self.region_obs_lpips_weight * obs_lpips
        weighted_dino = self.region_dino_weight * dino_loss
        weighted_boundary_tv = self.region_boundary_tv_weight * boundary_tv
        weighted_coord_obs_l1 = self.region_coord_l1_weight * coord_obs_l1 if coord_obs_l1 is not None else 0.0

        total = diffusion_mse + scale * (weighted_obs_l1 + weighted_obs_lpips
                                          + weighted_dino + weighted_boundary_tv
                                          + weighted_coord_obs_l1)

        if coord_diffusion_mse is not None:
            total = total + coord_diffusion_mse

        result = {
            'total': total,
            'diffusion_mse': diffusion_mse.detach(),
            'obs_l1': obs_l1.detach(),
            'obs_lpips': obs_lpips.detach(),
            'dino_loss': dino_loss.detach(),
            'boundary_tv': boundary_tv.detach(),
            'region_warmup': warmup.detach(),
            'region_gate': gate,
            'diffusion_weight': diffusion_weight,
        }

        if coord_diffusion_mse is not None:
            result['coord_diffusion_mse'] = coord_diffusion_mse.detach()
        if coord_obs_l1 is not None:
            result['coord_obs_l1'] = coord_obs_l1.detach()

        # Increment step counter
        self._region_loss_step += 1

        return result

    def _create_scheduler(self, opt: Options) -> tuple:
        ckpt = getattr(opt, "pretrained_model_name_or_path", opt.spatialgen_ckpt)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
            clip_sample=False
        )

        val_scheduler = DPMSolverMultistepScheduler.from_pretrained(ckpt, subfolder="scheduler")
        val_scheduler = DPMSolverMultistepScheduler.from_config(
            val_scheduler.config,
            timestep_spacing="trailing",
            steps_offset=1,
            clip_sample=False,
            clip_sample_range=1.0,
            prediction_type="v_prediction",
            use_karras_sigmas=True,
        )
        return noise_scheduler, val_scheduler

    def init_dual_ffn_from_pretrained(self):
        """Initialize both dual FFN branches from pretrained weights.

        Called when loading a non-dual-FFN checkpoint into a dual-FFN model.
        Copies pretrained ff -> ff_halluc so both branches start from pretrained weights.
        """
        from genrec.models.mvmm_attention import BasicMVMMTransformerBlock
        count = 0
        for name, module in self.named_modules():
            if isinstance(module, BasicMVMMTransformerBlock) and module.dual_ffn:
                # Copy pretrained ff -> ff_halluc (both branches keep pretrained weights)
                module.ff_halluc.load_state_dict(module.ff.state_dict())
                # Copy norm3 -> norm3_halluc
                if hasattr(module, 'norm3_halluc'):
                    module.norm3_halluc.load_state_dict(module.norm3.state_dict())
                count += 1
        if count > 0:
            logging.info(f"Dual-FFN init: copied pretrained ff->ff_halluc in {count} blocks")

    def forward(
        self,
        input_imgs: torch.FloatTensor,
        input_rays: torch.FloatTensor,
        input_coord_map: torch.FloatTensor,
        input_pc_renders: torch.FloatTensor,
        output_imgs: torch.FloatTensor,
        output_rays: torch.FloatTensor,
        output_coord_map: torch.FloatTensor,
        output_pc_renders: torch.FloatTensor,
        output_coord_target: Optional[torch.FloatTensor] = None,
        input_warp_masks: Optional[torch.FloatTensor] = None,
        output_warp_masks: Optional[torch.FloatTensor] = None,
        T_in: int = 3,
        T_out: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 1.0,
        do_classifier_free_guidance: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        geo_index_map=None,
        out_out_index_map=None,
        input_dyn_masks: Optional[torch.FloatTensor] = None,  # (B, T_in, 1, H, W) 1=dynamic
        output_dyn_masks: Optional[torch.FloatTensor] = None,  # (B, T_out, 1, H, W) 1=dynamic
        output_reliability_maps: Optional[torch.FloatTensor] = None,  # (B, T_out, 1, H, W) viewing-angle reliability [0, 1]
        output_depth: Optional[torch.FloatTensor] = None,  # (B, T_out, H, W) per-pixel depth for inverse-depth loss weighting
    ):
        # 1. Setup & Checks
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        real_bsz = input_imgs.shape[0]
        num_sample_views = T_in + T_out
        total_scenes = real_bsz * num_sample_views * self.num_tasks

        # 2. Indices Setup
        # Structure: (Batch, Task, View)
        all_indices = torch.arange(total_scenes, device=self.device).long()
        indices_grid = all_indices.view(real_bsz, self.num_tasks, num_sample_views)

        input_indices = indices_grid[:, :, :T_in].reshape(-1)
        output_indices = indices_grid[:, :, T_in:].reshape(-1)

        # 3. Text Embeddings
        if prompt_embeds is None:
            prompt_embeds = self._encode_text(self.default_text_prompt)
        prompt_embeds = prompt_embeds.repeat(real_bsz * self.num_tasks * num_sample_views, 1, 1)

        # 4. Ray Encoding
        input_ray_embeds = self._process_rays(input_rays)
        output_ray_embeds = self._process_rays(output_rays)

        # 5. Helper Functions
        def _flatten(x):
            return einops.rearrange(x, "b t c h w -> (b t) c h w")

        def _interleave_tasks(latent_list, bsz, num_views):
            """
            Rearranges list of (B*T, ...) task tensors into (B*Tasks*T, ...)
            aligned with indices_grid (Batch -> Task -> View).
            """
            # Unflatten each task: (B*T, ...) -> (B, T, ...)
            unflattened = [
                einops.rearrange(t, '(b t) ... -> b t ...', b=bsz, t=num_views)
                for t in latent_list
            ]
            # Stack tasks: (B, Tasks, T, ...)
            stacked = torch.stack(unflattened, dim=1)
            # Flatten: (B*Tasks*T, ...)
            return einops.rearrange(stacked, 'b task t ... -> (b task t) ...')

        def _repeat_shared(latent, bsz, num_tasks, num_views):
            """
            Repeats shared data (rays, pc) for every task.
            Input: (B*T, ...) -> Output: (B*Tasks*T, ...)
            """
            # Unflatten: (B, T, ...)
            unflattened = einops.rearrange(latent, '(b t) ... -> b t ...', b=bsz, t=num_views)
            # Repeat for tasks: (B, Tasks, T, ...)
            repeated = unflattened.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1)
            # Flatten: (B*Tasks*T, ...)
            return einops.rearrange(repeated, 'b task t ... -> (b task t) ...')

        # Combine dynamic masks with warp masks: dynamic pixels → unobserved
        if output_dyn_masks is not None:
            output_static = 1.0 - output_dyn_masks  # (B, T_out, 1, H, W) 1=static=valid
            if output_warp_masks is not None:
                output_warp_masks = output_warp_masks * output_static
            else:
                output_warp_masks = output_static
        if input_dyn_masks is not None:
            input_static = 1.0 - input_dyn_masks  # (B, T_in, 1, H, W) 1=static=valid
            if input_warp_masks is not None:
                input_warp_masks = input_warp_masks * input_static
            else:
                input_warp_masks = input_static

        # 6. Latent Preparation
        # Define Task mapping based on num_tasks
        # Task 0 is always Images. Task 1 is Coord Map (if num_tasks > 1).
        raw_inputs = [input_imgs]
        raw_outputs = [output_imgs]
        if self.num_tasks > 1:
            raw_inputs.append(input_coord_map)
            # Use all-view coord maps for supervision if available (no black holes)
            raw_outputs.append(output_coord_target if output_coord_target is not None else output_coord_map)

        # Encode all tasks
        input_latents_list = []
        output_latents_list = []

        vaes = [self.image_vae]
        if self.num_tasks > 1:
            vaes.append(self.depth_vae)

        for task_idx, (raw_in, raw_out) in enumerate(zip(raw_inputs, raw_outputs)):
            task_vae = vaes[task_idx]
            input_latents_list.append(self.prepare_img_latents(
                _flatten(raw_in), real_bsz, self.mixed_dtype, self.device, generator, vae=task_vae
            ))
            output_latents_list.append(self.prepare_img_latents(
                _flatten(raw_out), real_bsz, self.mixed_dtype, self.device, generator, vae=task_vae
            ))

        # Encode PC Latents — Task-specific conditioning
        # Task 0 (RGB): PC renders as conditioning
        input_pc_latents = self.prepare_img_latents(
            _flatten(input_pc_renders), real_bsz, self.mixed_dtype, self.device, generator
        )
        if self.use_vae_skip and output_warp_masks is not None:
            output_pc_latents = self.encode_pc_with_features(
                _flatten(output_pc_renders), _flatten(output_warp_masks)
            )
        else:
            output_pc_latents = self.prepare_img_latents(
                _flatten(output_pc_renders), real_bsz, self.mixed_dtype, self.device, generator
            )

        # Mask PC latents in unobserved regions using warp masks
        out_mask_soft = None  # Soft [0,1] mask at latent res, reused for observation mask channel
        if self.mask_pc_latents and output_warp_masks is not None:
            latent_h, latent_w = output_pc_latents.shape[-2:]
            out_mask_soft = F.interpolate(
                _flatten(output_warp_masks), size=(latent_h, latent_w), mode='area'
            )
            in_mask_soft = F.interpolate(
                _flatten(input_warp_masks), size=(latent_h, latent_w), mode='area'
            )
            # Threshold raw warp mask at full res, then downsample to latent res
            out_mask = F.interpolate(
                (_flatten(output_warp_masks) > 0.5).float(), size=(latent_h, latent_w), mode='area'
            )
            out_mask = (out_mask > 0.5).to(output_pc_latents.dtype)
            in_mask = F.interpolate(
                (_flatten(input_warp_masks) > 0.5).float(), size=(latent_h, latent_w), mode='area'
            )
            in_mask = (in_mask > 0.5).to(input_pc_latents.dtype)

            output_pc_latents = self._mask_latents(output_pc_latents, out_mask)
            input_pc_latents = self._mask_latents(input_pc_latents, in_mask)

        input_cond_list = [input_pc_latents]
        output_cond_list = [output_pc_latents]

        if self.num_tasks > 1:
            # Task 1 (Coords): loo_pred coord maps encoded by depth VAE as conditioning
            # When coord_fp32_vae is enabled, disable autocast to keep depth VAE in float32
            # (mirrors the evaluate path which already disables autocast for coord VAE)
            _coord_vae_dtype = torch.float32 if self.coord_fp32_vae else self.mixed_dtype
            with torch.amp.autocast(device_type=self.device.type, enabled=not self.coord_fp32_vae):
                input_coord_cond = self.prepare_img_latents(
                    _flatten(input_coord_map), real_bsz, _coord_vae_dtype, self.device, generator, vae=self.depth_vae
                )
                if self.use_depth_vae_skip and output_warp_masks is not None:
                    output_coord_cond = self.encode_coord_with_features(
                        _flatten(output_coord_map), _flatten(output_warp_masks), dtype=_coord_vae_dtype
                    )
                else:
                    output_coord_cond = self.prepare_img_latents(
                        _flatten(output_coord_map), real_bsz, _coord_vae_dtype, self.device, generator, vae=self.depth_vae
                    )
            # Cast latents back to mixed precision for UNet
            if self.coord_fp32_vae:
                input_coord_cond = input_coord_cond.to(self.mixed_dtype)
                output_coord_cond = output_coord_cond.to(self.mixed_dtype)
            # Coord maps share the same forward-warp coverage as PC renders
            if self.mask_pc_latents and output_warp_masks is not None:
                output_coord_cond = self._mask_coord_latents(output_coord_cond, out_mask)
                input_coord_cond = self._mask_coord_latents(input_coord_cond, in_mask)
            input_cond_list.append(input_coord_cond)
            output_cond_list.append(output_coord_cond)

        # ===== CFG Conditioning Dropout =====
        cfg_drop_mask = None
        if self.training and self.cfg_dropout_prob > 0.0:
            cfg_drop_mask = torch.rand(real_bsz, device=self.device) < self.cfg_dropout_prob

            if cfg_drop_mask.any():
                # Expand to (B*T,) for input/output views
                drop_in = cfg_drop_mask[:, None].expand(-1, T_in).reshape(-1)
                drop_out = cfg_drop_mask[:, None].expand(-1, T_out).reshape(-1)
                drop_in_4d = drop_in[:, None, None, None]
                drop_out_4d = drop_out[:, None, None, None]

                # Build null PC latents
                if self.use_learned_null_token:
                    null_pc_in = self.pc_null_token.to(dtype=input_cond_list[0].dtype).expand_as(input_cond_list[0])
                    null_pc_out = self.pc_null_token.to(dtype=output_cond_list[0].dtype).expand_as(output_cond_list[0])
                else:
                    null_pc_in = torch.zeros_like(input_cond_list[0])
                    null_pc_out = torch.zeros_like(output_cond_list[0])

                input_cond_list[0] = torch.where(drop_in_4d, null_pc_in, input_cond_list[0])
                output_cond_list[0] = torch.where(drop_out_4d, null_pc_out, output_cond_list[0])

                # Replace coord latents if present
                if len(input_cond_list) > 1:
                    if self.use_learned_null_token:
                        null_coord_in = self.coord_null_token.to(dtype=input_cond_list[1].dtype).expand_as(input_cond_list[1])
                        null_coord_out = self.coord_null_token.to(dtype=output_cond_list[1].dtype).expand_as(output_cond_list[1])
                    else:
                        null_coord_in = torch.zeros_like(input_cond_list[1])
                        null_coord_out = torch.zeros_like(output_cond_list[1])
                    input_cond_list[1] = torch.where(drop_in_4d, null_coord_in, input_cond_list[1])
                    output_cond_list[1] = torch.where(drop_out_4d, null_coord_out, output_cond_list[1])
            else:
                cfg_drop_mask = None  # No samples were actually dropped

        # Flatten Ray embeds (assuming they kept dims)
        input_ray_embeds = _flatten(input_ray_embeds)
        output_ray_embeds = _flatten(output_ray_embeds)

        # 7. Timestep Sampling
        if self.v_as_flow_prediction:
            # Flow matching: sample continuous timesteps in [0, 1]
            # Use logit-normal sampling for better training dynamics
            u = torch.randn(real_bsz, device=self.device) # mean=0.5, std=1.0 in logit space
            t_continuous = torch.sigmoid(u)  # Logit-normal distributed in [0, 1]

            # Build scheduler mapping tables up-front so the optional clamp override
            # can consult them; independent of t_continuous so the reorder is a no-op.
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=self.mixed_dtype)
            alpha_t = alphas_cumprod.sqrt()
            sigma_t = (1.0 - alphas_cumprod).sqrt()

            dm_timesteps = self.noise_scheduler.timesteps

            # fp32 mapping table for the auto-derive lookup and searchsorted. bf16
            # spacing near the clamp boundary (~0.004) collapses adjacent timesteps
            # onto the same value and would yield wrong-index ties.
            _ac_f32 = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=torch.float32)
            _at_f32 = _ac_f32.sqrt()
            _st_f32 = (1.0 - _ac_f32).sqrt()
            dm_timesteps_continuous = _at_f32[dm_timesteps] / (_at_f32[dm_timesteps] + _st_f32[dm_timesteps])

            # When UNet is frozen, cap t_continuous so the region-loss gate stays
            # active (t_ratio = 1 - t_continuous must stay below max_t_ratio).
            max_t_ratio = None
            if self.freeze_unet and self.use_region_loss and not getattr(self, 'use_latent_l1_loss', False):
                max_t_ratio = (self.region_gate_center
                               + self.region_gate_temperature
                               * math.log((1 - self.region_gate_compute_eps)
                                          / self.region_gate_compute_eps))

            # Optional override: decouple t-sampling range from the region gate.
            # "auto" derives from recon_t_threshold via the scheduler's FM mapping.
            override = self.train_t_clamp_ratio
            if override is not None:
                if isinstance(override, str) and override == "auto":
                    if self.recon_t_threshold > 0:
                        mask = (dm_timesteps < self.recon_t_threshold)
                        nonzero = mask.nonzero()
                        if nonzero.numel() > 0:
                            first_below = nonzero[0].item()
                            min_t_continuous = dm_timesteps_continuous[first_below].item()
                            max_t_ratio = 1.0 - min_t_continuous
                else:
                    max_t_ratio = float(override)

            # Resample from a logit-normal TRUNCATED to t_continuous > min_tc instead
            # of clamping (which piles mass at the boundary and collapses t_discrete
            # onto a single value). Inverse CDF of N(0,1) conditioned on u > u_min
            # preserves the logit-normal shape above the threshold.
            if max_t_ratio is not None:
                min_tc = float(1.0 - max_t_ratio)
                u_min = math.log(min_tc / (1.0 - min_tc))
                cdf_min = 0.5 * (1.0 + math.erf(u_min / math.sqrt(2.0)))
                xi = torch.rand(real_bsz, device=self.device)
                p = (cdf_min + xi * (1.0 - cdf_min)).clamp(1e-6, 1.0 - 1e-6)
                u = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
                t_continuous = torch.sigmoid(u)

            t_discrete = torch.searchsorted(dm_timesteps_continuous, t_continuous).cpu().long()
            t_discrete = t_discrete.clamp(0, len(dm_timesteps) - 1)
            t_discrete = dm_timesteps[t_discrete].to(device=self.device, dtype=torch.long)

            t = t_discrete.unsqueeze(1).unsqueeze(2).repeat(1, self.num_tasks, num_sample_views).reshape(-1)
            t[input_indices] = 0
        else:
            T_max = self.noise_scheduler.config.num_train_timesteps
            effective_T_max = T_max
            # When UNet is frozen, region losses through the VAE decoder are the
            # only gradient source.  Clamp timesteps so the region-loss gate
            # stays above region_gate_compute_eps (avoids the no-grad early exit).
            if self.freeze_unet and self.use_region_loss and not getattr(self, 'use_latent_l1_loss', False):
                max_t_ratio = (self.region_gate_center
                               + self.region_gate_temperature
                               * math.log((1 - self.region_gate_compute_eps)
                                          / self.region_gate_compute_eps))
                effective_T_max = max(1, int(max_t_ratio * T_max))

            # Optional override: decouple t-sampling clamp from the region gate.
            override = self.train_t_clamp_ratio
            if override is not None:
                if isinstance(override, str) and override == "auto":
                    if self.recon_t_threshold > 0:
                        effective_T_max = min(effective_T_max, int(self.recon_t_threshold))
                else:
                    effective_T_max = max(1, int(float(override) * T_max))

            sampled_timesteps = torch.randint(0, effective_T_max, size=(real_bsz,), device=self.device).long()
            # Expand sampled_timesteps to shape (real_bsz, num_tasks, num_views), then flatten to (total_scenes,)
            t = sampled_timesteps.unsqueeze(1).unsqueeze(2).repeat(1, self.num_tasks, num_sample_views).reshape(-1)
            t[input_indices] = 0 # Condition views are timestep 0 (clean)

        # 8. Align Data (Interleave Tasks)
        # We must align data to (Batch, Task, View) to match input_indices/output_indices order

        # Specific Task Data
        aligned_input_latents = _interleave_tasks(input_latents_list, real_bsz, T_in)
        aligned_target_latents = _interleave_tasks(output_latents_list, real_bsz, T_out)

        # Shared Data (Rays & PC)
        aligned_input_rays = _repeat_shared(input_ray_embeds, real_bsz, self.num_tasks, T_in)
        aligned_output_rays = _repeat_shared(output_ray_embeds, real_bsz, self.num_tasks, T_out)

        aligned_input_pc = _interleave_tasks(input_cond_list, real_bsz, T_in)
        aligned_output_pc = _interleave_tasks(output_cond_list, real_bsz, T_out)

        # 9. Add Noise
        noise = torch.randn_like(aligned_target_latents)
        target_timesteps = t[output_indices]

        if self.v_as_flow_prediction:
            # Expand t_continuous and t_discrete to match aligned_target_latents shape
            # t_continuous: (real_bsz,) -> (real_bsz * num_tasks * T_out, 1, 1, 1)
            t_cont_expanded = t_continuous.unsqueeze(1).unsqueeze(2).repeat(1, self.num_tasks, T_out).reshape(-1, 1, 1, 1)
            t_disc_expanded = t_discrete.unsqueeze(1).unsqueeze(2).repeat(1, self.num_tasks, T_out).reshape(-1)

            flow_source = aligned_output_pc if self.init_from_pc_renders else noise

            noisy_targets = t_cont_expanded * aligned_target_latents + (1 - t_cont_expanded) * flow_source
            noisy_targets = ((alpha_t[t_disc_expanded] + sigma_t[t_disc_expanded]).view(-1, 1, 1, 1) * noisy_targets).to(dtype=self.mixed_dtype)
        else:
            noisy_targets = self.noise_scheduler.add_noise(aligned_target_latents, noise, target_timesteps).to(dtype=self.mixed_dtype)

        # 10. Model Input Construction
        # Determine channel dimensions from the first task (assuming all tasks share latent dim)
        c_dim = input_latents_list[0].shape[1] + input_ray_embeds.shape[1] + 1 + input_pc_latents.shape[1]

        model_input = torch.zeros(
            (total_scenes, c_dim, input_latents_list[0].shape[-2], input_latents_list[0].shape[-1]),
            device=self.device, dtype=self.mixed_dtype
        )

        # Create Observation Mask Channel
        obs_mask_channel = torch.zeros((total_scenes, 1, model_input.shape[-2], model_input.shape[-1]),
                                       device=self.device, dtype=self.mixed_dtype)
        obs_mask_channel[input_indices] = 1.0
        if self.use_observation_mask and out_mask_soft is not None:
            obs_mask_channel[output_indices] = _repeat_shared(
                out_mask_soft.to(self.mixed_dtype), real_bsz, self.num_tasks, T_out
            )

        # Zero obs mask for CFG-dropped samples (all views: input + output)
        if cfg_drop_mask is not None:
            cfg_drop_all = cfg_drop_mask[:, None, None].expand(
                -1, self.num_tasks, num_sample_views
            ).reshape(-1)
            obs_mask_channel[cfg_drop_all.bool()] = 0.0

        # Fill Input Views
        model_input[input_indices] = torch.cat([
            aligned_input_latents, aligned_input_rays, obs_mask_channel[input_indices], aligned_input_pc
        ], dim=1)

        # Fill Target Views
        model_input[output_indices] = torch.cat([
            noisy_targets, aligned_output_rays, obs_mask_channel[output_indices], aligned_output_pc
        ], dim=1)

        # Build multi-scale obs_mask dict for dual-FFN routing
        if self.dual_ffn and out_mask_soft is not None:
            latent_h, latent_w = output_pc_latents.shape[-2:]
            full_obs_mask = torch.ones(
                (total_scenes, 1, latent_h, latent_w),
                device=self.device, dtype=self.mixed_dtype,
            )
            # Input views: fully observed (already ones)
            # Output views: use soft warp mask
            full_obs_mask[output_indices] = _repeat_shared(
                out_mask_soft.to(self.mixed_dtype), real_bsz, self.num_tasks, T_out
            )
            obs_mask_dict = {}
            for scale in [1, 2, 4, 8]:
                h, w = latent_h // scale, latent_w // scale
                if scale == 1:
                    m = full_obs_mask
                else:
                    m = F.interpolate(full_obs_mask, size=(h, w), mode='bilinear', align_corners=False)
                # Flatten to token format: (N, H*W, 1)
                obs_mask_dict[h * w] = m.flatten(2).permute(0, 2, 1)
            cross_attention_kwargs = cross_attention_kwargs or {}
            cross_attention_kwargs["obs_mask_dict"] = obs_mask_dict

        # 10. UNet Forward
        pred_full = self.unet(
            model_input,
            timestep=t,  # UNet expects discrete timestep indices
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        # 11. Extract prediction for output indices
        pred = pred_full[output_indices, :self.latent_channels]  # (N, 4, H, W)

        # Prediction Type Logic
        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")

        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction" and self.v_as_flow_prediction:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=self.mixed_dtype)
            alpha_bar = alphas_cumprod.gather(0, target_timesteps)
            alpha_t = alpha_bar.sqrt().view(-1, 1, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
            # Reformulate the prediction as a pseudo-flow prediction
            pred = alpha_t * (noisy_targets - pred) - sigma_t * (noisy_targets + pred)
            flow_source = aligned_output_pc if self.init_from_pc_renders else noise
            target = aligned_target_latents - flow_source
        elif pred_type == "v_prediction":
            # Explicit device casting for DDP safety
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=self.mixed_dtype)
            alpha_bar = alphas_cumprod.gather(0, target_timesteps)
            alpha_t = alpha_bar.sqrt().view(-1, 1, 1, 1)
            sigma_t = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
            target = alpha_t * noise - sigma_t * aligned_target_latents
        else:
            target = aligned_target_latents

        # Per-element reconstruction loss — always standard MSE for the mean
        per_elem_loss = (pred - target) ** 2

        # 12. SNR Weighting (Robust) - Only for diffusion, not flow matching
        if self.snr_weighting_mode is not None and not self.v_as_flow_prediction:
            # Re-fetch alphas_cumprod to be safe
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=self.mixed_dtype)
            alpha_bar = alphas_cumprod.gather(0, target_timesteps)
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)

            if self.snr_weighting_mode == "min_snr_gamma":
                # Use registered buffer self.snr_gamma
                weight = torch.minimum(snr, self.snr_gamma) / (snr + 1e-8)
            elif self.snr_weighting_mode == "mid_snr_peak":
                weight = torch.minimum(snr, self.snr_mid) / torch.maximum(snr, self.snr_mid)
            else:
                weight = torch.ones_like(snr)

            per_elem_loss = per_elem_loss * weight.view(-1, 1, 1, 1)

        # 12b. Dual Reconstruction Side-Branch Loss (Sparse Geometric Attention)
        if self.dual_recon and output_warp_masks is not None and geo_index_map is not None:
            # Extract task indices
            total_outputs = pred.shape[0]
            all_idx = torch.arange(total_outputs, device=self.device)
            task0_idx = all_idx.view(real_bsz, self.num_tasks, T_out)[:, 0, :].reshape(-1)

            # Handle CFG dropout
            if cfg_drop_mask is not None and cfg_drop_mask.any():
                keep = ~cfg_drop_mask
                keep_out = keep[:, None].expand(-1, T_out).reshape(-1)
                recon_task0_idx = task0_idx[keep_out]
                recon_warp_masks = output_warp_masks[keep]
                recon_reliability = output_reliability_maps[keep] if output_reliability_maps is not None else None
                recon_geo_index_map = self._filter_geo_index_map(geo_index_map, keep)
                recon_out_out_index_map = self._filter_geo_index_map(out_out_index_map, keep) if out_out_index_map is not None else None
                recon_output_pc_renders = output_pc_renders[keep]
                recon_output_warp_masks_raw = output_warp_masks[keep]
                recon_input_imgs = input_imgs[keep]
                recon_output_imgs = output_imgs[keep]
                recon_input_rays = input_rays[keep] if getattr(self, 'recon_use_plucker', False) else None
                recon_output_rays = output_rays[keep] if getattr(self, 'recon_use_plucker', False) else None
            else:
                recon_task0_idx = task0_idx
                recon_warp_masks = output_warp_masks
                recon_reliability = output_reliability_maps
                recon_geo_index_map = geo_index_map
                recon_out_out_index_map = out_out_index_map
                recon_output_pc_renders = output_pc_renders
                recon_output_warp_masks_raw = output_warp_masks
                recon_input_imgs = input_imgs
                recon_output_imgs = output_imgs
                recon_input_rays = input_rays if getattr(self, 'recon_use_plucker', False) else None
                recon_output_rays = output_rays if getattr(self, 'recon_use_plucker', False) else None

            # --- Recon side-branch (trains only recon_branch, detached from UNet) ---
            if self.recon_unrolled:
                # Unrolled denoising: run N-step inference (no_grad) for realistic x0
                x0_unet, temb = self._unrolled_denoise_for_recon(
                    aligned_input_latents, aligned_input_rays, aligned_output_rays,
                    aligned_input_pc, aligned_output_pc, obs_mask_channel,
                    prompt_embeds, input_indices, output_indices,
                    cross_attention_kwargs, real_bsz, T_in, T_out,
                )
                if cfg_drop_mask is not None and cfg_drop_mask.any():
                    keep_out = keep[:, None].expand(-1, T_out).reshape(-1)
                    x0_unet = x0_unet[keep_out]
                    temb = temb[keep_out]
                target_timesteps_task0 = None
                aligned_target_latents_task0 = None
            else:
                # Original single-step x0 from the training forward pass
                task0_out_idx = torch.arange(len(output_indices), device=self.device).view(
                    real_bsz, self.num_tasks, T_out)[:, 0, :].reshape(-1)
                if cfg_drop_mask is not None and cfg_drop_mask.any():
                    keep_out_local = keep[:, None].expand(-1, T_out).reshape(-1)
                    task0_out_idx = task0_out_idx[keep_out_local]

                temb = self.unet._cached_temb.detach()[output_indices][task0_out_idx]

                pred_task0 = pred[recon_task0_idx]
                noisy_targets_task0 = noisy_targets[recon_task0_idx]
                target_timesteps_task0 = target_timesteps[recon_task0_idx]
                aligned_target_latents_task0 = aligned_target_latents[recon_task0_idx]

                x0_unet = self._reconstruct_x0(pred_task0, noisy_targets_task0, target_timesteps_task0).detach()

            if self.recon_pixel_space:
                # ===== Pixel-space recon branch =====
                # Per-sample low-t gate (unrolled path skips this — all samples are valid)
                if self.recon_unrolled:
                    t_low_mask = None
                elif self.recon_t_threshold > 0:
                    t_low_mask = (target_timesteps_task0 < self.recon_t_threshold).to(self.device).float()
                else:
                    t_low_mask = None

                if t_low_mask is not None and t_low_mask.sum() == 0:
                    # All samples above threshold: skip branch forward, keep recon_branch params in DDP graph
                    recon_l1 = torch.zeros((), device=self.device)
                    recon_lpips = torch.zeros((), device=self.device)
                    recon_grad = torch.zeros((), device=self.device)
                    recon_delta_sparsity = torch.zeros((), device=self.device)
                    recon_gram = torch.zeros((), device=self.device)
                    recon_loss = 0 * sum(p.sum() for p in self.recon_branch.parameters())
                    recon_l1_grad_norm = torch.zeros((), device=self.device)
                    recon_lpips_grad_norm = torch.zeros((), device=self.device)
                    recon_grad_grad_norm = torch.zeros((), device=self.device)
                    recon_delta_sparsity_grad_norm = torch.zeros((), device=self.device)
                    recon_gram_grad_norm = torch.zeros((), device=self.device)
                else:
                    # Decode x0_pred to pixel space (frozen VAE, no grad).
                    # Routes through decode_with_skip so a loaded (frozen) skip
                    # adapter shapes the decode that the recon branch refines;
                    # falls back to plain vae.decode when use_vae_skip is off.
                    scaling = self.image_vae.config.scaling_factor
                    with torch.no_grad():
                        decoded_x0 = self.decode_with_skip(x0_unet / scaling)  # [-1, 1]

                    # Prepare source images: (B*T_in, 3, H, W) in [-1, 1]
                    source_images = einops.rearrange(recon_input_imgs, 'b t c h w -> (b t) c h w')

                    # PC renders and warp masks at pixel resolution
                    pc_renders_flat = einops.rearrange(recon_output_pc_renders, 'b t c h w -> (b t) c h w')
                    warp_masks_flat = einops.rearrange(recon_output_warp_masks_raw, 'b t c h w -> (b t) c h w')

                    # Optional Plucker ray conditioning at pixel resolution
                    if getattr(self, 'recon_use_plucker', False) and recon_input_rays is not None:
                        target_plucker = einops.rearrange(recon_output_rays, 'b t c h w -> (b t) c h w').to(dtype=decoded_x0.dtype)
                        source_plucker = einops.rearrange(recon_input_rays, 'b t c h w -> (b t) c h w').to(dtype=decoded_x0.dtype)
                    else:
                        target_plucker = None
                        source_plucker = None

                    # Pixel-space recon branch forward
                    pixel_delta = self.recon_branch(
                        decoded_x0, pc_renders_flat, warp_masks_flat,
                        source_images, recon_geo_index_map, recon_out_out_index_map, temb, T_out,
                        target_plucker=target_plucker,
                        source_plucker=source_plucker,
                    )

                    # Observation mask at pixel resolution (use recon-specific kernel if configured)
                    obs_mask_pixel = self._compute_pixel_obs_mask(recon_warp_masks, use_recon_kernel=True)
                    # Reliability weighting for recon branch
                    if getattr(self, 'use_reliability_weighting', False) and recon_reliability is not None:
                        _pixel_rel = self._compute_pixel_obs_mask(recon_reliability, use_recon_kernel=True)  # same blur, no downsample
                        obs_mask_pixel = obs_mask_pixel * _pixel_rel
                    pixel_delta = pixel_delta * obs_mask_pixel
                    pixel_corrected = decoded_x0 + pixel_delta

                    # GT pixels: output_imgs are already in [-1, 1]
                    gt_pixels = einops.rearrange(recon_output_imgs, 'b t c h w -> (b t) c h w')

                    # Per-sample L1 in pixel space (observed regions only)
                    denom = obs_mask_pixel.sum(dim=[1, 2, 3]).clamp(min=1)
                    per_sample_l1 = (obs_mask_pixel * (pixel_corrected - gt_pixels).abs()).sum(dim=[1, 2, 3]) / denom

                    # Per-sample edge-preserving (gradient-domain L1) loss in pixel space.
                    # Only count finite-difference gradients where both neighbors are inside
                    # the observed region, so the mask boundary itself doesn't register as
                    # a "real" edge. Cost is tiny (a few subtractions); skip entirely when
                    # the weight is zero for backward compatibility.
                    if self.recon_grad_weight > 0.0:
                        _dx_pred = pixel_corrected[..., :, 1:] - pixel_corrected[..., :, :-1]
                        _dx_gt = gt_pixels[..., :, 1:] - gt_pixels[..., :, :-1]
                        _dx_mask = obs_mask_pixel[..., :, 1:] * obs_mask_pixel[..., :, :-1]
                        _dy_pred = pixel_corrected[..., 1:, :] - pixel_corrected[..., :-1, :]
                        _dy_gt = gt_pixels[..., 1:, :] - gt_pixels[..., :-1, :]
                        _dy_mask = obs_mask_pixel[..., 1:, :] * obs_mask_pixel[..., :-1, :]
                        _dx_denom = _dx_mask.sum(dim=[1, 2, 3]).clamp(min=1)
                        _dy_denom = _dy_mask.sum(dim=[1, 2, 3]).clamp(min=1)
                        _dx_sum = (_dx_mask * (_dx_pred - _dx_gt).abs()).sum(dim=[1, 2, 3])
                        _dy_sum = (_dy_mask * (_dy_pred - _dy_gt).abs()).sum(dim=[1, 2, 3])
                        per_sample_grad = 0.5 * (_dx_sum / _dx_denom + _dy_sum / _dy_denom)
                    else:
                        per_sample_grad = torch.zeros_like(per_sample_l1)

                    # Masked L1 sparsity on pixel_delta — LPIPS-to-GT does not penalize edit
                    # magnitude, so without this the branch is free to nudge every observed
                    # pixel. tau>0 switches on the evidence-weighted variant: the penalty dies
                    # where pc_render disagrees with decoded_x0 (i.e. where there IS signal to
                    # correct), keeping pixel_delta near zero everywhere else.
                    if self.recon_delta_sparsity_weight > 0.0:
                        if self.recon_delta_sparsity_evidence_tau > 0.0:
                            # pc_render is in [0,1]; decoded_x0 is in [-1,1]. Compare in [-1,1].
                            _pc_11 = pc_renders_flat * 2.0 - 1.0
                            _disc = (_pc_11 - decoded_x0).detach().abs().mean(dim=1, keepdim=True)
                            _signal = (_disc / self.recon_delta_sparsity_evidence_tau).clamp_(0.0, 1.0)
                            _sparsity_w = (1.0 - _signal) * obs_mask_pixel
                        else:
                            _sparsity_w = obs_mask_pixel
                        _sparsity_denom = _sparsity_w.sum(dim=[1, 2, 3]).clamp(min=1)
                        per_sample_sparsity = (_sparsity_w * pixel_delta.abs()).sum(dim=[1, 2, 3]) / _sparsity_denom
                    else:
                        per_sample_sparsity = torch.zeros_like(per_sample_l1)

                    # Per-sample LPIPS in pixel space
                    lpips_net = self._get_recon_lpips_net()
                    # Normalize to [0,1] for LPIPS input range, then back to [-1,1] for the net
                    pc_01 = (pixel_corrected + 1) / 2  # [-1,1] -> [0,1]
                    gt_01 = (gt_pixels + 1) / 2
                    if self.recon_pixel_lpips_full:
                        # Full-image LPIPS: unobserved region is the (detached) UNet output, so
                        # LPIPS gradient flows only through pixel_delta in observed regions, but
                        # VGG features span the mask boundary giving the branch a seam-aware signal.
                        pred_lpips = pc_01
                        gt_lpips = gt_01
                    else:
                        pred_lpips = pc_01 * obs_mask_pixel + 0.5 * (1 - obs_mask_pixel)
                        gt_lpips = gt_01 * obs_mask_pixel + 0.5 * (1 - obs_mask_pixel)
                    if self.lpips_resize > 0:
                        pred_lpips = F.interpolate(pred_lpips, size=(self.lpips_resize, self.lpips_resize),
                                                   mode='bilinear', align_corners=False)
                        gt_lpips = F.interpolate(gt_lpips, size=(self.lpips_resize, self.lpips_resize),
                                                 mode='bilinear', align_corners=False)
                    lpips_vals = lpips_net(pred_lpips * 2 - 1, gt_lpips * 2 - 1).flatten()  # (N,)

                    # Per-sample VGG Gram-matrix style loss. Global (full-image) statistic —
                    # no masking (would distort the gram distribution); pixel_delta is already
                    # zero outside observed regions, so gradients only flow through observed pixels.
                    if self.recon_gram_weight > 0.0:
                        vgg_net = self._get_recon_vgg_net()
                        # [-1,1] -> [0,1] -> ImageNet-normalized for VGG19.
                        pred_vgg = ((pixel_corrected + 1) / 2 - self._imagenet_mean) / self._imagenet_std
                        gt_vgg = ((gt_pixels + 1) / 2 - self._imagenet_mean) / self._imagenet_std
                        per_sample_gram = torch.stack([
                            gram_loss(gt_vgg[i:i + 1], pred_vgg[i:i + 1], vgg_net)
                            for i in range(pred_vgg.shape[0])
                        ])
                    else:
                        per_sample_gram = torch.zeros_like(per_sample_l1)

                    # Reduce per-sample losses with optional low-t weighting
                    if t_low_mask is not None:
                        weight_sum = t_low_mask.sum().clamp(min=1)
                        recon_l1 = (per_sample_l1 * t_low_mask).sum() / weight_sum
                        recon_lpips = (lpips_vals * t_low_mask).sum() / weight_sum
                        recon_grad = (per_sample_grad * t_low_mask).sum() / weight_sum
                        recon_delta_sparsity = (per_sample_sparsity * t_low_mask).sum() / weight_sum
                        recon_gram = (per_sample_gram * t_low_mask).sum() / weight_sum
                    else:
                        recon_l1 = per_sample_l1.mean()
                        recon_lpips = lpips_vals.mean()
                        recon_grad = per_sample_grad.mean()
                        recon_delta_sparsity = per_sample_sparsity.mean()
                        recon_gram = per_sample_gram.mean()

                    recon_loss = (
                        self.recon_l1_weight * recon_l1
                        + self.recon_lpips_weight * recon_lpips
                        + self.recon_grad_weight * recon_grad
                        + self.recon_delta_sparsity_weight * recon_delta_sparsity
                        + self.recon_gram_weight * recon_gram
                    )

                    # Per-loss gradient norms on recon branch params (diagnostic).
                    # Uses autograd.grad with retain_graph=True so the main backward on
                    # `total` still works; .detach() prevents these norms from entering
                    # any subsequent autograd graph.
                    _recon_params = [p for p in self.recon_branch.parameters() if p.requires_grad]
                    _l1_term = self.recon_l1_weight * recon_l1
                    _lpips_term = self.recon_lpips_weight * recon_lpips
                    _l1_grads = torch.autograd.grad(
                        _l1_term, _recon_params, retain_graph=True, allow_unused=True,
                    )
                    _lpips_grads = torch.autograd.grad(
                        _lpips_term, _recon_params, retain_graph=True, allow_unused=True,
                    )
                    _zero = torch.zeros((), device=self.device)
                    _l1_sq = sum(
                        (g.detach().float().pow(2).sum() for g in _l1_grads if g is not None),
                        _zero,
                    )
                    _lpips_sq = sum(
                        (g.detach().float().pow(2).sum() for g in _lpips_grads if g is not None),
                        _zero,
                    )
                    recon_l1_grad_norm = _l1_sq.sqrt()
                    recon_lpips_grad_norm = _lpips_sq.sqrt()

                    # Edge-loss grad norm only meaningful when the term is active
                    if self.recon_grad_weight > 0.0:
                        _grad_term = self.recon_grad_weight * recon_grad
                        _grad_grads = torch.autograd.grad(
                            _grad_term, _recon_params, retain_graph=True, allow_unused=True,
                        )
                        _grad_sq = sum(
                            (g.detach().float().pow(2).sum() for g in _grad_grads if g is not None),
                            _zero,
                        )
                        recon_grad_grad_norm = _grad_sq.sqrt()
                    else:
                        recon_grad_grad_norm = torch.zeros((), device=self.device)

                    # Sparsity-loss grad norm only meaningful when the term is active
                    if self.recon_delta_sparsity_weight > 0.0:
                        _sparsity_term = self.recon_delta_sparsity_weight * recon_delta_sparsity
                        _sparsity_grads = torch.autograd.grad(
                            _sparsity_term, _recon_params, retain_graph=True, allow_unused=True,
                        )
                        _sparsity_sq = sum(
                            (g.detach().float().pow(2).sum() for g in _sparsity_grads if g is not None),
                            _zero,
                        )
                        recon_delta_sparsity_grad_norm = _sparsity_sq.sqrt()
                    else:
                        recon_delta_sparsity_grad_norm = torch.zeros((), device=self.device)

                    # Gram-loss grad norm only meaningful when the term is active
                    if self.recon_gram_weight > 0.0:
                        _gram_term = self.recon_gram_weight * recon_gram
                        _gram_grads = torch.autograd.grad(
                            _gram_term, _recon_params, retain_graph=True, allow_unused=True,
                        )
                        _gram_sq = sum(
                            (g.detach().float().pow(2).sum() for g in _gram_grads if g is not None),
                            _zero,
                        )
                        recon_gram_grad_norm = _gram_sq.sqrt()
                    else:
                        recon_gram_grad_norm = torch.zeros((), device=self.device)
            else:
                # ===== Latent-space recon branch with pixel-space supervision =====
                # Decode x0_unet to pixel space for branch input (no grad, one-way).
                # Use decode_with_skip so a loaded (frozen) skip adapter is exercised
                # consistently here and at inference; falls back to plain vae.decode
                # when use_vae_skip is off.
                scaling = self.image_vae.config.scaling_factor
                with torch.no_grad():
                    decoded_x0 = self.decode_with_skip(x0_unet / scaling)  # [-1, 1]

                # Prepare pixel-res inputs
                source_images = einops.rearrange(recon_input_imgs, 'b t c h w -> (b t) c h w')
                pc_renders_flat = einops.rearrange(recon_output_pc_renders, 'b t c h w -> (b t) c h w')
                warp_masks_flat = einops.rearrange(recon_output_warp_masks_raw, 'b t c h w -> (b t) c h w')

                # Recon branch: pixel-res attention + latent fusion → 4ch latent delta
                x0_delta = self.recon_branch(
                    decoded_x0, pc_renders_flat, warp_masks_flat,
                    source_images, x0_unet,
                    recon_geo_index_map, recon_out_out_index_map, temb, T_out,
                )

                # Mask to observed regions (latent resolution)
                obs_mask = self.compute_observation_mask(recon_warp_masks)
                if getattr(self, 'use_reliability_weighting', False) and recon_reliability is not None:
                    obs_mask = obs_mask * self.compute_reliability_mask(recon_reliability)
                x0_delta = x0_delta * obs_mask
                x0_corrected = x0_unet + x0_delta

                # Subset skip features to match CFG-kept samples
                if self.use_vae_skip and self._pc_skip_features is not None:
                    if cfg_drop_mask is not None and cfg_drop_mask.any():
                        keep_out = (~cfg_drop_mask)[:, None].expand(-1, T_out).reshape(-1)
                        recon_skip = [sf[keep_out] for sf in self._pc_skip_features]
                    else:
                        recon_skip = self._pc_skip_features
                else:
                    recon_skip = None

                # VAE decode for pixel-space loss (WITH gradients for backprop)
                decoded_corrected = self.decode_with_skip(x0_corrected / scaling, skip_features=recon_skip)

                # GT pixels
                gt_pixels = einops.rearrange(recon_output_imgs, 'b t c h w -> (b t) c h w')

                # Pixel-res observation mask for loss
                obs_mask_pixel = self._compute_pixel_obs_mask(recon_warp_masks)
                if getattr(self, 'use_reliability_weighting', False) and recon_reliability is not None:
                    _pixel_rel = self._compute_pixel_obs_mask(recon_reliability)
                    obs_mask_pixel = obs_mask_pixel * _pixel_rel

                # L1 loss in pixel space
                recon_l1 = (obs_mask_pixel * (decoded_corrected - gt_pixels).abs()).sum() / obs_mask_pixel.sum().clamp(min=1)

                # LPIPS loss in pixel space
                lpips_net = self._get_recon_lpips_net()
                pc_01 = (decoded_corrected + 1) / 2  # [-1,1] -> [0,1]
                gt_01 = (gt_pixels + 1) / 2
                pred_masked = pc_01 * obs_mask_pixel + 0.5 * (1 - obs_mask_pixel)
                gt_masked = gt_01 * obs_mask_pixel + 0.5 * (1 - obs_mask_pixel)
                if self.lpips_resize > 0:
                    pred_masked = F.interpolate(pred_masked, size=(self.lpips_resize, self.lpips_resize),
                                                mode='bilinear', align_corners=False)
                    gt_masked = F.interpolate(gt_masked, size=(self.lpips_resize, self.lpips_resize),
                                              mode='bilinear', align_corners=False)
                recon_lpips = lpips_net(pred_masked * 2 - 1, gt_masked * 2 - 1).mean()

                recon_loss = self.recon_l1_weight * recon_l1 + self.recon_lpips_weight * recon_lpips

            # --- Diffusion MSE (standard, no spatial downweighting) ---
            diffusion_mse_task0 = per_elem_loss[task0_idx].mean()

            total = diffusion_mse_task0 + recon_loss
            if self.num_tasks > 1:
                task1_idx = all_idx.view(real_bsz, self.num_tasks, T_out)[:, 1, :].reshape(-1)
                diffusion_mse_task1 = per_elem_loss[task1_idx].mean()
                total = total + diffusion_mse_task1
            else:
                diffusion_mse_task1 = torch.zeros(1, device=self.device).squeeze()

            result = {
                'total': total,
                'diffusion_mse': diffusion_mse_task0.detach(),
                'coord_diffusion_mse': diffusion_mse_task1.detach(),
                'recon_l1': recon_l1.detach(),
                'recon_lpips': recon_lpips.detach(),
            }
            if self.recon_pixel_space:
                result['recon_grad'] = recon_grad.detach()
                result['recon_delta_sparsity'] = recon_delta_sparsity.detach()
                result['recon_gram'] = recon_gram.detach()
                result['recon_l1_grad_norm'] = recon_l1_grad_norm.detach()
                result['recon_lpips_grad_norm'] = recon_lpips_grad_norm.detach()
                result['recon_grad_grad_norm'] = recon_grad_grad_norm.detach()
                result['recon_delta_sparsity_grad_norm'] = recon_delta_sparsity_grad_norm.detach()
                result['recon_gram_grad_norm'] = recon_gram_grad_norm.detach()
            if cfg_drop_mask is not None:
                result['cfg_drop_rate'] = cfg_drop_mask.float().mean().item()
            return result

        # DDP safety: when dual_recon is enabled but recon branch was skipped,
        # add dummy term to keep recon_branch params in gradient graph
        if self.dual_recon and (output_warp_masks is None or geo_index_map is None):
            per_elem_loss = per_elem_loss + 0 * sum(p.sum() for p in self.recon_branch.parameters())

        # 13. Region-Specific Loss
        if self.use_region_loss:
            # Extract task indices — needed for per-task diffusion MSE split
            total_outputs = pred.shape[0]  # B * num_tasks * T_out
            all_idx = torch.arange(total_outputs, device=self.device)
            task0_idx = all_idx.view(real_bsz, self.num_tasks, T_out)[:, 0, :].reshape(-1)
            # raise Exception(f"Task0 idx: {task0_idx}, pred shape: {pred.shape}, T_out: {T_out}, num_tasks: {self.num_tasks}")
            # Split diffusion MSE per task so auxiliary losses are properly balanced
            diffusion_mse_task0 = per_elem_loss[task0_idx].mean()
            _avg_diff_weight = 1.0  # for wandb logging; updated below if downweighting is active
            if self.num_tasks > 1:
                task1_idx = all_idx.view(real_bsz, self.num_tasks, T_out)[:, 1, :].reshape(-1)
                diffusion_mse_task1 = per_elem_loss[task1_idx].mean()
            else:
                diffusion_mse_task1 = torch.zeros(1, device=self.device).squeeze()

            if self.use_latent_l1_loss:
                # Warmup and step counter
                _warmup = (self._region_loss_step.float() / max(self.region_loss_warmup_steps, 1)).clamp(0.0, 1.0)
                self._region_loss_step += 1

                total = diffusion_mse_task0
                if self.num_tasks > 1:
                    total = total + diffusion_mse_task1

                result = {
                    'diffusion_mse': diffusion_mse_task0.detach(),
                    'coord_diffusion_mse': diffusion_mse_task1.detach(),
                    'coord_obs_l1': torch.zeros(1, device=self.device).squeeze(),
                }

                # Auxiliary L1 on predicted clean latents in observed regions
                if output_warp_masks is not None:
                    # Handle CFG dropout
                    if cfg_drop_mask is not None and cfg_drop_mask.any():
                        _keep_out = (~cfg_drop_mask)[:, None].expand(-1, T_out).reshape(-1)
                        _l1_idx = task0_idx[_keep_out]
                        _l1_warp = output_warp_masks[~cfg_drop_mask]
                        _l1_reliability = output_reliability_maps[~cfg_drop_mask] if output_reliability_maps is not None else None
                    else:
                        _l1_idx = task0_idx
                        _l1_warp = output_warp_masks
                        _l1_reliability = output_reliability_maps

                    if len(_l1_idx) > 0:
                        x0_pred = self._reconstruct_x0(
                            pred[_l1_idx], noisy_targets[_l1_idx], target_timesteps[_l1_idx]
                        )
                        x0_gt = aligned_target_latents[_l1_idx]
                        _obs_mask = self.compute_observation_mask(_l1_warp)

                        # Reliability weighting: downweight observed regions seen from very different angles
                        if self.use_reliability_weighting and _l1_reliability is not None:
                            _reliability_mask = self.compute_reliability_mask(_l1_reliability)
                            _obs_mask = _obs_mask * _reliability_mask

                        _l1 = (_obs_mask * (x0_pred - x0_gt).abs()).sum() / _obs_mask.sum().clamp(min=1.0)

                        # Timestep gate — keep per-scene tensor for spatial downweighting
                        if self.v_as_flow_prediction:
                            _t_ratio = 1.0 - t_continuous
                        else:
                            _T_max = float(self.noise_scheduler.config.num_train_timesteps)
                            _per_scene_ts = target_timesteps.view(real_bsz, self.num_tasks, T_out)[:, 0, 0]
                            _t_ratio = _per_scene_ts.float() / _T_max
                        _per_scene_gate = self._compute_soft_gate(_t_ratio)  # (real_bsz,)
                        if cfg_drop_mask is not None and cfg_drop_mask.any():
                            _gate = _per_scene_gate[~cfg_drop_mask].mean().item()
                        else:
                            _gate = _per_scene_gate.mean().item()

                        # Spatially downweight diffusion MSE in observed regions
                        if self.region_diffusion_downweight > 0:
                            _gate_exp = _per_scene_gate[:, None].expand(-1, T_out).reshape(-1, 1, 1, 1)

                            if cfg_drop_mask is not None and cfg_drop_mask.any():
                                _gate_kept = _gate_exp[_keep_out]
                                _weight_map = 1.0 - _obs_mask * _gate_kept * self.region_diffusion_downweight
                                _kept_loss = (per_elem_loss[_l1_idx] * _weight_map).mean()
                                _drop_out = cfg_drop_mask[:, None].expand(-1, T_out).reshape(-1)
                                _dropped_idx = task0_idx[_drop_out]
                                _n_kept = _l1_idx.shape[0]
                                _n_dropped = _dropped_idx.shape[0]
                                if _n_dropped > 0:
                                    _dropped_loss = per_elem_loss[_dropped_idx].mean()
                                    diffusion_mse_task0 = (_kept_loss * _n_kept + _dropped_loss * _n_dropped) / (_n_kept + _n_dropped)
                                else:
                                    diffusion_mse_task0 = _kept_loss
                                _avg_diff_weight = _weight_map.mean().item()
                            else:
                                _weight_map = 1.0 - _obs_mask * _gate_exp * self.region_diffusion_downweight
                                diffusion_mse_task0 = (per_elem_loss[task0_idx] * _weight_map).mean()
                                _avg_diff_weight = _weight_map.mean().item()

                            # Re-assemble total with updated diffusion_mse_task0
                            total = diffusion_mse_task0
                            if self.num_tasks > 1:
                                total = total + diffusion_mse_task1

                        total = total + self.latent_l1_weight * _gate * _warmup * _l1

                        result['diffusion_mse'] = diffusion_mse_task0.detach()
                        result['diffusion_weight'] = _avg_diff_weight
                        result['latent_l1'] = _l1.detach()
                        result['latent_l1_gate'] = _gate
                        result['latent_l1_warmup'] = _warmup.detach()

                result['total'] = total
                result.setdefault('diffusion_weight', 1.0)
                if cfg_drop_mask is not None:
                    result['cfg_drop_rate'] = cfg_drop_mask.float().mean().item()
                return result

            # Compute per-scene t_ratio and soft gate
            if self.v_as_flow_prediction:
                per_scene_t_ratio = 1.0 - t_continuous  # (real_bsz,)
            else:
                T_max = float(self.noise_scheduler.config.num_train_timesteps)
                per_scene_ts = target_timesteps.view(real_bsz, self.num_tasks, T_out)[:, 0, 0]
                per_scene_t_ratio = per_scene_ts.float() / T_max  # (real_bsz,)

            per_scene_gate = self._compute_soft_gate(per_scene_t_ratio)  # (real_bsz,)
            gate_avg = per_scene_gate.mean().item()

            # Compute skip when gate is negligible for ALL samples
            if per_scene_gate.max().item() < self.region_gate_compute_eps:
                total = diffusion_mse_task0
                if self.num_tasks > 1:
                    total = total + diffusion_mse_task1

                self._region_loss_step += 1
                result = {
                    'total': total,
                    'diffusion_mse': diffusion_mse_task0.detach(),
                    'coord_diffusion_mse': diffusion_mse_task1.detach(),
                    'obs_l1': torch.zeros(1, device=self.device).squeeze(),
                    'obs_lpips': torch.zeros(1, device=self.device).squeeze(),
                    'dino_loss': torch.zeros(1, device=self.device).squeeze(),
                    'boundary_tv': torch.zeros(1, device=self.device).squeeze(),
                    'coord_obs_l1': torch.zeros(1, device=self.device).squeeze(),
                    'region_warmup': (self._region_loss_step.float() / max(self.region_loss_warmup_steps, 1)).clamp(0.0, 1.0).detach(),
                    'region_gate': gate_avg,
                    'diffusion_weight': 1.0,
                }
                if cfg_drop_mask is not None:
                    result['cfg_drop_rate'] = cfg_drop_mask.float().mean().item()
                return result

            # Exclude CFG-dropped samples from pixel-space region losses
            if cfg_drop_mask is not None and cfg_drop_mask.any():
                keep = ~cfg_drop_mask  # (real_bsz,)
                keep_out = keep[:, None].expand(-1, T_out).reshape(-1)  # (real_bsz*T_out,)
                region_task0_idx = task0_idx[keep_out]
                if self.num_tasks > 1 and self.region_coord_l1_weight > 0:
                    region_task1_idx = task1_idx[keep_out]
                region_warp_masks = output_warp_masks[keep]  # (B_keep, T_out, 1, H, W)
                region_reliability = output_reliability_maps[keep] if output_reliability_maps is not None else None
            else:
                region_task0_idx = task0_idx
                if self.num_tasks > 1 and self.region_coord_l1_weight > 0:
                    region_task1_idx = task1_idx
                region_warp_masks = output_warp_masks
                region_reliability = output_reliability_maps

            # All samples dropped → skip region losses, return diffusion MSE only
            if len(region_task0_idx) == 0:
                total = diffusion_mse_task0 + (diffusion_mse_task1 if self.num_tasks > 1 else 0)
                self._region_loss_step += 1
                return {
                    'total': total,
                    'diffusion_mse': diffusion_mse_task0.detach(),
                    'coord_diffusion_mse': diffusion_mse_task1.detach(),
                    'obs_l1': torch.zeros(1, device=self.device).squeeze(),
                    'obs_lpips': torch.zeros(1, device=self.device).squeeze(),
                    'dino_loss': torch.zeros(1, device=self.device).squeeze(),
                    'boundary_tv': torch.zeros(1, device=self.device).squeeze(),
                    'coord_obs_l1': torch.zeros(1, device=self.device).squeeze(),
                    'region_warmup': (self._region_loss_step.float() / max(self.region_loss_warmup_steps, 1)).clamp(0, 1).detach(),
                    'region_gate': gate_avg,
                    'diffusion_weight': 1.0,
                    'cfg_drop_rate': cfg_drop_mask.float().mean().item(),
                }

            # Compute observation mask from forward-warp coverage masks
            # obs_mask shape: (B_keep*T_out, 1, H_lat, W_lat)
            obs_mask = self.compute_observation_mask(region_warp_masks)

            # Reliability weighting: downweight observed regions seen from very different angles
            if self.use_reliability_weighting and region_reliability is not None:
                reliability_mask = self.compute_reliability_mask(region_reliability)
                obs_mask = obs_mask * reliability_mask

            # Spatially downweight diffusion MSE in observed regions
            if self.region_diffusion_downweight > 0:
                # Expand per-scene gate to per-sample: (real_bsz,) -> (real_bsz*T_out, 1, 1, 1)
                gate_expanded = per_scene_gate[:, None].expand(-1, T_out).reshape(-1, 1, 1, 1)

                if cfg_drop_mask is not None and cfg_drop_mask.any():
                    # CFG-kept samples: spatially downweighted
                    keep = ~cfg_drop_mask
                    keep_out = keep[:, None].expand(-1, T_out).reshape(-1)
                    gate_kept = gate_expanded[keep_out]
                    weight_map = 1.0 - obs_mask * gate_kept * self.region_diffusion_downweight
                    kept_loss = (per_elem_loss[region_task0_idx] * weight_map).mean()
                    # CFG-dropped samples: full strength (no obs_mask available)
                    drop_out = cfg_drop_mask[:, None].expand(-1, T_out).reshape(-1)
                    dropped_idx = task0_idx[drop_out]
                    n_kept = region_task0_idx.shape[0]
                    n_dropped = dropped_idx.shape[0]
                    if n_dropped > 0:
                        dropped_loss = per_elem_loss[dropped_idx].mean()
                        diffusion_mse_task0 = (kept_loss * n_kept + dropped_loss * n_dropped) / (n_kept + n_dropped)
                    else:
                        diffusion_mse_task0 = kept_loss
                    _avg_diff_weight = weight_map.mean().item()
                else:
                    # No CFG dropout: all samples get spatial downweighting
                    weight_map = 1.0 - obs_mask * gate_expanded * self.region_diffusion_downweight
                    diffusion_mse_task0 = (per_elem_loss[task0_idx] * weight_map).mean()
                    _avg_diff_weight = weight_map.mean().item()

            pred_task0 = pred[region_task0_idx]
            target_task0 = target[region_task0_idx]
            noisy_targets_task0 = noisy_targets[region_task0_idx]
            target_timesteps_task0 = target_timesteps[region_task0_idx]
            aligned_target_latents_task0 = aligned_target_latents[region_task0_idx]

            boundary_mask = self.compute_boundary_mask(obs_mask)

            # Reconstruct x0 for pixel-space losses (task 0 only)
            x0_pred = self._reconstruct_x0(pred_task0, noisy_targets_task0, target_timesteps_task0)

            # Shared VAE decoding (once for all pixel-space losses)
            scaling = self.image_vae.config.scaling_factor
            with torch.no_grad():
                gt_pixels = self.image_vae.decode(aligned_target_latents_task0 / scaling).sample
                gt_pixels = (gt_pixels / 2 + 0.5).clamp(0, 1)

            # Decode predictions — with skip features if available
            if self.use_vae_skip and self._pc_skip_features is not None:
                # Subset skip features to match CFG-kept samples
                if cfg_drop_mask is not None and cfg_drop_mask.any():
                    keep_out = (~cfg_drop_mask)[:, None].expand(-1, T_out).reshape(-1)
                    region_skip = [sf[keep_out] for sf in self._pc_skip_features]
                else:
                    region_skip = self._pc_skip_features
                # Detach x0_pred: pixel-loss gradients train only skip convs + LoRA,
                # not the UNet. UNet trains exclusively via diffusion MSE.
                pred_pixels = self.decode_with_skip(x0_pred.detach() / scaling, skip_features=region_skip)
            else:
                pred_pixels = self.image_vae.decode(x0_pred / scaling).sample
            pred_pixels = (pred_pixels / 2 + 0.5).clamp(0, 1)

            pixel_obs_mask = F.interpolate(obs_mask, size=pred_pixels.shape[-2:],
                                            mode='bilinear', align_corners=False)
            pixel_boundary_mask = F.interpolate(boundary_mask, size=pred_pixels.shape[-2:],
                                                 mode='bilinear', align_corners=False)

            # Debug visualization: save conditioning signals + masks every 100 steps
            # if self._region_loss_step % 1 == 0:
            #     self._debug_visualize_region_masks(
            #         output_pc_renders, output_coord_map, obs_mask, pixel_obs_mask,
            #         gt_pixels, pred_pixels, pixel_boundary_mask,
            #     )

            # Compute region losses (task 0 only)
            obs_l1 = self._compute_obs_l1(pred_pixels, gt_pixels, pixel_obs_mask)
            obs_lpips = self._compute_obs_lpips(pred_pixels, gt_pixels, pixel_obs_mask)
            if self.use_dino_loss:
                dino_loss = self._compute_dino_loss(pred_pixels, gt_pixels)
            else:
                dino_loss = torch.zeros(1, device=self.device, dtype=pred_pixels.dtype).squeeze()

            boundary_tv = self._compute_boundary_tv(pred_pixels, pixel_boundary_mask)

            # Compute coord L1 loss (Task 1) in pixel space
            if self.num_tasks > 1 and self.region_coord_l1_weight > 0:
                pred_task1 = pred[region_task1_idx]
                noisy_targets_task1 = noisy_targets[region_task1_idx]
                target_timesteps_task1 = target_timesteps[region_task1_idx]
                aligned_target_latents_task1 = aligned_target_latents[region_task1_idx]

                x0_pred_task1 = self._reconstruct_x0(pred_task1, noisy_targets_task1, target_timesteps_task1)

                coord_scaling = self.depth_vae.config.scaling_factor
                # Disable autocast for depth VAE decode when coord_fp32_vae is set
                # to preserve precision in the L1 loss supervision signal
                _coord_no_ac = torch.amp.autocast(device_type=self.device.type, enabled=not self.coord_fp32_vae)
                with torch.no_grad(), _coord_no_ac:
                    _gt_latents = aligned_target_latents_task1.float() if self.coord_fp32_vae else aligned_target_latents_task1
                    gt_coord_pixels = self.depth_vae.decode(_gt_latents / coord_scaling).sample
                    gt_coord_pixels = gt_coord_pixels[:, :3]  # depth_vae outputs 4 channels; keep XYZ

                with _coord_no_ac:
                    if self.use_depth_vae_skip and self._coord_skip_features is not None:
                        if cfg_drop_mask is not None and cfg_drop_mask.any():
                            keep_out = (~cfg_drop_mask)[:, None].expand(-1, T_out).reshape(-1)
                            coord_region_skip = [sf[keep_out] for sf in self._coord_skip_features]
                        else:
                            coord_region_skip = self._coord_skip_features
                        _pred_latents = x0_pred_task1.detach().float() if self.coord_fp32_vae else x0_pred_task1.detach()
                        pred_coord_pixels = self.decode_depth_with_skip(
                            _pred_latents / coord_scaling, skip_features=coord_region_skip
                        )
                    else:
                        _pred_latents = x0_pred_task1.float() if self.coord_fp32_vae else x0_pred_task1
                        pred_coord_pixels = self.depth_vae.decode(_pred_latents / coord_scaling).sample
                pred_coord_pixels = pred_coord_pixels[:, :3]

                pixel_obs_mask_coord = F.interpolate(obs_mask, size=pred_coord_pixels.shape[-2:],
                                                      mode='bilinear', align_corners=False)

                # Inverse-depth weighting: nearby surfaces get more gradient
                if self.coord_inverse_depth_weight and output_depth is not None:
                    if cfg_drop_mask is not None and cfg_drop_mask.any():
                        region_depth = output_depth[~cfg_drop_mask]  # (B_keep, T_out, H, W)
                    else:
                        region_depth = output_depth
                    depth_flat = region_depth.reshape(-1, *region_depth.shape[-2:]).unsqueeze(1)  # (B_keep*T_out, 1, H, W)
                    depth_at_coord_res = F.interpolate(depth_flat.float(), size=pred_coord_pixels.shape[-2:],
                                                        mode='bilinear', align_corners=False)
                    inv_depth = 1.0 / depth_at_coord_res.clamp(min=1e-3)
                    inv_depth = inv_depth / inv_depth.mean().clamp(min=1e-6)  # normalize: mean weight = 1
                    pixel_obs_mask_coord = pixel_obs_mask_coord * inv_depth

                coord_obs_l1 = self._compute_obs_l1(pred_coord_pixels, gt_coord_pixels, pixel_obs_mask_coord)
            else:
                coord_obs_l1 = torch.zeros(1, device=self.device, dtype=pred_pixels.dtype).squeeze()

            result = self._assemble_region_loss(
                diffusion_mse_task0, obs_l1, obs_lpips, dino_loss, boundary_tv,
                coord_diffusion_mse=diffusion_mse_task1,
                coord_obs_l1=coord_obs_l1,
                gate=gate_avg,
                diffusion_weight=_avg_diff_weight,
            )
            if self.use_reliability_weighting and region_reliability is not None:
                result['reliability_mean'] = reliability_mask.mean().item()
            if cfg_drop_mask is not None:
                result['cfg_drop_rate'] = cfg_drop_mask.float().mean().item()
            return result

        if cfg_drop_mask is not None:
            return {'total': per_elem_loss.mean(), 'cfg_drop_rate': cfg_drop_mask.float().mean().item()}
        return per_elem_loss.mean()

    @staticmethod
    def _rescale_noise_cfg(noise_cfg, noise_pred_cond, guidance_rescale):
        """Rescale guided noise to match the std of the conditional prediction (Lin et al.)."""
        std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        noise_rescaled = noise_cfg * (std_cond / (std_cfg + 1e-8))
        return guidance_rescale * noise_rescaled + (1.0 - guidance_rescale) * noise_cfg

    @staticmethod
    def _rescale_noise_cfg_regional(noise_cfg, noise_pred_cond, guidance_rescale, mask,
                                    guidance_rescale_unobs=None):
        """Region-wise rescale: compute weighted std independently for observed/unobserved regions."""
        gr_unobs = guidance_rescale_unobs if guidance_rescale_unobs is not None else guidance_rescale

        def _weighted_std(x, w):
            # w: (N, 1, H, W), x: (N, C, H, W)
            w = w.expand_as(x)
            w_sum = w.sum(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
            mean = (x * w).sum(dim=[1, 2, 3], keepdim=True) / w_sum
            var = (w * (x - mean) ** 2).sum(dim=[1, 2, 3], keepdim=True) / w_sum
            return var.sqrt().clamp(min=1e-8)

        obs_std_cond = _weighted_std(noise_pred_cond, mask)
        obs_std_cfg = _weighted_std(noise_cfg, mask)
        obs_ratio = obs_std_cond / obs_std_cfg

        inv_mask = 1.0 - mask
        unobs_std_cond = _weighted_std(noise_pred_cond, inv_mask)
        unobs_std_cfg = _weighted_std(noise_cfg, inv_mask)
        unobs_ratio = unobs_std_cond / unobs_std_cfg

        blended = mask * (guidance_rescale * (noise_cfg * obs_ratio) + (1.0 - guidance_rescale) * noise_cfg) \
                + inv_mask * (gr_unobs * (noise_cfg * unobs_ratio) + (1.0 - gr_unobs) * noise_cfg)
        return blended

    @staticmethod
    def _threshold_delta(delta: torch.Tensor, percentile: float) -> torch.Tensor:
        """Clip spatial outliers in the autoguidance delta.

        Autoguidance computes delta = main_unet - guide_unet, then scales
        it by autoguidance_scale.  Extreme spatial outliers in the delta
        get amplified and cause burnt patches.  Clipping the delta at the
        given percentile *before* amplification prevents this at the source
        without touching the latent scale (no x0 recovery, no rescaling).
        """
        s = torch.quantile(delta.abs().flatten(1).float(), percentile, dim=1)
        s = s.clamp(min=1e-6).to(delta.dtype).view(-1, 1, 1, 1)
        mean = delta.mean(dim=[2, 3], keepdim=True)
        delta = delta.clamp(-s, s)
        delta = delta + (mean - delta.mean(dim=[2, 3], keepdim=True))
        return delta

    @torch.no_grad()
    def evaluate(self, input_imgs, input_rays, input_pc_renders, output_rays, output_pc_renders,
                 # Added coord maps for multi-task support
                 input_coord_map=None,
                 output_coord_map=None,
                 input_warp_masks=None,
                 output_warp_masks=None,
                 T_in=3, T_out=1,
                 height=None, width=None, guidance_scale=1.0, generator=None, num_inference_steps=None,
                 geo_index_map=None,
                 out_out_index_map=None,
                 **kwargs):

        # Autoguidance support
        guide_unet = kwargs.pop("guide_unet", None)
        autoguidance_scale = kwargs.pop("autoguidance_scale", 1.5)
        autoguidance_scale_unobs = kwargs.pop("autoguidance_scale_unobs", None)
        use_autoguidance = guide_unet is not None and autoguidance_scale != 1.0

        # CFG Rescale (Lin et al.) — match guided std to conditional std
        guidance_rescale = kwargs.pop("guidance_rescale", 0.0)
        guidance_rescale_unobs = kwargs.pop("guidance_rescale_unobs", None)

        # Region-aware plain CFG: optional scale for unobserved regions
        guidance_unobs = kwargs.pop("guidance_unobs", None)

        # Observed-region latent blending (inpainting-style)
        obs_latent_blend = kwargs.pop("obs_latent_blend", False)
        obs_blend_strength = kwargs.pop("obs_blend_strength", 0.8)

        # Dynamic thresholding (Imagen, Saharia et al. 2022)
        dynamic_threshold_p = kwargs.pop("dynamic_threshold_percentile", 0.0)

        # Recon branch step gating: only apply corrections for the last N% of steps
        recon_step_ratio = kwargs.pop("recon_step_ratio", 0.0)

        # Skip recon correction on the very last denoising step
        skip_last_step_correction = kwargs.pop("skip_last_step_correction", False)

        # Ensure eval mode
        was_training = self.training
        self.eval()

        # Resolve effective dtype: honour torch.amp.autocast if active
        if torch.is_autocast_enabled():
            _eval_dtype = torch.get_autocast_gpu_dtype()
        else:
            _eval_dtype = self.mixed_dtype

        # VAE native dtype — run encode/decode without autocast for fp32 precision
        _vae_dtype = next(self.image_vae.parameters()).dtype

        try:
            # 1. Setup & Dimensions
            real_bsz = input_imgs.shape[0]
            num_sample_views = T_in + T_out
            total_scenes = real_bsz * num_sample_views * self.num_tasks

            # Helper: Indices Grid (Batch, Task, View) to match forward pass logic
            all_indices = torch.arange(total_scenes, device=self.device).long()
            indices_grid = all_indices.view(real_bsz, self.num_tasks, num_sample_views)

            input_indices = indices_grid[:, :, :T_in].reshape(-1)
            output_indices = indices_grid[:, :, T_in:].reshape(-1)

            # Helpers for alignment
            def _flatten(x):
                return einops.rearrange(x, "b t c h w -> (b t) c h w")

            def _interleave_tasks(latent_list, bsz, num_views):
                # Rearrange (B*T) inputs into (Batch, Task, View) sequence
                unflattened = [einops.rearrange(t, '(b t) ... -> b t ...', b=bsz, t=num_views) for t in latent_list]
                return einops.rearrange(torch.stack(unflattened, dim=1), 'b task t ... -> (b task t) ...')

            def _repeat_shared(latent, bsz, num_tasks, num_views):
                # Repeat shared inputs (Rays/PC) for every task
                unflattened = einops.rearrange(latent, '(b t) ... -> b t ...', b=bsz, t=num_views)
                repeated = unflattened.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1)
                return einops.rearrange(repeated, 'b task t ... -> (b task t) ...')

            # 2. Encode Inputs (Conditioning) — VAE in native (fp32) precision
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                # Task 0: Images
                img_latents = self.prepare_img_latents(_flatten(input_imgs), real_bsz, _vae_dtype, self.device)
                input_latents_list = [img_latents.to(_eval_dtype)]

                # Task 1: Coords (if applicable)
                if self.num_tasks > 1:
                    if input_coord_map is None:
                        raise ValueError("input_coord_map is required when num_tasks > 1")
                    coord_latents = self.prepare_img_latents(_flatten(input_coord_map), real_bsz, _vae_dtype, self.device, vae=self.depth_vae)
                    input_latents_list.append(coord_latents.to(_eval_dtype))

            # Align inputs to (Batch, Task, View)
            aligned_input_latents = _interleave_tasks(input_latents_list, real_bsz, T_in)

            # 3. Prepare Shared Context (Rays & PC)
            input_ray_embeds = _flatten(self._process_rays(input_rays)).to(dtype=_eval_dtype)
            output_ray_embeds = _flatten(self._process_rays(output_rays)).to(dtype=_eval_dtype)

            # Task-specific conditioning latents — VAE in native (fp32) precision
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                input_pc_latents = self.prepare_img_latents(_flatten(input_pc_renders), real_bsz, _vae_dtype, self.device)
                if self.use_vae_skip and output_warp_masks is not None:
                    output_pc_latents = self.encode_pc_with_features(
                        _flatten(output_pc_renders), _flatten(output_warp_masks), dtype=_vae_dtype
                    )
                else:
                    output_pc_latents = self.prepare_img_latents(_flatten(output_pc_renders), real_bsz, _vae_dtype, self.device)
            input_pc_latents = input_pc_latents.to(_eval_dtype)
            output_pc_latents = output_pc_latents.to(_eval_dtype)

            # Save clean PC latents for observed-region blending before masking
            obs_blend_latents = None
            obs_blend_mask = None
            if obs_latent_blend and output_warp_masks is not None:
                obs_blend_latents = output_pc_latents.clone()
                latent_h, latent_w = obs_blend_latents.shape[-2:]
                obs_blend_mask = F.interpolate(
                    _flatten(output_warp_masks), size=(latent_h, latent_w), mode='area'
                )

            # Mask PC latents in unobserved regions using warp masks
            out_mask_soft = None  # Soft [0,1] mask at latent res, reused for observation mask channel
            if self.mask_pc_latents and output_warp_masks is not None:
                latent_h, latent_w = output_pc_latents.shape[-2:]
                out_mask_soft = F.interpolate(
                    _flatten(output_warp_masks), size=(latent_h, latent_w), mode='area'
                )
                in_mask_soft = F.interpolate(
                    _flatten(input_warp_masks), size=(latent_h, latent_w), mode='area'
                )
                # Threshold raw warp mask at full res, then downsample to latent res
                out_mask = F.interpolate(
                    (_flatten(output_warp_masks) > 0.5).float(), size=(latent_h, latent_w), mode='area'
                )
                out_mask = (out_mask > 0.5).to(output_pc_latents.dtype)
                in_mask = F.interpolate(
                    (_flatten(input_warp_masks) > 0.5).float(), size=(latent_h, latent_w), mode='area'
                )
                in_mask = (in_mask > 0.5).to(input_pc_latents.dtype)

                output_pc_latents = self._mask_latents(output_pc_latents, out_mask)
                input_pc_latents = self._mask_latents(input_pc_latents, in_mask)

            input_cond_list = [input_pc_latents]
            output_cond_list = [output_pc_latents]

            if self.num_tasks > 1:
                # Task 1 (Coords): loo_pred coord maps encoded by depth VAE as conditioning
                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    input_coord_cond = self.prepare_img_latents(
                        _flatten(input_coord_map), real_bsz, _vae_dtype, self.device, vae=self.depth_vae
                    )
                    if output_coord_map is not None:
                        if self.use_depth_vae_skip and output_warp_masks is not None:
                            output_coord_cond = self.encode_coord_with_features(
                                _flatten(output_coord_map), _flatten(output_warp_masks), dtype=_vae_dtype
                            )
                        else:
                            output_coord_cond = self.prepare_img_latents(
                                _flatten(output_coord_map), real_bsz, _vae_dtype, self.device, vae=self.depth_vae
                            )
                    else:
                        # Fallback: use zeros if output coord map not available
                        output_coord_cond = torch.zeros_like(output_pc_latents)
                input_coord_cond = input_coord_cond.to(_eval_dtype)
                output_coord_cond = output_coord_cond.to(_eval_dtype)
                # Coord maps share the same forward-warp coverage as PC renders
                if self.mask_pc_latents and output_warp_masks is not None:
                    output_coord_cond = self._mask_coord_latents(output_coord_cond, out_mask)
                    input_coord_cond = self._mask_coord_latents(input_coord_cond, in_mask)
                input_cond_list.append(input_coord_cond)
                output_cond_list.append(output_coord_cond)

            # Align Shared Data
            aligned_all_rays = torch.zeros((total_scenes, *input_ray_embeds.shape[1:]), device=self.device, dtype=_eval_dtype)
            aligned_all_rays[input_indices] = _repeat_shared(input_ray_embeds, real_bsz, self.num_tasks, T_in)
            aligned_all_rays[output_indices] = _repeat_shared(output_ray_embeds, real_bsz, self.num_tasks, T_out)

            aligned_all_pc = torch.zeros((total_scenes, *input_pc_latents.shape[1:]), device=self.device, dtype=_eval_dtype)
            aligned_all_pc[input_indices] = _interleave_tasks(input_cond_list, real_bsz, T_in)
            aligned_all_pc[output_indices] = _interleave_tasks(output_cond_list, real_bsz, T_out)

            # Build unconditional tensors for conditioning-based CFG
            do_cfg = not use_autoguidance and guidance_scale > 1.0
            if do_cfg:
                uncond_all_pc = torch.zeros_like(aligned_all_pc)
                if self.use_learned_null_token:
                    task0_all = indices_grid[:, 0, :].reshape(-1)
                    uncond_all_pc[task0_all] = self.pc_null_token.to(dtype=_eval_dtype).expand(
                        len(task0_all), -1, aligned_all_pc.shape[2], aligned_all_pc.shape[3])
                    if self.num_tasks > 1:
                        task1_all = indices_grid[:, 1, :].reshape(-1)
                        uncond_all_pc[task1_all] = self.coord_null_token.to(dtype=_eval_dtype).expand(
                            len(task1_all), -1, aligned_all_pc.shape[2], aligned_all_pc.shape[3])

            # 4. Observation Mask Channel & Text Embeds
            obs_mask_channel = torch.zeros((total_scenes, 1, img_latents.shape[-2], img_latents.shape[-1]), device=self.device, dtype=_eval_dtype)
            obs_mask_channel[input_indices] = 1.0
            if self.use_observation_mask and out_mask_soft is not None:
                obs_mask_channel[output_indices] = _repeat_shared(
                    out_mask_soft.to(_eval_dtype), real_bsz, self.num_tasks, T_out
                )

            if do_cfg:
                uncond_obs_mask = torch.zeros_like(obs_mask_channel)

            prompt_embeds = self._encode_text(self.default_text_prompt).repeat(total_scenes, 1, 1)
            full_prompt = prompt_embeds  # No text-level CFG needed; conditioning-based CFG is used instead

            # 5. Sampling Loop
            num_steps = num_inference_steps or 25
            self.val_scheduler.set_timesteps(num_steps, device=self.device)

            # Random Noise for targets (generated for all tasks)
            # Shape matches total number of target slots: (RealBatch * Tasks * T_out)
            num_targets = len(output_indices)

            if self.v_as_flow_prediction:
                if self.init_from_pc_renders:
                    # Flow matching with non-Gaussian source: start from PC render latents (matches training)
                    latents = aligned_all_pc[output_indices].clone().to(dtype=_eval_dtype)
                else:
                    # Flow matching: start from pure noise (t=1)
                    latents = torch.randn((num_targets, self.latent_channels, img_latents.shape[-2], img_latents.shape[-1]),
                                          device=self.device, dtype=_eval_dtype, generator=generator)
            else:
                latents = torch.randn((num_targets, self.latent_channels, img_latents.shape[-2], img_latents.shape[-1]),
                                      device=self.device, dtype=_eval_dtype, generator=generator) * self.val_scheduler.init_noise_sigma

            timesteps = self.val_scheduler.timesteps
            _final_pixel_corrected = None  # set on last step for pixel-space recon

            for step_idx, t in enumerate(timesteps):
                if not self.v_as_flow_prediction:
                    t_in = torch.full((total_scenes,), t, device=self.device, dtype=torch.long)
                    t_in[input_indices] = 0 # Condition is always clean
                else:
                    t_fm = (1 / num_steps) * step_idx
                    alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=_eval_dtype)
                    alpha_t = alphas_cumprod.sqrt()
                    sigma_t = (1.0 - alphas_cumprod).sqrt()
                    dm_timesteps = self.noise_scheduler.timesteps

                    dm_to_fm = lambda t : alpha_t[t] / (alpha_t[t] + sigma_t[t])
                    dm_timesteps_continuous = dm_to_fm(dm_timesteps)

                    t_discrete = torch.searchsorted(dm_timesteps_continuous, t_fm)
                    t_discrete = t_discrete.clamp(0, len(dm_timesteps) - 1)
                    t_discrete = dm_timesteps[t_discrete].to(device=self.device, dtype=torch.long)

                    t_in = torch.full((total_scenes,), t_discrete, device=self.device, dtype=torch.long)
                    t_in[input_indices] = 0

                # Construct Full Model Input
                current_full_latents = torch.zeros((total_scenes, self.latent_channels, *latents.shape[2:]), device=self.device, dtype=_eval_dtype)

                # Insert aligned inputs (images/coords) into input slots
                current_full_latents[input_indices] = aligned_input_latents
                # Insert noisy targets into output slots
                if self.v_as_flow_prediction:
                    latents = (alpha_t[t_discrete] + sigma_t[t_discrete]) * latents

                current_full_latents[output_indices] = latents.to(dtype=_eval_dtype)

                model_in_cond = torch.cat([current_full_latents, aligned_all_rays, obs_mask_channel, aligned_all_pc], dim=1)

                if do_cfg:
                    model_in_uncond = torch.cat([current_full_latents, aligned_all_rays, uncond_obs_mask, uncond_all_pc], dim=1)
                    model_in = torch.cat([model_in_uncond, model_in_cond], dim=0)
                    t_in_cfg = torch.cat([t_in, t_in], dim=0)
                    prompt_cfg = torch.cat([full_prompt, full_prompt], dim=0)
                    unet_output = self.unet(model_in, t_in_cfg, encoder_hidden_states=prompt_cfg).sample
                    uncond_out, cond_out = unet_output.chunk(2)
                    delta = cond_out - uncond_out
                    if guidance_unobs is not None and out_mask_soft is not None:
                        scale_map = torch.full_like(uncond_out, guidance_scale)
                        obs_scale = out_mask_soft * guidance_scale + (1 - out_mask_soft) * guidance_unobs
                        obs_scale_expanded = _repeat_shared(obs_scale, real_bsz, self.num_tasks, T_out)
                        obs_scale_expanded = obs_scale_expanded.expand_as(uncond_out[output_indices])
                        scale_map[output_indices] = obs_scale_expanded.to(scale_map.dtype)
                        unet_output = uncond_out + scale_map * delta
                    else:
                        unet_output = uncond_out + guidance_scale * delta
                    if guidance_rescale > 0.0:
                        if guidance_rescale_unobs is not None and out_mask_soft is not None:
                            rescale_mask = torch.ones(total_scenes, 1, *unet_output.shape[2:],
                                                      device=self.device, dtype=unet_output.dtype)
                            out_rescale_mask = _repeat_shared(out_mask_soft, real_bsz, self.num_tasks, T_out)
                            rescale_mask[output_indices] = out_rescale_mask.to(rescale_mask.dtype)
                            unet_output = self._rescale_noise_cfg_regional(
                                unet_output, cond_out, guidance_rescale, rescale_mask,
                                guidance_rescale_unobs=guidance_rescale_unobs)
                        else:
                            unet_output = self._rescale_noise_cfg(unet_output, cond_out, guidance_rescale)
                elif use_autoguidance:
                    unet_output = self.unet(model_in_cond, t_in, encoder_hidden_states=full_prompt).sample
                    cond_ref = unet_output  # save pre-guidance prediction for rescale
                    guide_output = guide_unet(model_in_cond, t_in, encoder_hidden_states=full_prompt).sample
                    delta = unet_output - guide_output
                    # Threshold the guidance delta before amplification to prevent
                    # spatial outliers from being scaled up and causing burnt patches
                    if dynamic_threshold_p > 0.0:
                        delta = self._threshold_delta(delta, dynamic_threshold_p)
                    if autoguidance_scale_unobs is not None and out_mask_soft is not None:
                        # Spatially-varying guidance: observed regions get autoguidance_scale,
                        # unobserved regions get autoguidance_scale_unobs
                        scale_map = torch.full_like(unet_output, autoguidance_scale)
                        obs_scale = out_mask_soft * autoguidance_scale + (1 - out_mask_soft) * autoguidance_scale_unobs
                        obs_scale_expanded = _repeat_shared(obs_scale, real_bsz, self.num_tasks, T_out)
                        obs_scale_expanded = obs_scale_expanded.expand_as(unet_output[output_indices])
                        scale_map[output_indices] = obs_scale_expanded.to(scale_map.dtype)
                        unet_output = guide_output + scale_map * delta
                        if guidance_rescale > 0.0:
                            # Region-wise rescale: all-ones for input views, soft mask for output views
                            rescale_mask = torch.ones(total_scenes, 1, *unet_output.shape[2:],
                                                      device=self.device, dtype=unet_output.dtype)
                            out_rescale_mask = _repeat_shared(out_mask_soft, real_bsz, self.num_tasks, T_out)
                            rescale_mask[output_indices] = out_rescale_mask.to(rescale_mask.dtype)
                            unet_output = self._rescale_noise_cfg_regional(
                                unet_output, cond_ref, guidance_rescale, rescale_mask,
                                guidance_rescale_unobs=guidance_rescale_unobs)
                    else:
                        unet_output = guide_output + autoguidance_scale * delta
                        if guidance_rescale > 0.0:
                            if guidance_rescale_unobs is not None and out_mask_soft is not None:
                                rescale_mask = torch.ones(total_scenes, 1, *unet_output.shape[2:],
                                                          device=self.device, dtype=unet_output.dtype)
                                out_rescale_mask = _repeat_shared(out_mask_soft, real_bsz, self.num_tasks, T_out)
                                rescale_mask[output_indices] = out_rescale_mask.to(rescale_mask.dtype)
                                unet_output = self._rescale_noise_cfg_regional(
                                    unet_output, cond_ref, guidance_rescale, rescale_mask,
                                    guidance_rescale_unobs=guidance_rescale_unobs)
                            else:
                                unet_output = self._rescale_noise_cfg(unet_output, cond_ref, guidance_rescale)
                else:
                    unet_output = self.unet(model_in_cond, t_in, encoder_hidden_states=full_prompt).sample

                model_pred = unet_output[output_indices, :self.latent_channels]

                x0_delta = None  # initialized for Euler step check below
                is_last_step = (step_idx == len(timesteps) - 1)

                # --- Recon side-branch x0 correction at inference ---
                if self.recon_last_step_only:
                    # Pixel-branch last-step-only: avoids lossy re-encodes at intermediate steps
                    apply_recon = is_last_step
                else:
                    recon_start_step = int(len(timesteps) * (1.0 - recon_step_ratio))
                    apply_recon = (recon_step_ratio <= 0.0) or (step_idx >= recon_start_step)
                    if is_last_step and skip_last_step_correction:
                        apply_recon = False
                if self.dual_recon and geo_index_map is not None and apply_recon:
                    with torch.no_grad():
                        # Cached temporal embedding; for CFG the UNet saw 2x batch, take cond half
                        cached_temb = self.unet._cached_temb
                        if do_cfg:
                            cached_temb = cached_temb[total_scenes:]  # conditional half

                        # Index to task 0 output views only
                        task0_out_local = torch.arange(len(output_indices), device=self.device).view(
                            real_bsz, self.num_tasks, T_out)[:, 0, :].reshape(-1)

                        temb = cached_temb[output_indices][task0_out_local]

                        # Reconstruct x0 from model_pred for task 0 views.
                        pred_task0 = model_pred[task0_out_local]
                        latents_task0 = latents[task0_out_local]

                        if self.v_as_flow_prediction:
                            t_for_recon = torch.full((len(task0_out_local),), t_discrete.item(), device=self.device, dtype=torch.long)
                        else:
                            t_for_recon = torch.full((len(task0_out_local),), t.item(), device=self.device, dtype=torch.long)

                        alphas_cp = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=_eval_dtype)
                        t_val = t_for_recon[0]
                        ab = alphas_cp[t_val]
                        at = ab.sqrt().view(1, 1, 1, 1)
                        st = (1.0 - ab).sqrt().view(1, 1, 1, 1)

                        pred_type = getattr(self.noise_scheduler.config, "prediction_type", "epsilon")
                        if pred_type == "v_prediction":
                            x0_unet = at * latents_task0 - st * pred_task0
                        elif pred_type == "epsilon":
                            x0_unet = (latents_task0 - st * pred_task0) / at.clamp(min=1e-8)
                        else:
                            x0_unet = pred_task0

                        if self.recon_pixel_space:
                            # ===== Pixel-space recon at inference =====
                            scaling = self.image_vae.config.scaling_factor
                            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                                decoded_x0 = self.decode_with_skip(x0_unet.to(_vae_dtype) / scaling, dtype=_vae_dtype)  # [-1, 1]

                            pc_renders_flat = _flatten(output_pc_renders)
                            warp_masks_flat = _flatten(output_warp_masks)
                            source_images = _flatten(input_imgs)  # (B*T_in, 3, H, W) [-1,1]

                            if getattr(self, 'recon_use_plucker', False):
                                target_plucker = _flatten(output_rays).to(dtype=decoded_x0.dtype)
                                source_plucker = _flatten(input_rays).to(dtype=decoded_x0.dtype)
                            else:
                                target_plucker = None
                                source_plucker = None

                            pixel_delta = self.recon_branch(
                                decoded_x0, pc_renders_flat, warp_masks_flat,
                                source_images, geo_index_map, out_out_index_map, temb, T_out,
                                target_plucker=target_plucker,
                                source_plucker=source_plucker,
                            )

                            obs_mask_pixel = self._compute_pixel_obs_mask(output_warp_masks)
                            pixel_delta = pixel_delta * obs_mask_pixel
                            pixel_corrected = decoded_x0 + pixel_delta

                            if is_last_step:
                                # Last step: stash corrected pixels directly, skip lossy VAE round-trip
                                _final_pixel_corrected = pixel_corrected
                            else:
                                # Re-encode corrected pixels back to latent space
                                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                                    corrected_latent = self.image_vae.encode(
                                        pixel_corrected.to(dtype=_vae_dtype)
                                    ).latent_dist.mode() * scaling
                                corrected_latent = corrected_latent.to(_eval_dtype)

                                # x0_delta in latent space for scheduler step
                                x0_delta = corrected_latent - x0_unet

                                if not self.v_as_flow_prediction:
                                    x0_corrected = x0_unet + x0_delta
                                    if pred_type == "v_prediction":
                                        corrected_pred = (at * latents_task0 - x0_corrected) / st.clamp(min=1e-8)
                                    elif pred_type == "epsilon":
                                        corrected_pred = (latents_task0 - at * x0_corrected) / st.clamp(min=1e-8)
                                    else:
                                        corrected_pred = x0_corrected
                                    model_pred[task0_out_local] = corrected_pred
                        else:
                            # ===== Latent-space recon at inference =====
                            # Decode x0_unet for branch input (already inside torch.no_grad())
                            scaling = self.image_vae.config.scaling_factor
                            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                                decoded_x0 = self.decode_with_skip(x0_unet.to(_vae_dtype) / scaling, dtype=_vae_dtype)  # [-1, 1]

                            pc_renders_flat = _flatten(output_pc_renders)  # (B*T_out, 3, H, W)
                            warp_masks_flat = _flatten(output_warp_masks)  # (B*T_out, 1, H, W)
                            source_images = _flatten(input_imgs)  # (B*T_in, 3, H, W)

                            x0_delta = self.recon_branch(
                                decoded_x0, pc_renders_flat, warp_masks_flat,
                                source_images, x0_unet,
                                geo_index_map, out_out_index_map, temb, T_out,
                            )

                            # Only correct observed regions — unobserved regions are purely UNet-generated
                            obs_mask_recon = self.compute_observation_mask(output_warp_masks)
                            x0_delta = x0_delta * obs_mask_recon

                            if not self.v_as_flow_prediction:
                                # Standard diffusion: convert corrected x0 back to prediction space
                                x0_corrected = x0_unet + x0_delta
                                if pred_type == "v_prediction":
                                    corrected_pred = (at * latents_task0 - x0_corrected) / st.clamp(min=1e-8)
                                elif pred_type == "epsilon":
                                    corrected_pred = (latents_task0 - at * x0_corrected) / st.clamp(min=1e-8)
                                else:
                                    corrected_pred = x0_corrected
                                model_pred[task0_out_local] = corrected_pred
                        # else: x0_delta applied directly to velocity below

                if self.v_as_flow_prediction:
                    dt = 1.0 / num_steps

                    alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=_eval_dtype)
                    alpha_bar = alphas_cumprod.gather(0, t_discrete)
                    alpha_t = alpha_bar.sqrt().view(-1, 1, 1, 1)
                    sigma_t = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)

                    velocity = alpha_t * (latents - model_pred) - sigma_t * (latents + model_pred)

                    # Apply recon branch correction directly in velocity space
                    if self.dual_recon and x0_delta is not None:
                        correction = ((alpha_t + sigma_t) / sigma_t.clamp(min=1e-8) * x0_delta).to(velocity.dtype)
                        velocity[task0_out_local] = velocity[task0_out_local] + correction

                    latents = latents / (alpha_t + sigma_t)
                    latents = latents + dt * velocity
                else:
                    # Standard diffusion scheduler step
                    latents = self.val_scheduler.step(model_pred, t, latents).prev_sample

                # Observed-region latent blending (inpainting-style)
                if obs_blend_latents is not None and step_idx < len(timesteps) - 1:
                    task0_idx = torch.arange(num_targets, device=self.device).view(
                        real_bsz, self.num_tasks, T_out)[:, 0, :].reshape(-1)
                    blend_noise = torch.randn_like(obs_blend_latents)
                    if self.v_as_flow_prediction:
                        # Compute next step's t_discrete via the same fm→dm mapping the loop uses
                        next_t_fm = (1 / num_steps) * (step_idx + 1)
                        ac = self.noise_scheduler.alphas_cumprod.to(device=self.device, dtype=_eval_dtype)
                        at = ac.sqrt()
                        st = (1.0 - ac).sqrt()
                        dm_ts = self.noise_scheduler.timesteps
                        dm_cont = at[dm_ts] / (at[dm_ts] + st[dm_ts])
                        next_td = torch.searchsorted(dm_cont, next_t_fm).clamp(0, len(dm_ts) - 1)
                        next_td = dm_ts[next_td].to(device=self.device, dtype=torch.long)
                        t_expand = next_td.expand(obs_blend_latents.shape[0])
                        noisy_clean = self.noise_scheduler.add_noise(obs_blend_latents, blend_noise, t_expand)
                        normalizer = (ac[next_td].sqrt() + (1 - ac[next_td]).sqrt())
                        noisy_clean = noisy_clean / normalizer
                    else:
                        next_t = timesteps[step_idx + 1]
                        t_expand = next_t.expand(obs_blend_latents.shape[0])
                        noisy_clean = self.noise_scheduler.add_noise(obs_blend_latents, blend_noise, t_expand)
                    blended = obs_blend_mask * noisy_clean + (1 - obs_blend_mask) * latents[task0_idx]
                    latents[task0_idx] = (obs_blend_strength * blended + (1 - obs_blend_strength) * latents[task0_idx]).to(latents.dtype)

            # 6. Decode & Return
            # latents shape: (RealBatch * Tasks * T_out, C, H, W)
            # We separate tasks to decode them with the correct VAEs
            reshaped_latents = einops.rearrange(latents, "(b task t) c h w -> b task t c h w", b=real_bsz, task=self.num_tasks, t=T_out)

            results = {}


            # --- Decode Task 0: Images ---
            if _final_pixel_corrected is not None:
                # Use pixel-corrected images directly (avoids lossy VAE round-trip)
                imgs = (_final_pixel_corrected / 2 + 0.5).clamp(0, 1)
            else:
                task0_latents = einops.rearrange(reshaped_latents[:, 0], "b t c h w -> (b t) c h w")
                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    imgs = self.decode_with_skip(task0_latents / self.image_vae.config.scaling_factor, dtype=_vae_dtype)
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
            results['imgs'] = einops.rearrange(imgs, "(b t) c h w -> b t c h w", t=T_out)

            # Post-hoc confidence: run DINOv2+DPT on decoded images + PC renders
            if self.use_posthoc_confidence and self._posthoc_confidence_model is not None:
                pc_flat = einops.rearrange(output_pc_renders, "b t c h w -> (b t) c h w").float().clamp(0, 1)
                if pc_flat.shape[-2:] != imgs.shape[-2:]:
                    pc_flat = F.interpolate(pc_flat, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
                # Flatten warp masks for pixel branch (None-safe)
                obs_mask_flat = None
                if output_warp_masks is not None:
                    obs_mask_flat = einops.rearrange(output_warp_masks, "b t c h w -> (b t) c h w").float()
                posthoc_conf = self._posthoc_confidence_model(
                    imgs.float(), pc_flat, output_size=imgs.shape[-2:], obs_mask=obs_mask_flat,
                )
                results['confidence'] = einops.rearrange(posthoc_conf, "(b t) c h w -> b t c h w", t=T_out)

            # --- Decode Task 1: Coords (if applicable) ---
            if self.num_tasks > 1:
                task1_latents = einops.rearrange(reshaped_latents[:, 1], "b t c h w -> (b t) c h w")

                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    coords = self.decode_depth_with_skip(task1_latents / self.depth_vae.config.scaling_factor, dtype=_vae_dtype)
                coords = coords[:, :3]  # depth_vae outputs 4 channels; keep only XYZ
                coords = coords.clamp(-1.0, 1.0)
                results['coords'] = einops.rearrange(coords, "(b t) c h w -> b t c h w", t=T_out)

                return results # Return dict for multi-task

            # Always return dict for consistency (breaking change from tensor return)
            return results

        finally:
            if was_training: self.train()


def build_model(opt: Options) -> GenerativeReconstructionDiffusionModel:
    return GenerativeReconstructionDiffusionModel(opt)
