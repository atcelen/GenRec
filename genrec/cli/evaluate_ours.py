"""
Shared evaluation engine for GenRec.

Computes L1, PSNR, SSIM, and LPIPS split by global/observed/unobserved regions,
plus dataset-level FID, FD-DINOv2, CMMD, CLIP-IQA, and TSED, and saves per-scene
results and a summary JSON.

Uses RealEstate10KDatasetWithEulerRaw or DL3DVDataset + model.evaluate_with_raw_images()
so that geometry processing (DA3, coord maps, PC renders) is handled
internally by the model.

Usage:
    # RE10K evaluation (default):
    python -m genrec.cli.evaluate_ours \
        --config configs/re10k.yaml \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir ./ours_eval_results \
        --scene_list assets/re10k_test_scenes_100.txt \
        --step_size 10

    # DL3DV evaluation:
    python -m genrec.cli.evaluate_ours \
        --config configs/dl3dv.yaml \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir ./dl3dv_eval_results \
        --dataset_type dl3dv \
        --data_path /path/to/DL3DV-10K \
        --longest_side 616 \
        --step_size 4 \
        --scene_list assets/re10k_test_scenes_100.txt \
        --use_gt_poses

    # Multi-GPU:
    torchrun --nproc_per_node=4 -m genrec.cli.evaluate_ours \
        --config configs/re10k.yaml \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir ./ours_eval_results
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from genrec.data.dataset import (
    RealEstate10KDatasetWithEulerRaw,
    DL3DVDataset,
    MipNeRF360Dataset,
    collate_fn_raw,
)
from genrec.models import UNetMVMM2DConditionModel
from genrec.models.geo_model import GeometryModel
from genrec.models.model import GenerativeReconstructionDiffusionModel
from genrec.utils.config import load_experiment_config
from genrec.weights import resolve_weights
from genrec.utils.coord_metrics import compute_all_coord_metrics
from genrec.utils.eval_metrics import (
    compute_masked_l1,
    compute_masked_psnr,
    compute_masked_ssim,
    compute_masked_lpips,
    compute_confidence_weighted_lpips,
    compute_spearman_rank_correlation,
    compute_ause,
    plot_metric_histograms,
    FIDAccumulator,
    FDDINOv2Accumulator,
    CMMDAccumulator,
    CLIPIQAAccumulator,
    TSEDAccumulator,
)


# ------------------------------
# Distributed Utilities
# ------------------------------

def setup_distributed() -> Tuple[int, int, torch.device]:
    """Setup distributed environment, falling back to single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device(f"cuda:{local_rank}")
    else:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ------------------------------
# Two-pass long-horizon helpers (used only when --two_pass_interp is set)
# ------------------------------

def _slerp_R(R_a: torch.Tensor, R_b: torch.Tensor, t: float) -> torch.Tensor:
    """SLERP between two rotation matrices.

    R_a, R_b: (3, 3) rotation matrices. t: scalar in [0, 1]. Returns (3, 3).
    Routes via axis-angle of R_b @ R_a.T.
    """
    R_rel = R_b @ R_a.transpose(-1, -2)
    cos_theta = ((R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta.abs() < 1e-6:
        return R_a.clone()

    # Axis from skew-symmetric part
    axis = torch.tensor([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ], device=R_a.device, dtype=R_a.dtype) / (2.0 * torch.sin(theta))

    theta_t = t * theta
    K = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], device=R_a.device, dtype=R_a.dtype)
    R_step = (
        torch.eye(3, device=R_a.device, dtype=R_a.dtype)
        + torch.sin(theta_t) * K
        + (1.0 - torch.cos(theta_t)) * (K @ K)
    )
    return R_step @ R_a


def _interp_w2c_lerp_slerp(
    w2c_a: torch.Tensor,
    w2c_b: torch.Tensor,
    num_intermediate: int,
) -> torch.Tensor:
    """Generate `num_intermediate` w2c matrices strictly between w2c_a and w2c_b.

    Uses LERP on translation + SLERP on rotation in the camera-to-world frame
    (more natural for a camera trajectory than interpolating w2c directly).
    Endpoints are NOT included; t values are k/(num_intermediate+1) for k=1..N.
    """
    c2w_a = torch.inverse(w2c_a.float())
    c2w_b = torch.inverse(w2c_b.float())
    R_a, t_a = c2w_a[:3, :3], c2w_a[:3, 3]
    R_b, t_b = c2w_b[:3, :3], c2w_b[:3, 3]

    out = []
    for k in range(1, num_intermediate + 1):
        s = k / float(num_intermediate + 1)
        R_k = _slerp_R(R_a, R_b, s)
        t_k = (1.0 - s) * t_a + s * t_b
        c2w_k = torch.eye(4, device=w2c_a.device, dtype=R_a.dtype)
        c2w_k[:3, :3] = R_k
        c2w_k[:3, 3] = t_k
        out.append(torch.inverse(c2w_k))
    return torch.stack(out, dim=0)  # (N, 4, 4)


def _plucker_rays_for_views(
    w2c: torch.Tensor,         # (T, 4, 4)
    intrinsics: torch.Tensor,  # (T, 3, 3)
    H: int, W: int,
) -> torch.Tensor:
    """Compute Plucker rays at (H, W) for a stack of cameras. Returns (T, 6, H, W).

    Mirrors the per-view ray construction in geo_model.batched_forward.
    """
    from genrec.utils.cam_ops import get_plucker_rays, get_ray_directions
    T = w2c.shape[0]
    ray_dirs = torch.stack([
        get_ray_directions(
            H=H, W=W,
            focal=(intrinsics[t, 0, 0].item(), intrinsics[t, 1, 1].item()),
            principal=(intrinsics[t, 0, 2].item(), intrinsics[t, 1, 2].item()),
        )
        for t in range(T)
    ], dim=0).to(w2c.device)  # (T, H, W, 3)
    c2w = torch.inverse(w2c.float())
    plucker = get_plucker_rays(directions=ray_dirs, c2w=c2w, keepdim=True)  # (T, H, W, 6)
    return einops.rearrange(plucker, "T H W C -> T C H W")


# ------------------------------
# Metrics saving
# ------------------------------

def save_metrics(
    results: List[Dict],
    output_dir: str,
    filename: str = "metrics.json",
    dataset_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save evaluation metrics to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    def _stats(values):
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary = {"num_samples": len(results)}

    regions = ["global", "observed", "unobserved"]
    metrics = ["l1", "psnr", "ssim", "lpips"]
    for region in regions:
        region_dict = {}
        for metric in metrics:
            key = f"{region}/{metric}"
            values = [r[key] for r in results if key in r]
            if values:
                region_dict[metric] = _stats(values)
        if region_dict:
            summary[region] = region_dict

    obs_pct_values = [r["obs_pct"] for r in results if "obs_pct" in r]
    if obs_pct_values:
        summary["obs_coverage"] = _stats(obs_pct_values)

    time_values = [r["inference_time"] for r in results if "inference_time" in r]
    if time_values:
        summary["inference_time"] = _stats(time_values)
        summary["inference_time"]["total"] = float(np.sum(time_values))

    for metric in ["psnr", "ssim", "lpips"]:
        values = [r[metric] for r in results if metric in r]
        if values:
            summary[metric] = _stats(values)

    # Confidence metrics summary
    conf_metrics = ["spearman", "ause", "psnr", "ssim", "lpips"]
    conf_dict = {}
    for metric in conf_metrics:
        key = f"confidence/{metric}"
        values = [r[key] for r in results if r.get(key, 0) != 0]
        if values:
            conf_dict[metric] = _stats(values)
    if conf_dict:
        summary["confidence"] = conf_dict

    # TSED summary
    tsed_values = [r["tsed"] for r in results if r.get("tsed", 0) != 0]
    if tsed_values:
        summary["tsed"] = {
            "tsed": _stats(tsed_values),
            "median_sed": _stats([r["tsed/median_sed"] for r in results if r.get("tsed", 0) != 0]),
            "n_matches": _stats([r["tsed/n_matches"] for r in results if r.get("tsed", 0) != 0]),
        }

    # Coordinate map metrics summary
    coord_keys = [k for k in results[0].keys() if k.startswith("coord/")]
    if coord_keys:
        coord_dict = {}
        for key in coord_keys:
            values = [r[key] for r in results if key in r and r[key] != 0]
            if values:
                coord_dict[key.replace("coord/", "")] = _stats(values)
        if coord_dict:
            summary["coord_metrics"] = coord_dict

    if dataset_metrics:
        summary["dataset_metrics"] = dataset_metrics

    summary["per_sample"] = results

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to: {output_path}")


# ------------------------------
# Main evaluator
# ------------------------------

class OursEvaluator:
    def __init__(
        self,
        cfg,
        model_opts,
        device: torch.device,
        checkpoint_path: str,
        output_dir: str,
        scene_list: Optional[str],
        step_size: int,
        seed: int,
        num_inference_steps: int = 25,
        rank: int = 0,
        world_size: int = 1,
        save_visualizations: bool = False,
        save_pointclouds: bool = False,
        guide_checkpoint: Optional[str] = None,
        autoguidance_scale: float = 1.5,
        autoguidance_scale_unobs: Optional[float] = None,
        confidence_checkpoint: Optional[str] = None,
        guidance_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        guidance_rescale_unobs: Optional[float] = None,
        guidance_unobs: Optional[float] = None,
        conf_mask_thresh: float = 0.0,
        obs_masks_dir: Optional[str] = None,
        use_gt_poses: bool = False,
        skip_coord_metrics: bool = False,
        save_images: bool = False,
        save_video: bool = False,
        video_fps: int = 10,
        depth_filter_thresh: float = 0.5,
        depth_grad_thresh: float = 0.0,
        obs_blend: bool = False,
        obs_blend_strength: float = 0.8,
        dynamic_threshold_percentile: float = 0.0,
        recon_step_ratio: float = 0.0,
        skip_last_step_correction: bool = False,
        sample_from_end: bool = False,
        hires_warp_res: int = 0,
        source_only_da3: bool = False,
        dataset_type: str = "re10k",
        data_path: Optional[str] = None,
        longest_side: Optional[int] = None,
        interpolation: bool = False,
        factor: Optional[int] = None,
        no_vae_skip: bool = False,
        num_heldout: int = 0,
        two_pass_interp: bool = False,
    ):
        self.cfg = cfg
        self.model_opts = model_opts
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.scene_list = scene_list
        self.step_size = step_size
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_master = rank == 0
        self.save_visualizations = save_visualizations
        self.save_pointclouds = save_pointclouds
        self.guide_checkpoint = guide_checkpoint
        self.autoguidance_scale = autoguidance_scale
        self.autoguidance_scale_unobs = autoguidance_scale_unobs
        self.confidence_checkpoint = confidence_checkpoint
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.guidance_rescale_unobs = guidance_rescale_unobs
        self.guidance_unobs = guidance_unobs
        self.conf_mask_thresh = conf_mask_thresh
        # Observation-mask source for the observed/unobserved metric split:
        #   "paper" (default) -> auto-download the paper's masks from HuggingFace
        #   "none"            -> use the internally computed warp masks
        #   a .npz file       -> masks packed by scene name
        #   a directory       -> per-scene <safe_name>.npy files
        if obs_masks_dir == "none":
            obs_masks_dir = None
        elif obs_masks_dir == "paper":
            from genrec.weights import resolve_obs_masks
            obs_masks_dir = resolve_obs_masks(dataset_type)
            if self.is_master:
                print(f"Using paper observation masks: {obs_masks_dir}")
        self.obs_masks_dir = obs_masks_dir
        self._obs_masks_npz = None  # lazily opened .npz handle
        self.use_gt_poses = use_gt_poses
        self.skip_coord_metrics = skip_coord_metrics
        self.save_images = save_images
        self.save_video = save_video
        self.video_fps = video_fps
        self.depth_filter_thresh = depth_filter_thresh
        self.depth_grad_thresh = depth_grad_thresh
        self.obs_blend = obs_blend
        self.obs_blend_strength = obs_blend_strength
        self.dynamic_threshold_percentile = dynamic_threshold_percentile
        self.recon_step_ratio = recon_step_ratio
        self.skip_last_step_correction = skip_last_step_correction
        self.sample_from_end = sample_from_end
        self.hires_warp_res = hires_warp_res
        self.source_only_da3 = source_only_da3
        self.dataset_type = dataset_type
        self.data_path_override = data_path
        self.longest_side = longest_side
        self.interpolation = interpolation
        self.factor = factor
        self.no_vae_skip = no_vae_skip
        self.num_heldout = num_heldout
        self.two_pass_interp = two_pass_interp
        if self.num_heldout > 0 and not self.use_gt_poses:
            if rank == 0:
                print(
                    "[num_heldout] Forcing return_camera_poses=True; held-out frames "
                    "require GT camera poses for downstream 3DGS evaluation."
                )

        if source_only_da3:
            if not skip_coord_metrics:
                if rank == 0:
                    print("[source_only_da3] Auto-setting skip_coord_metrics=True (no valid target depth)")
                self.skip_coord_metrics = True
            if not use_gt_poses and rank == 0:
                print("[WARNING] source_only_da3=True but use_gt_poses=False — GT poses are required")

        self.mp_mode = cfg.val.mixed_precision.lower()
        self.amp_dtype = (
            torch.bfloat16 if self.mp_mode == "bf16"
            else torch.float16 if self.mp_mode == "fp16"
            else torch.float32
        )

        if self.is_master:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[rank {rank}] Output: {self.output_dir}")

        self.model = None

    def build_model(self):
        """Load generative model (with integrated geometry model)."""
        if self.is_master:
            print("Loading generative model...")

        model = GenerativeReconstructionDiffusionModel(self.model_opts, mp_mode=self.mp_mode)
        model.to(self.device)
        model.eval()

        # Resolve registry name / URL / local path -> local file (auto-downloads from HF).
        checkpoint_path = resolve_weights(self.checkpoint_path)
        if self.is_master:
            print(f"Loading weights from {checkpoint_path}...")

        if str(checkpoint_path).endswith(".safetensors"):
            # Release checkpoint format: flat state dict, DA3 weights
            # intentionally omitted (they are fetched from the HuggingFace hub
            # at model construction).
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            real_missing = [k for k in missing if not k.startswith("geometry_model.")]
            if real_missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint does not match the model architecture "
                    f"(check the `model:` section of the config).\n"
                    f"Missing keys: {real_missing[:10]}{'...' if len(real_missing) > 10 else ''}\n"
                    f"Unexpected keys: {list(unexpected)[:10]}{'...' if len(unexpected) > 10 else ''}"
                )
        else:
            # Legacy training-checkpoint format (.pth).
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if "ema_model" in ckpt:
                state_dict = ckpt["ema_model"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith("module."):
                    name = name[7:]
                if name.startswith("_orig_mod."):
                    name = name[10:]
                if name.startswith("ema_model."):
                    name = name[10:]
                new_state_dict[name] = v

            try:
                model.load_state_dict(new_state_dict, strict=True)
            except Exception as e:
                if self.is_master:
                    print(f"Strict loading failed, retrying strict=False: {e}")
                model.load_state_dict(new_state_dict, strict=False)

        self.model = model

        if self.no_vae_skip:
            self.model.use_vae_skip = False
            self.model.use_depth_vae_skip = False
            for vae in (self.model.image_vae, getattr(self.model, "depth_vae", None)):
                if vae is not None and hasattr(vae, "peft_config") and vae.peft_config:
                    vae.disable_adapters()
            if self.is_master:
                print("[ablation] VAE skip connections + decoder LoRA disabled")

        # Cast VAEs to fp32 for higher-precision encode/decode
        self.model.image_vae.to(torch.float32)
        if hasattr(self.model, "depth_vae") and self.model.depth_vae is not None:
            self.model.depth_vae.to(torch.float32)

        # Guide UNet for autoguidance
        self.guide_unet = None
        if self.guide_checkpoint:
            if self.is_master:
                print(f"Loading guide UNet from {self.guide_checkpoint}...")
            guide_ckpt = torch.load(
                resolve_weights(self.guide_checkpoint), map_location=self.device, weights_only=False
            )

            if "ema_model" in guide_ckpt:
                guide_sd = guide_ckpt["ema_model"]
            elif "model" in guide_ckpt:
                guide_sd = guide_ckpt["model"]
            else:
                guide_sd = guide_ckpt

            # Extract UNet keys and strip prefixes
            unet_sd = {}
            for k, v in guide_sd.items():
                name = k
                for prefix in ("module.", "_orig_mod.", "ema_model."):
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                if name.startswith("unet."):
                    unet_sd[name[5:]] = v

            guide_unet = UNetMVMM2DConditionModel.from_config(self.model.unet.config)
            info = guide_unet.load_state_dict(unet_sd, strict=False)
            if self.is_master:
                print(f"Guide UNet loaded: {len(unet_sd)} keys, missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")
            guide_unet.to(self.device).eval()
            guide_unet.requires_grad_(False)
            self.guide_unet = guide_unet

        # Post-hoc confidence network
        if self.confidence_checkpoint:
            if self.is_master:
                print(f"Loading post-hoc confidence model from {self.confidence_checkpoint}...")
            self.model.load_posthoc_confidence_model(self.confidence_checkpoint)

        # LPIPS network
        if self.is_master:
            from kiui.lpips import LPIPS
            self.lpips_net = LPIPS(net="vgg").to(self.device).eval()
        else:
            self.lpips_net = None

        # Spatial LPIPS for confidence-weighted LPIPS (only when needed)
        self.spatial_lpips_net = None
        if self.confidence_checkpoint and self.is_master:
            import lpips as lpips_pkg
            self.spatial_lpips_net = lpips_pkg.LPIPS(net="vgg", spatial=True).to(self.device).eval()

    @torch.no_grad()
    def _run_two_pass(self, raw_images, scene_ids, T_in, T_out, gt_extrinsics, gt_intrinsics, eval_kwargs):
        """Two-pass long-horizon inference.

        Pass 1 runs the standard `evaluate_with_raw_images` with the dataset/config
        T_in + T_out (= trained max views). Pass 2 runs (V_anchor - 1) calls, each
        with T_in=2 input views = the two bracketing anchors [anchor_i, anchor_{i+1}]
        (= the two existing frames closest to all interp cameras in this window),
        and T_out=trained_max-2 output views generated at LERP+SLERP intermediate
        cameras between those two anchors. The point cloud used as conditioning in
        every pass-2 call is still rendered from the original source-view depth
        (no predicted depth is ever fed back).

        Returns the pass-1 result dict (so existing downstream metric code is
        unchanged) augmented with three extra keys:
          - long_horizon_imgs:        (B, V_long, 3, H, W) full stitched RGB in [0,1]
          - long_horizon_extrinsics:  (B, V_long, 4, 4)
          - long_horizon_intrinsics:  (B, V_long, 3, 3)
        where V_long = V_anchor + (V_anchor - 1) * N_interp.
        """
        from genrec.models.geo_model import GeometryModel as _GM
        # Pass-2 windows: T_in=2 (the two bracketing anchors), T_out fills the
        # rest up to trained max. Stays fully in-distribution on attention seq
        # length AND on T_in (trained T_in ∈ {1, 2}).
        trained_max_views = self.model_opts.num_views
        WINDOW_T_IN = 2
        WINDOW_T_OUT = trained_max_views - WINDOW_T_IN
        if WINDOW_T_OUT < 1:
            raise ValueError(
                f"Two-pass interp needs at least 1 generated frame per pair, but "
                f"WINDOW_T_OUT={WINDOW_T_OUT} (trained_max={trained_max_views}). "
                f"Use a model with num_views >= 3."
            )
        N_INTERP = WINDOW_T_OUT

        # ----- Pass 1 -----
        pass1 = self.model.evaluate_with_raw_images(
            raw_images=raw_images,
            scene_ids=scene_ids,
            T_in=T_in,
            T_out=T_out,
            num_inference_steps=self.num_inference_steps,
            conf_mask_thresh=self.conf_mask_thresh,
            gt_extrinsics=gt_extrinsics,
            gt_intrinsics=gt_intrinsics,
            depth_filter_thresh=self.depth_filter_thresh,
            depth_grad_thresh=self.depth_grad_thresh,
            source_only_da3=self.source_only_da3,
            **eval_kwargs,
        )

        # Pass-1 anchors: 2 source views + T_out generated frames.
        anchor_extr = pass1["extrinsics"].float()       # (B, V_anchor, 4, 4)
        anchor_intr = pass1["intrinsics"].float()       # (B, V_anchor, 3, 3)
        src_depth = pass1["depth"].float()              # (B, V_anchor, H_d, W_d)
        processed = pass1["processed_images"].float()   # (B, V_anchor, 3, H_p, W_p) in [-1, 1]
        gen_imgs_01 = pass1["imgs"].float()             # (B, T_out, 3, H_g, W_g) in [0, 1]
        norm_min = pass1.get("coord_norm_min", None)    # (B,) or None
        norm_max = pass1.get("coord_norm_max", None)

        B = anchor_extr.shape[0]
        V_anchor = T_in + T_out
        H_p, W_p = processed.shape[-2:]

        # Resize generated frames to processed-image resolution if needed (VAE
        # latent decode usually matches, but keep this safe).
        if gen_imgs_01.shape[-2:] != (H_p, W_p):
            gen_imgs_01 = F.interpolate(
                gen_imgs_01.flatten(0, 1), size=(H_p, W_p),
                mode="bilinear", align_corners=False,
            ).reshape(B, T_out, 3, H_p, W_p)

        # Anchor RGB tensor in [-1, 1]: source views from `processed`, generated
        # views replaced with pass-1 outputs.
        anchor_imgs_neg1 = processed.clone()
        anchor_imgs_neg1[:, T_in:] = gen_imgs_01 * 2.0 - 1.0

        # Source-image RGB in [0, 1] for forward warp (uses processed source views).
        src_imgs_01 = (processed[:, :T_in] + 1.0) * 0.5  # (B, T_in, 3, H_p, W_p)

        # Source-view depth + intrinsics: depth comes back at original image
        # resolution; resize to (H_p, W_p) if needed and rescale intrinsics
        # accordingly so render_views_gpu has consistent (H, W) across all inputs.
        src_depth_in = src_depth[:, :T_in]              # (B, T_in, H_d, W_d)
        if src_depth_in.shape[-2:] != (H_p, W_p):
            H_d, W_d = src_depth_in.shape[-2:]
            sx = W_p / float(W_d)
            sy = H_p / float(H_d)
            src_depth_in = F.interpolate(
                src_depth_in.unsqueeze(2).float(), size=(H_p, W_p),
                mode="bilinear", align_corners=False,
            ).squeeze(2)  # (B, T_in, H_p, W_p)
            src_intr_p = anchor_intr[:, :T_in].clone()
            src_intr_p[..., 0, :] *= sx
            src_intr_p[..., 1, :] *= sy
            anchor_intr_p = anchor_intr.clone()
            anchor_intr_p[..., 0, :] *= sx
            anchor_intr_p[..., 1, :] *= sy
        else:
            src_intr_p = anchor_intr[:, :T_in].clone()
            anchor_intr_p = anchor_intr

        src_extr = anchor_extr[:, :T_in]  # (B, T_in, 4, 4)

        # Per-batch normalization bounds (with full-tensor fallback).
        if norm_min is None or norm_max is None:
            norm_min = src_depth_in.new_zeros(B)
            norm_max = src_depth_in.new_ones(B)
        norm_min = norm_min.to(src_extr.device).float()
        norm_max = norm_max.to(src_extr.device).float()

        # Source coord maps (normalized in [-1, 1]) per scene — fed to
        # render_views_gpu so output coord maps come out in the same coord frame.
        src_coords_norm = []
        for b in range(B):
            sc = _GM.unproject_depth_to_world_coords(
                depth=src_depth_in[b],
                intrinsics=src_intr_p[b],
                extrinsics=src_extr[b],
                normalize=True,
                norm_min=norm_min[b],
                norm_max=norm_max[b],
                log_scale_coords=getattr(self.model, "log_scale_coords", False),
            )
            src_coords_norm.append(sc)
        src_coords_norm = torch.stack(src_coords_norm, dim=0)  # (B, T_in, 3, H_p, W_p)

        long_imgs_01 = []
        long_extr = []
        long_intr = []

        # Pass 2: for each adjacent anchor pair, generate N_INTERP intermediate frames.
        for i in range(V_anchor - 1):
            # Interpolating cameras between anchor i and anchor i+1.
            interp_extr_per_b = []
            for b in range(B):
                ie = _interp_w2c_lerp_slerp(
                    anchor_extr[b, i], anchor_extr[b, i + 1], N_INTERP,
                )
                interp_extr_per_b.append(ie)
            interp_extr = torch.stack(interp_extr_per_b, dim=0).to(src_extr.device)  # (B, N_INTERP, 4, 4)
            # Intrinsics: copy from anchor i for all interp cameras.
            interp_intr = anchor_intr_p[:, i:i + 1].expand(-1, N_INTERP, -1, -1).contiguous()

            # Build T_in=2 input view block: the two bracketing anchors only.
            # Source views are intentionally dropped — the only inputs are the
            # two closest existing frames to the interp cameras.
            in_extr = torch.cat([anchor_extr[:, i:i + 1], anchor_extr[:, i + 1:i + 2]], dim=1)
            in_intr = torch.cat([anchor_intr_p[:, i:i + 1], anchor_intr_p[:, i + 1:i + 2]], dim=1)
            in_imgs = torch.cat([
                anchor_imgs_neg1[:, i:i + 1],
                anchor_imgs_neg1[:, i + 1:i + 2],
            ], dim=1)  # (B, 2, 3, H_p, W_p) in [-1, 1]

            # Render source PC at all 8 (= T_in_window + T_out_window) cameras.
            all_target_extr = torch.cat([in_extr, interp_extr], dim=1)  # (B, 8, 4, 4)
            all_target_intr = torch.cat([in_intr, interp_intr], dim=1)

            pc_renders_per_b = []
            warp_masks_per_b = []
            coord_maps_per_b = []
            for b in range(B):
                warp_result = _GM.render_views_gpu(
                    source_depth=src_depth_in[b],
                    source_w2c=src_extr[b],
                    source_intrinsics=src_intr_p[b],
                    target_w2c=all_target_extr[b],
                    target_intrinsics=all_target_intr[b],
                    source_images=src_imgs_01[b],
                    source_coords=src_coords_norm[b],
                    foreground_masking=False,
                    cameraray_filtering=False,
                    depth_filter_thresh=self.depth_filter_thresh,
                    depth_grad_thresh=self.depth_grad_thresh,
                )
                pc_renders_per_b.append(warp_result["images"])  # (8, 3, H_p, W_p)
                warp_masks_per_b.append(warp_result["masks"])    # (8, 1, H_p, W_p)
                coord_maps_per_b.append(warp_result["coords"])   # (8, 3, H_p, W_p)
            all_pc_renders = torch.stack(pc_renders_per_b, dim=0)
            all_warp_masks = torch.stack(warp_masks_per_b, dim=0)
            all_coord_maps = torch.stack(coord_maps_per_b, dim=0)

            in_pc_renders = all_pc_renders[:, :WINDOW_T_IN]
            out_pc_renders = all_pc_renders[:, WINDOW_T_IN:]
            in_warp_masks = all_warp_masks[:, :WINDOW_T_IN]
            out_warp_masks = all_warp_masks[:, WINDOW_T_IN:]
            in_coord_map = all_coord_maps[:, :WINDOW_T_IN].to(dtype=self.model.mixed_dtype)
            out_coord_map = all_coord_maps[:, WINDOW_T_IN:].to(dtype=self.model.mixed_dtype)

            # Plucker rays at (H_p, W_p).
            in_rays_per_b = [_plucker_rays_for_views(in_extr[b], in_intr[b], H_p, W_p) for b in range(B)]
            out_rays_per_b = [_plucker_rays_for_views(interp_extr[b], interp_intr[b], H_p, W_p) for b in range(B)]
            in_rays = torch.stack(in_rays_per_b, dim=0)
            out_rays = torch.stack(out_rays_per_b, dim=0)

            # Direct call to model.evaluate (skips geometry pipeline + DA3).
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                pair = self.model.evaluate(
                    input_imgs=in_imgs,
                    input_rays=in_rays,
                    input_pc_renders=in_pc_renders,
                    output_rays=out_rays,
                    output_pc_renders=out_pc_renders,
                    input_coord_map=in_coord_map,
                    output_coord_map=out_coord_map,
                    input_warp_masks=in_warp_masks,
                    output_warp_masks=out_warp_masks,
                    T_in=WINDOW_T_IN,
                    T_out=WINDOW_T_OUT,
                    num_inference_steps=self.num_inference_steps,
                    **eval_kwargs,
                )
            pair_imgs_01 = pair["imgs"].float()  # (B, N_INTERP, 3, H_g, W_g) in [0, 1]
            if pair_imgs_01.shape[-2:] != (H_p, W_p):
                pair_imgs_01 = F.interpolate(
                    pair_imgs_01.flatten(0, 1), size=(H_p, W_p),
                    mode="bilinear", align_corners=False,
                ).reshape(B, N_INTERP, 3, H_p, W_p)

            # Append to trajectory: anchor i (only on first iter), then N_INTERP frames.
            if i == 0:
                long_imgs_01.append((anchor_imgs_neg1[:, 0:1] + 1.0) * 0.5)
                long_extr.append(anchor_extr[:, 0:1])
                long_intr.append(anchor_intr_p[:, 0:1])
            long_imgs_01.append(pair_imgs_01)
            long_extr.append(interp_extr)
            long_intr.append(interp_intr)
            long_imgs_01.append((anchor_imgs_neg1[:, i + 1:i + 2] + 1.0) * 0.5)
            long_extr.append(anchor_extr[:, i + 1:i + 2])
            long_intr.append(anchor_intr_p[:, i + 1:i + 2])

        long_imgs_01 = torch.cat(long_imgs_01, dim=1).clamp(0, 1)
        long_extr = torch.cat(long_extr, dim=1)
        long_intr = torch.cat(long_intr, dim=1)

        pass1["long_horizon_imgs"] = long_imgs_01
        pass1["long_horizon_extrinsics"] = long_extr
        pass1["long_horizon_intrinsics"] = long_intr
        return pass1

    @torch.no_grad()
    def _recompute_warp_masks_hires(self, eval_results, T_in, T_out):
        """Re-run forward warp at higher resolution and NN-downsample masks.

        Matches GEN3C's approach of warping at 704x1280 then downsampling,
        which produces denser coverage masks.
        """
        from genrec.utils.forward_warp import forward_warp, reliable_depth_mask_range_batch

        B = eval_results["depth"].shape[0]
        depth = eval_results["depth"]           # (B, T, H_orig, W_orig)
        extr = eval_results["extrinsics"]       # (B, T, 4, 4)
        intr = eval_results["intrinsics"]       # (B, T, 3, 3)
        H_orig, W_orig = depth.shape[2], depth.shape[3]

        # Compute target resolution preserving aspect ratio
        longest = max(H_orig, W_orig)
        scale = self.hires_warp_res / longest
        H_hi = int(round(H_orig * scale / 8)) * 8  # keep divisible by 8
        W_hi = int(round(W_orig * scale / 8)) * 8

        all_masks = []
        for b in range(B):
            scene_depth = depth[b]          # (T, H_orig, W_orig)
            scene_extr = extr[b].float()    # (T, 4, 4)
            scene_intr = intr[b].float()    # (T, 3, 3)

            # Upsample source depth to high resolution
            src_depth = scene_depth[:T_in]  # (T_in, H_orig, W_orig)
            src_depth_hi = F.interpolate(
                src_depth.unsqueeze(1), size=(H_hi, W_hi),
                mode="bilinear", align_corners=False,
            ).squeeze(1)  # (T_in, H_hi, W_hi)

            # Rescale intrinsics to high resolution
            sx, sy = W_hi / W_orig, H_hi / H_orig
            src_intr_hi = scene_intr[:T_in].clone()
            src_intr_hi[:, 0, :] *= sx
            src_intr_hi[:, 1, :] *= sy
            tgt_intr_hi = scene_intr[T_in:T_in + T_out].clone()
            tgt_intr_hi[:, 0, :] *= sx
            tgt_intr_hi[:, 1, :] *= sy

            # Depth validity + edge filter at high resolution
            depth_mask = (src_depth_hi > 0) & torch.isfinite(src_depth_hi)
            depth_mask = depth_mask.unsqueeze(1).float()  # (T_in, 1, H_hi, W_hi)
            if self.depth_filter_thresh > 0:
                reliable = reliable_depth_mask_range_batch(
                    src_depth_hi, window_size=5, ratio_thresh=self.depth_filter_thresh,
                ).float()
                depth_mask = depth_mask * reliable

            # Dummy single-channel frame (mask output doesn't depend on content)
            dummy_frame = torch.ones(
                T_in, 1, H_hi, W_hi, device=depth.device, dtype=depth.dtype,
            )

            src_w2c = scene_extr[:T_in]
            tgt_w2c = scene_extr[T_in:T_in + T_out]

            masks_per_target = []
            for j in range(T_out):
                tgt_w2c_j = tgt_w2c[j:j+1].expand(T_in, -1, -1)
                tgt_K_j = tgt_intr_hi[j:j+1].expand(T_in, -1, -1)
                _, wmask, _, _, _ = forward_warp(
                    frame1=dummy_frame,
                    mask1=depth_mask,
                    depth1=src_depth_hi.unsqueeze(1),
                    w2c1=src_w2c,
                    w2c2=tgt_w2c_j,
                    intrinsic1=src_intr_hi,
                    intrinsic2=tgt_K_j,
                    is_image=False,
                    cameraray_filtering=False,
                    foreground_masking=False,
                    n_views=T_in,
                )
                masks_per_target.append(wmask[0])  # (1, H_hi, W_hi)

            # Stack and NN-downsample to original resolution
            hires_masks = torch.stack(masks_per_target, dim=0)  # (T_out, 1, H_hi, W_hi)
            lo_masks = F.interpolate(
                hires_masks, size=(H_orig, W_orig), mode="nearest",
            )  # (T_out, 1, H_orig, W_orig)
            all_masks.append(lo_masks)

        return torch.stack(all_masks, dim=0)  # (B, T_out, 1, H_orig, W_orig)

    def build_dataloader(self):
        """Build validation dataloader using RealEstate10KDatasetWithEulerRaw or DL3DVDataset."""
        T_in = self.model_opts.num_input_views
        T_out = self.model_opts.num_views - T_in

        if self.is_master:
            print(f"Building dataset ({self.dataset_type}): T_in={T_in}, T_out={T_out}, step_size={self.step_size}")

        # Resolve data path: CLI override > cfg.data.path > matching entry in cfg.data.datasets
        if self.data_path_override is not None:
            data_path = self.data_path_override
        elif hasattr(self.cfg.data, 'path'):
            data_path = self.cfg.data.path
        elif hasattr(self.cfg.data, 'datasets'):
            matching = [d for d in self.cfg.data.datasets if d.get('type') == self.dataset_type]
            if not matching:
                available = [d.get('type') for d in self.cfg.data.datasets]
                raise ValueError(
                    f"No dataset of type '{self.dataset_type}' in cfg.data.datasets "
                    f"(available: {available}). Pass --data_path or --dataset_type."
                )
            data_path = matching[0]['path']
        else:
            raise ValueError(
                "cfg.data.path not found and no cfg.data.datasets. "
                "Please specify --data_path explicitly."
            )

        # Scene files live directly under data_path (flat directory).
        common_kwargs = dict(
            data_path=data_path,
            train_scenes_path="",
            test_scenes_path="",
            sample_criterion="sequential",
            step_size=self.step_size,
            T_in=T_in,
            T_out=T_out,
            is_train=False,
            return_camera_poses=self.use_gt_poses or self.num_heldout > 0,
            return_depths=True,
            scene_list_file=self.scene_list,
            sample_from_end=self.sample_from_end,
        )

        if self.num_heldout > 0:
            common_kwargs["num_heldout"] = self.num_heldout

        # Interpolation mode: reorder frames so inputs are at trajectory endpoints
        if self.interpolation and T_in >= 2:
            common_kwargs["dynamic_T"] = True
            common_kwargs["T_in_range"] = (T_in, T_in)
            common_kwargs["num_views"] = T_in + T_out

        if self.longest_side is not None:
            common_kwargs["longest_side"] = self.longest_side

        if self.dataset_type == "dl3dv":
            dataset = DL3DVDataset(**common_kwargs)
        elif self.dataset_type == "mipnerf360":
            if self.factor is not None:
                common_kwargs["factor"] = self.factor
            # Always evaluate all scenes in the data_path — ignore --scene_list.
            common_kwargs["scene_list_file"] = None
            dataset = MipNeRF360Dataset(**common_kwargs)
        else:
            dataset = RealEstate10KDatasetWithEulerRaw(**common_kwargs)

        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn_raw,
        )
        if self.is_master:
            print(f"Dataset: {len(dataset)} scenes, DataLoader: {len(loader)} batches")
        return loader, T_in, T_out

    @torch.inference_mode()
    def run(self):
        """Main evaluation loop."""
        self.build_model()
        loader, T_in, T_out = self.build_dataloader()

        # Initialize dataset-level metric accumulators (master only)
        fid_acc = None
        fd_dino_acc = None
        cmmd_acc = None
        clip_iqa_acc = None
        if self.is_master:
            fid_acc = FIDAccumulator(device=str(self.device))
            fd_dino_acc = FDDINOv2Accumulator(device=str(self.device))
            cmmd_acc = CMMDAccumulator(device=str(self.device))
            clip_iqa_acc = CLIPIQAAccumulator(device=str(self.device))
            tsed_acc = TSEDAccumulator() if T_out >= 2 else None

        results = []
        all_obs_pct = []
        all_per_sample = {
            f"{region}/{metric}": []
            for region in ["global", "observed", "unobserved"]
            for metric in (["l1", "psnr", "ssim", "lpips"] if region == "global" else ["l1", "psnr", "ssim"])
        }

        successful = 0
        failed = 0

        progress = tqdm(loader, desc="Evaluating") if self.is_master else loader

        for batch_idx, batch in enumerate(progress):
            try:
                if batch is None:
                    failed += 1
                    continue

                scene_ids = batch.get("scene_ids", None)
                raw_images = batch.get("raw_images", None)
                if scene_ids is None or raw_images is None:
                    failed += 1
                    continue

                raw_images = raw_images.to(self.device, non_blocking=True)
                num_views = T_in + T_out
                raw_images = raw_images[:, :num_views]

                # Deterministic seed per sample
                current_seed = self.seed + batch_idx
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)

                # Extract GT camera poses if available
                gt_extrinsics = None
                gt_intrinsics = None
                if self.use_gt_poses and "w2c" in batch and "intrinsics" in batch:
                    gt_extrinsics = batch["w2c"][:, :num_views].to(self.device, non_blocking=True)
                    gt_intrinsics = batch["intrinsics"][:, :num_views].to(self.device, non_blocking=True)

                # Run model inference with integrated geometry processing
                torch.cuda.synchronize(self.device)
                t_start = time.perf_counter()
                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    eval_kwargs = {}
                    if self.guide_unet is not None:
                        eval_kwargs["guide_unet"] = self.guide_unet
                        eval_kwargs["autoguidance_scale"] = self.autoguidance_scale
                        if self.autoguidance_scale_unobs is not None:
                            eval_kwargs["autoguidance_scale_unobs"] = self.autoguidance_scale_unobs
                    if self.guidance_scale != 1.0:
                        eval_kwargs["guidance_scale"] = self.guidance_scale
                    if self.guidance_unobs is not None:
                        eval_kwargs["guidance_unobs"] = self.guidance_unobs
                    if self.guidance_rescale > 0.0:
                        eval_kwargs["guidance_rescale"] = self.guidance_rescale
                    if self.guidance_rescale_unobs is not None:
                        eval_kwargs["guidance_rescale_unobs"] = self.guidance_rescale_unobs
                    if self.obs_blend:
                        eval_kwargs["obs_latent_blend"] = True
                        eval_kwargs["obs_blend_strength"] = self.obs_blend_strength
                    if self.dynamic_threshold_percentile > 0.0:
                        eval_kwargs["dynamic_threshold_percentile"] = self.dynamic_threshold_percentile
                    if self.recon_step_ratio > 0.0:
                        eval_kwargs["recon_step_ratio"] = self.recon_step_ratio
                    if self.skip_last_step_correction:
                        eval_kwargs["skip_last_step_correction"] = True

                    if self.two_pass_interp:
                        eval_results = self._run_two_pass(
                            raw_images=raw_images,
                            scene_ids=scene_ids,
                            T_in=T_in,
                            T_out=T_out,
                            gt_extrinsics=gt_extrinsics,
                            gt_intrinsics=gt_intrinsics,
                            eval_kwargs=eval_kwargs,
                        )
                    else:
                        eval_results = self.model.evaluate_with_raw_images(
                            raw_images=raw_images,
                            scene_ids=scene_ids,
                            T_in=T_in,
                            T_out=T_out,
                            num_inference_steps=self.num_inference_steps,
                            conf_mask_thresh=self.conf_mask_thresh,
                            gt_extrinsics=gt_extrinsics,
                            gt_intrinsics=gt_intrinsics,
                                            depth_filter_thresh=self.depth_filter_thresh,
                            depth_grad_thresh=self.depth_grad_thresh,
                            source_only_da3=self.source_only_da3,
                            **eval_kwargs,
                        )

                torch.cuda.synchronize(self.device)
                inference_time = time.perf_counter() - t_start

                preds = eval_results["imgs"]  # (B, T_out, C, H, W)
                # GT from DA3-processed images
                gt = eval_results["processed_images"][:, T_in:]  # (B, T_out, 3, H, W) in [-1, 1]
                gt = (gt + 1.0) / 2.0  # -> [0, 1]
                output_pc_renders = eval_results["output_pc_renders"]  # (B, T_out, C, H, W)
                output_warp_masks = eval_results["output_warp_masks"]  # (B, T_out, 1, H, W)

                # Override with external observation masks if provided
                if self.obs_masks_dir:
                    scene_id = scene_ids[0] if isinstance(scene_ids, (list, tuple)) else scene_ids
                    safe_name = scene_id.replace('/', '_')
                    if str(self.obs_masks_dir).endswith(".npz"):
                        if self._obs_masks_npz is None:
                            self._obs_masks_npz = np.load(self.obs_masks_dir)
                        if safe_name not in self._obs_masks_npz:
                            raise KeyError(
                                f"Scene '{safe_name}' not in the observation-mask archive "
                                f"{self.obs_masks_dir}. The published masks only cover the "
                                f"paper's evaluation scenes — pass --obs_masks_dir none to "
                                f"use internally computed masks for other scenes."
                            )
                        loaded = self._obs_masks_npz[safe_name].astype(np.float32)  # (T_out, 1, H, W)
                    else:
                        mask_path = os.path.join(self.obs_masks_dir, f"{safe_name}.npy")
                        loaded = np.load(mask_path)  # (T_out, 1, H, W) float32
                    if loaded.shape[0] != T_out:
                        raise ValueError(
                            f"Observation masks for '{safe_name}' cover {loaded.shape[0]} target "
                            f"views but this run has T_out={T_out}. The published masks match the "
                            f"single-view extrapolation protocol only — for other protocols (e.g. "
                            f"--interpolation) pass --obs_masks_dir none or your own masks."
                        )
                    output_warp_masks = torch.from_numpy(loaded).unsqueeze(0).to(self.device)  # (1, T_out, 1, H, W)

                # Recompute warp masks at higher resolution (matches GEN3C pipeline)
                if self.hires_warp_res > 0:
                    output_warp_masks = self._recompute_warp_masks_hires(
                        eval_results, T_in, T_out,
                    )

                # Resize GT to match prediction resolution if needed
                if gt.shape[-2:] != preds.shape[-2:]:
                    B_gt, T_gt = gt.shape[:2]
                    gt_flat = gt.reshape(B_gt * T_gt, *gt.shape[2:])
                    gt_flat = F.interpolate(gt_flat, size=preds.shape[-2:], mode="bilinear", align_corners=False)
                    gt = gt_flat.reshape(B_gt, T_gt, *gt_flat.shape[1:])

                # Convert to float32 outside autocast
                output_pc_renders = output_pc_renders.float()
                preds_flat = einops.rearrange(preds, "B T C H W -> (B T) C H W").float().clamp(0, 1)
                gt_flat = einops.rearrange(gt, "B T C H W -> (B T) C H W").float().clamp(0, 1)

                # Raw binary observation mask from renderer (matches GEN3C)
                obs_mask = einops.rearrange(output_warp_masks, "B T C H W -> (B T) C H W").float()
                # Resize mask to match prediction resolution if needed
                if obs_mask.shape[-2:] != preds_flat.shape[-2:]:
                    obs_mask = F.interpolate(obs_mask, size=preds_flat.shape[-2:], mode="nearest")
                obs_pct = obs_mask[:, 0].mean(dim=[1, 2])  # (N,)

                unobs_mask = 1.0 - obs_mask
                ones_mask = torch.ones_like(obs_mask)

                # Compute metrics for 3 regions
                g_l1 = compute_masked_l1(preds_flat, gt_flat, ones_mask)
                g_psnr = compute_masked_psnr(preds_flat, gt_flat, ones_mask)
                g_ssim = compute_masked_ssim(preds_flat, gt_flat, ones_mask)

                o_l1 = compute_masked_l1(preds_flat, gt_flat, obs_mask)
                o_psnr = compute_masked_psnr(preds_flat, gt_flat, obs_mask)
                o_ssim = compute_masked_ssim(preds_flat, gt_flat, obs_mask)

                u_l1 = compute_masked_l1(preds_flat, gt_flat, unobs_mask)
                u_psnr = compute_masked_psnr(preds_flat, gt_flat, unobs_mask)
                u_ssim = compute_masked_ssim(preds_flat, gt_flat, unobs_mask)

                # LPIPS + dataset accumulators (master only)
                g_lpips_val = torch.zeros_like(g_l1)
                clip_iqa_scores = {}
                tsed_batch_scores = {}
                if self.is_master:
                    g_lpips_val = compute_masked_lpips(preds_flat, gt_flat, ones_mask, self.lpips_net)
                    fid_acc.update_real(gt_flat)
                    fid_acc.update_fake(preds_flat)
                    fd_dino_acc.update_real(gt_flat)
                    fd_dino_acc.update_fake(preds_flat)
                    if cmmd_acc is not None:
                        cmmd_acc.update_real(gt_flat)
                        cmmd_acc.update_fake(preds_flat)
                    if clip_iqa_acc is not None:
                        clip_iqa_scores = clip_iqa_acc.update(preds_flat)

                    # TSED: multiview photoconsistency
                    if tsed_acc is not None and "extrinsics" in eval_results:
                        # Output-view predicted images: (B, T_out, C, H, W)
                        pred_out = preds.float().clamp(0, 1)
                        # Extrinsics for output views only
                        ext_out = eval_results["extrinsics"][:, T_in:]  # (B, T_out, 4, 4)
                        # Scale intrinsics from DA3 process_res to prediction resolution
                        intr_out = eval_results["intrinsics"][:, T_in:].clone()  # (B, T_out, 3, 3)
                        da3_H, da3_W = eval_results["processed_images"].shape[-2:]
                        pred_H, pred_W = preds.shape[-2:]
                        intr_out[:, :, 0, :] *= pred_W / da3_W
                        intr_out[:, :, 1, :] *= pred_H / da3_H
                        tsed_batch_scores = tsed_acc.update(pred_out, intr_out, ext_out)

                # Confidence calibration metrics
                conf_spearman_val = 0.0
                conf_ause_val = 0.0
                cw_psnr_val = 0.0
                cw_ssim_val = 0.0
                cw_lpips_val = 0.0
                if "confidence" in eval_results:
                    conf_map = eval_results["confidence"]  # (B, T_out, 1, H, W)
                    conf_flat = einops.rearrange(conf_map, "B T C H W -> (B T) C H W").float()
                    if conf_flat.shape[-2:] != preds_flat.shape[-2:]:
                        conf_flat = F.interpolate(conf_flat, size=preds_flat.shape[-2:], mode="bilinear", align_corners=False)
                    pixel_error = (preds_flat - gt_flat).abs().mean(dim=1)  # (N, H, W)
                    conf_2d = conf_flat.squeeze(1)  # (N, H, W)
                    spearman_vals = compute_spearman_rank_correlation(conf_2d, pixel_error, subsample=4096)
                    ause_vals = compute_ause(conf_2d, pixel_error)
                    conf_spearman_val = float(spearman_vals.mean().item())
                    conf_ause_val = float(ause_vals.mean().item())

                    # Confidence-weighted quality metrics
                    cw_psnr = compute_masked_psnr(preds_flat, gt_flat, conf_flat)
                    cw_ssim = compute_masked_ssim(preds_flat, gt_flat, conf_flat)
                    cw_psnr_val = float(cw_psnr.mean().item())
                    cw_ssim_val = float(cw_ssim.mean().item())
                    cw_lpips_val = 0.0
                    if self.spatial_lpips_net is not None:
                        cw_lpips = compute_confidence_weighted_lpips(
                            preds_flat, gt_flat, conf_flat, self.spatial_lpips_net,
                        )
                        cw_lpips_val = float(cw_lpips.mean().item())

                # Coordinate map metrics
                coord_metrics = {}
                if not self.skip_coord_metrics and "coords" in eval_results:
                    coord_metrics = compute_all_coord_metrics(
                        pred_coords=eval_results["coords"],
                        gt_coord_maps=eval_results["coord_maps"],
                        extrinsics=eval_results["extrinsics"],
                        intrinsics=eval_results["intrinsics"],
                        warp_masks=output_warp_masks,
                        T_in=T_in, T_out=T_out,
                        coord_norm_min=eval_results.get("coord_norm_min"),
                        coord_norm_max=eval_results.get("coord_norm_max"),
                        log_scale_coords=eval_results.get("log_scale_coords", False),
                    )

                scene_id = scene_ids[0] if isinstance(scene_ids, (list, tuple)) else scene_ids
                result_dict = {
                    "scene_id": scene_id,
                    "inference_time": inference_time,
                    "obs_pct": float(obs_pct.mean().item()),
                    "psnr": float(g_psnr.mean().item()),
                    "ssim": float(g_ssim.mean().item()),
                    "lpips": float(g_lpips_val.mean().item()),
                    "global/l1": float(g_l1.mean().item()),
                    "global/psnr": float(g_psnr.mean().item()),
                    "global/ssim": float(g_ssim.mean().item()),
                    "global/lpips": float(g_lpips_val.mean().item()),
                    "observed/l1": float(o_l1.mean().item()),
                    "observed/psnr": float(o_psnr.mean().item()),
                    "observed/ssim": float(o_ssim.mean().item()),
                    "unobserved/l1": float(u_l1.mean().item()),
                    "unobserved/psnr": float(u_psnr.mean().item()),
                    "unobserved/ssim": float(u_ssim.mean().item()),
                    "clip_iqa/quality": clip_iqa_scores.get("quality", 0.0),
                    "clip_iqa/sharpness": clip_iqa_scores.get("sharpness", 0.0),
                    "clip_iqa/natural": clip_iqa_scores.get("natural", 0.0),
                    "confidence/spearman": conf_spearman_val,
                    "confidence/ause": conf_ause_val,
                    "confidence/psnr": cw_psnr_val,
                    "confidence/ssim": cw_ssim_val,
                    "confidence/lpips": cw_lpips_val,
                    "tsed": tsed_batch_scores.get("tsed", 0.0),
                    "tsed/median_sed": tsed_batch_scores.get("median_sed", 0.0),
                    "tsed/n_matches": tsed_batch_scores.get("n_matches", 0.0),
                }
                for k, v in coord_metrics.items():
                    result_dict[f"coord/{k}"] = v

                if self.is_master:
                    conf_str = ""
                    if conf_spearman_val != 0.0 or conf_ause_val != 0.0:
                        conf_str = (
                            f" | Spearman: {conf_spearman_val:.4f} | AUSE: {conf_ause_val:.4f}"
                            f" | CW-PSNR: {cw_psnr_val:.2f} | CW-SSIM: {cw_ssim_val:.4f}"
                            f" | CW-LPIPS: {cw_lpips_val:.4f}"
                        )
                    print(
                        f"Scene: {scene_id} | Time: {inference_time:.2f}s | Obs: {result_dict['obs_pct']*100:.1f}% | "
                        f"PSNR(G/O/U): {result_dict['global/psnr']:.2f}/{result_dict['observed/psnr']:.2f}/{result_dict['unobserved/psnr']:.2f} | "
                        f"SSIM(G/O/U): {result_dict['global/ssim']:.4f}/{result_dict['observed/ssim']:.4f}/{result_dict['unobserved/ssim']:.4f} | "
                        f"LPIPS: {result_dict['global/lpips']:.4f}{conf_str}"
                    )

                results.append(result_dict)

                # Accumulate for histograms
                all_obs_pct.append(obs_pct.cpu().numpy())
                all_per_sample["global/l1"].append(g_l1.cpu().numpy())
                all_per_sample["global/psnr"].append(g_psnr.cpu().numpy())
                all_per_sample["global/ssim"].append(g_ssim.cpu().numpy())
                all_per_sample["global/lpips"].append(g_lpips_val.cpu().numpy())
                all_per_sample["observed/l1"].append(o_l1.cpu().numpy())
                all_per_sample["observed/psnr"].append(o_psnr.cpu().numpy())
                all_per_sample["observed/ssim"].append(o_ssim.cpu().numpy())
                all_per_sample["unobserved/l1"].append(u_l1.cpu().numpy())
                all_per_sample["unobserved/psnr"].append(u_psnr.cpu().numpy())
                all_per_sample["unobserved/ssim"].append(u_ssim.cpu().numpy())

                successful += 1

            except Exception as e:
                print(f"[rank {self.rank}] Error on batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue

            # Save visualization (outside main try/except so inference errors
            # don't skip it, and visualization errors don't mask metrics)
            if self.save_visualizations and self.is_master:
                conf_for_vis = eval_results.get("confidence", None)
                self._save_visualization(preds_flat, gt_flat, output_pc_renders, scene_id, confidence=conf_for_vis, obs_mask=obs_mask)

            # Two-pass long-horizon: write the full stitched trajectory as PNGs
            # AND an MP4 fly-through so users can inspect the densified output.
            if self.two_pass_interp and self.is_master and "long_horizon_imgs" in eval_results:
                import cv2
                safe_scene = scene_id.replace("/", "_")
                lh_root = self.output_dir / "long_horizon"
                lh_dir = lh_root / safe_scene
                lh_dir.mkdir(parents=True, exist_ok=True)
                lh_imgs = eval_results["long_horizon_imgs"][0].clamp(0, 1).cpu().numpy()  # (V, 3, H, W)
                bgr_frames = []
                for t in range(lh_imgs.shape[0]):
                    arr = (lh_imgs[t].transpose(1, 2, 0) * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(lh_dir / f"frame_{t:03d}.png"), bgr)
                    bgr_frames.append(bgr)
                if bgr_frames:
                    h, w = bgr_frames[0].shape[:2]
                    mp4_path = lh_root / f"{safe_scene}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(mp4_path), fourcc, 10, (w, h))
                    for frame in bgr_frames:
                        writer.write(frame)
                    writer.release()

            if self.save_pointclouds and self.is_master:
                self._save_pointcloud(
                    eval_results, preds_flat, gt_flat, scene_id, T_in, T_out,
                    gt_depths=batch.get("depths"),
                    gt_w2c=batch.get("w2c"),
                    gt_intrinsics_raw=batch.get("intrinsics"),
                    gt_raw_images=batch.get("raw_images"),
                    heldout_raw_images=batch.get("heldout_raw_images"),
                    heldout_w2c=batch.get("heldout_w2c"),
                    heldout_intrinsics=batch.get("heldout_intrinsics"),
                    heldout_depths=batch.get("heldout_depths"),
                    heldout_frame_indices=batch.get("heldout_frame_indices"),
                )

            if self.save_images and self.is_master:
                import cv2
                img_dir = self.output_dir / "images"
                pred_dir = img_dir / "pred"
                gt_dir = img_dir / "gt"
                pred_dir.mkdir(parents=True, exist_ok=True)
                gt_dir.mkdir(parents=True, exist_ok=True)
                for t in range(preds_flat.shape[0]):
                    fname = f"{scene_id.replace('/', '_')}_{t:02d}.png"
                    pred_np = (preds_flat[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    gt_np = (gt_flat[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(str(pred_dir / fname), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(gt_dir / fname), cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))

            if self.save_video and self.is_master:
                import imageio
                video_dir = self.output_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                safe_scene = scene_id.replace("/", "_")

                # Source views from DA3-processed input frames, in [-1, 1].
                src = eval_results["processed_images"][:, :T_in]  # (B, T_in, 3, H_src, W_src)
                src = ((src + 1.0) / 2.0).clamp(0, 1)
                if src.shape[-2:] != preds.shape[-2:]:
                    B_s, T_s = src.shape[:2]
                    src = F.interpolate(
                        src.reshape(B_s * T_s, *src.shape[2:]),
                        size=preds.shape[-2:], mode="bilinear", align_corners=False,
                    ).reshape(B_s, T_s, *preds.shape[-3:])
                src_flat = einops.rearrange(src, "B T C H W -> (B T) C H W").float().clamp(0, 1)
                # Concatenate source + generated along the view dimension.
                video_frames = torch.cat([src_flat, preds_flat], dim=0).cpu().numpy()  # (T_in + V_gen, 3, H, W)

                mp4_path = video_dir / f"{safe_scene}.mp4"
                writer = imageio.get_writer(
                    str(mp4_path), fps=self.video_fps, codec="libx264",
                    quality=8, macro_block_size=8,
                )
                try:
                    for t in range(video_frames.shape[0]):
                        frame_hwc = np.transpose(
                            (video_frames[t] * 255).astype(np.uint8), (1, 2, 0)
                        )
                        writer.append_data(frame_hwc)
                finally:
                    writer.close()

        # --- Aggregate results ---
        if self.is_master:
            print(f"\nSuccessful: {successful}, Failed: {failed}")

            if results:
                # Inference time summary
                times = [r["inference_time"] for r in results]
                print(f"\nInference Time: {np.mean(times):.2f}s mean, {np.std(times):.2f}s std, {np.sum(times):.1f}s total")

                dataset_metrics = {}
                if fid_acc is not None:
                    fid = fid_acc.compute()
                    if fid is not None:
                        dataset_metrics["fid"] = fid
                if fd_dino_acc is not None:
                    fd = fd_dino_acc.compute()
                    if fd is not None:
                        dataset_metrics["fd_dinov2"] = fd
                if clip_iqa_acc is not None:
                    ciqa = clip_iqa_acc.compute()
                    if ciqa:
                        for k, v in ciqa.items():
                            dataset_metrics[f"clip_iqa_{k}"] = v
                if cmmd_acc is not None:
                    cmmd = cmmd_acc.compute()
                    if cmmd is not None:
                        dataset_metrics["cmmd"] = cmmd
                if tsed_acc is not None:
                    tsed_result = tsed_acc.compute()
                    if tsed_result is not None:
                        dataset_metrics["tsed"] = tsed_result["tsed"]
                        dataset_metrics["tsed_median_sed"] = tsed_result["median_sed"]
                        dataset_metrics["tsed_n_matches"] = tsed_result["n_matches"]
                        dataset_metrics["tsed_consistency_rate"] = tsed_result["consistency_rate"]

                # Print summary
                obs_mean = np.mean([r["obs_pct"] for r in results]) * 100
                print(f"\nObservation Coverage: {obs_mean:.1f}%")
                print(f"\n{'Region':<14} {'L1':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
                print("-" * 50)
                for region in ["global", "observed", "unobserved"]:
                    l1_v = np.mean([r[f"{region}/l1"] for r in results])
                    psnr_v = np.mean([r[f"{region}/psnr"] for r in results])
                    ssim_v = np.mean([r[f"{region}/ssim"] for r in results])
                    lpips_str = f"{np.mean([r[f'{region}/lpips'] for r in results]):8.4f}" if f"{region}/lpips" in results[0] else "     N/A"
                    print(f"  {region:<12} {l1_v:8.4f} {psnr_v:8.2f} {ssim_v:8.4f} {lpips_str}")

                if "fid" in dataset_metrics:
                    print(f"\n  FID:       {dataset_metrics['fid']:.2f}")
                if "fd_dinov2" in dataset_metrics:
                    print(f"  FD-DINOv2: {dataset_metrics['fd_dinov2']:.2f}")
                if "cmmd" in dataset_metrics:
                    print(f"  CMMD:      {dataset_metrics['cmmd']:.2f}")
                if "clip_iqa_quality" in dataset_metrics:
                    print(f"\n  CLIP-IQA Quality:   {dataset_metrics['clip_iqa_quality']:.4f}")
                    print(f"  CLIP-IQA Sharpness: {dataset_metrics['clip_iqa_sharpness']:.4f}")
                    print(f"  CLIP-IQA Natural:   {dataset_metrics['clip_iqa_natural']:.4f}")
                if "tsed" in dataset_metrics:
                    print(f"\n  TSED:              {dataset_metrics['tsed']:.4f}")
                    print(f"  Mean Median SED:   {dataset_metrics['tsed_median_sed']:.2f} px")
                    print(f"  Mean #Matches:     {dataset_metrics['tsed_n_matches']:.1f}")
                    print(f"  Consistency Rate:  {dataset_metrics['tsed_consistency_rate']:.4f}")

                # Confidence calibration summary
                spearman_vals = [r["confidence/spearman"] for r in results if r.get("confidence/spearman", 0) != 0]
                ause_vals = [r["confidence/ause"] for r in results if r.get("confidence/ause", 0) != 0]
                if spearman_vals:
                    print(f"\n  Confidence Spearman: {np.mean(spearman_vals):.4f} (std: {np.std(spearman_vals):.4f})")
                    print(f"  Confidence AUSE:     {np.mean(ause_vals):.4f} (std: {np.std(ause_vals):.4f})")
                    dataset_metrics["confidence_spearman"] = float(np.mean(spearman_vals))
                    dataset_metrics["confidence_ause"] = float(np.mean(ause_vals))

                    # Confidence-weighted quality metrics summary
                    cw_psnr_vals = [r["confidence/psnr"] for r in results if r.get("confidence/psnr", 0) != 0]
                    cw_ssim_vals = [r["confidence/ssim"] for r in results if r.get("confidence/ssim", 0) != 0]
                    cw_lpips_vals = [r["confidence/lpips"] for r in results if r.get("confidence/lpips", 0) != 0]
                    if cw_psnr_vals:
                        print(f"  Conf-Weighted PSNR:  {np.mean(cw_psnr_vals):.2f} (std: {np.std(cw_psnr_vals):.2f})")
                        print(f"  Conf-Weighted SSIM:  {np.mean(cw_ssim_vals):.4f} (std: {np.std(cw_ssim_vals):.4f})")
                        print(f"  Conf-Weighted LPIPS: {np.mean(cw_lpips_vals):.4f} (std: {np.std(cw_lpips_vals):.4f})")
                        dataset_metrics["confidence_psnr"] = float(np.mean(cw_psnr_vals))
                        dataset_metrics["confidence_ssim"] = float(np.mean(cw_ssim_vals))
                        dataset_metrics["confidence_lpips"] = float(np.mean(cw_lpips_vals))

                # Coordinate map metrics summary
                coord_l1_vals = [r["coord/coord_l1"] for r in results if "coord/coord_l1" in r]
                if coord_l1_vals:
                    print(f"\n  Coord L1:          {np.mean(coord_l1_vals):.4f} (std: {np.std(coord_l1_vals):.4f})")
                    coord_rmse_vals = [r["coord/coord_rmse"] for r in results if "coord/coord_rmse" in r]
                    print(f"  Coord RMSE:        {np.mean(coord_rmse_vals):.4f} (std: {np.std(coord_rmse_vals):.4f})")
                    depth_d1 = [r["coord/depth/delta_1"] for r in results if "coord/depth/delta_1" in r]
                    if depth_d1:
                        print(f"  Depth delta<1.25:  {np.mean(depth_d1):.4f}")
                        print(f"  Depth abs_rel:     {np.mean([r['coord/depth/abs_rel'] for r in results if 'coord/depth/abs_rel' in r]):.4f}")
                    pnp_rot = [r["coord/pnp_rot_err_mean"] for r in results if r.get("coord/pnp_n_valid", 0) > 0]
                    if pnp_rot:
                        print(f"  PnP rot err:       {np.mean(pnp_rot):.2f} deg")
                        pnp_trans = [r["coord/pnp_trans_err_mean"] for r in results if r.get("coord/pnp_n_valid", 0) > 0]
                        print(f"  PnP trans err:     {np.mean(pnp_trans):.4f}")
                        pnp_5 = [r["coord/pnp_5deg_5unit"] for r in results if r.get("coord/pnp_n_valid", 0) > 0]
                        print(f"  PnP 5deg/5unit:    {np.mean(pnp_5):.4f}")
                    mv_l1 = [r["coord/mv_3d_l1_mean"] for r in results if r.get("coord/mv_n_pairs", 0) > 0]
                    if mv_l1:
                        print(f"  MV 3D L1:          {np.mean(mv_l1):.4f}")
                        mv_reproj = [r["coord/mv_crossreproj_mean"] for r in results if r.get("coord/mv_n_pairs", 0) > 0]
                        print(f"  MV cross-reproj:   {np.mean(mv_reproj):.2f} px")

                save_metrics(results, str(self.output_dir), dataset_metrics=dataset_metrics or None)

                # Plot histograms
                if all_obs_pct:
                    cat_obs = np.concatenate(all_obs_pct)
                    cat_metrics = {k: np.concatenate(v) for k, v in all_per_sample.items() if v}
                    plot_metric_histograms(
                        obs_percentages=cat_obs,
                        metrics_dict=cat_metrics,
                        output_dir=str(self.output_dir),
                        prefix="ours_",
                    )
            else:
                print("No successful evaluations.")

    def _save_visualization(self, preds, gt, pc_renders, scene_id, confidence=None, obs_mask=None):
        import cv2
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # pc_renders: (B, T_out, C, H, W) -> (B*T_out, C, H, W)
        pc_flat = einops.rearrange(pc_renders, "B T C H W -> (B T) C H W").float().clamp(0, 1)

        # Prepare confidence if available: (B, T_out, 1, H, W) -> (B*T_out, 1, H, W)
        conf_flat = None
        if confidence is not None:
            conf_flat = einops.rearrange(confidence, "B T C H W -> (B T) C H W").float()
            if conf_flat.shape[-2:] != preds.shape[-2:]:
                conf_flat = F.interpolate(conf_flat, size=preds.shape[-2:], mode="bilinear", align_corners=False)

        # Build grid: each row is one output view, columns are the panels
        rows = []
        for i in range(preds.shape[0]):
            pc_np = (pc_flat[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_np = (gt[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred_np = (preds[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if pc_np.shape[:2] != gt_np.shape[:2]:
                pc_np = cv2.resize(pc_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_LINEAR)

            panels = [pc_np, gt_np, pred_np]

            # Observed-region L1 error heatmap (always shown if obs_mask available)
            if obs_mask is not None:
                error = (preds[i] - gt[i]).abs().mean(dim=0).cpu().numpy()  # (H, W)
                mask_np = obs_mask[i, 0].cpu().numpy()  # (H, W)
                if mask_np.shape != error.shape:
                    mask_np = cv2.resize(mask_np, (error.shape[1], error.shape[0]), interpolation=cv2.INTER_NEAREST)
                masked_error = error * mask_np
                error_norm = (masked_error / max(masked_error.max(), 1e-6) * 255).astype(np.uint8)
                error_color = cv2.applyColorMap(error_norm, cv2.COLORMAP_INFERNO)
                error_color = cv2.cvtColor(error_color, cv2.COLOR_BGR2RGB)
                # Black out unobserved regions (applyColorMap maps 0 to non-black)
                error_color[mask_np < 0.5] = 0
                panels.append(error_color)

            if conf_flat is not None:
                # Per-pixel error heatmap
                error = (preds[i] - gt[i]).abs().mean(dim=0).cpu().numpy()  # (H, W)
                error_norm = (error / max(error.max(), 1e-6) * 255).astype(np.uint8)
                error_color = cv2.applyColorMap(error_norm, cv2.COLORMAP_INFERNO)
                error_color = cv2.cvtColor(error_color, cv2.COLOR_BGR2RGB)

                conf = conf_flat[i, 0].cpu().numpy()  # (H, W), precision (unbounded positive)
                conf = conf / max(conf.max(), 1e-6)   # normalize to [0, 1] for display
                conf_uint8 = (conf * 255).astype(np.uint8)
                conf_color = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_VIRIDIS)
                conf_color = cv2.cvtColor(conf_color, cv2.COLOR_BGR2RGB)

                panels.extend([error_color, conf_color])

            rows.append(np.concatenate(panels, axis=1))

        grid = np.concatenate(rows, axis=0)
        save_path = vis_dir / f"{scene_id.replace('/', '_')}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


    def _save_pointcloud(self, eval_results, preds_flat, gt_flat, scene_id, T_in, T_out,
                         gt_depths=None, gt_w2c=None, gt_intrinsics_raw=None, gt_raw_images=None,
                         heldout_raw_images=None, heldout_w2c=None, heldout_intrinsics=None,
                         heldout_depths=None, heldout_frame_indices=None):
        """Save point clouds as .npz for visualize_pointclouds.py (Viser viewer).

        Format matches evaluate_seq.py: per-view coordinate maps + RGB images
        keyed as {prefix}_coord_{i} / {prefix}_img_{i}.

        When dataset GT depth + GT poses are available (gt_depths/gt_w2c/
        gt_intrinsics_raw/gt_raw_images all not None), the gt_* keys are
        derived from real ground-truth depth instead of reusing the predicted
        coord maps.
        """
        pc_dir = self.output_dir / "pointclouds"
        pc_dir.mkdir(parents=True, exist_ok=True)

        coord_maps = eval_results["coord_maps"]          # (B, T, 3, H, W)
        processed = eval_results["processed_images"]      # (B, T, 3, H, W) in [-1, 1]
        processed_01 = (processed + 1.0) / 2.0            # -> [0, 1]

        # World-frame TSDF data (extrinsics + intrinsics + per-scene norm bounds).
        # Saved alongside the existing normalized coords so visualize_pointclouds.py
        # can run Open3D TSDF fusion on input + predicted views.
        extr_all = eval_results.get("extrinsics")          # (B, T, 4, 4) or None
        intr_all = eval_results.get("intrinsics")          # (B, T, 3, 3) or None
        norm_min_all = eval_results.get("coord_norm_min")  # (B,) or None
        norm_max_all = eval_results.get("coord_norm_max")  # (B,) or None
        log_scale = bool(eval_results.get("log_scale_coords", False))
        have_tsdf_data = (
            extr_all is not None and intr_all is not None
            and norm_min_all is not None and norm_max_all is not None
        )

        use_real_gt_depth = (
            gt_depths is not None and gt_w2c is not None
            and gt_intrinsics_raw is not None and gt_raw_images is not None
        )

        # Model-predicted coordinate maps (from depth_vae decoder), if available.
        # These reflect the model's actual predictions, unlike full_coord_maps
        # which are DA3-derived dense unprojections.
        pred_coords = eval_results.get("coords")  # (B, T_out, 3, H, W) in [-1, 1] or None

        B = coord_maps.shape[0]
        for b in range(B):
            arrays = {}

            # Denormalize coord_maps back to world frame (same scale as extrinsics).
            # Inverts the forward normalization in geo_model.py:860-865.
            if have_tsdf_data:
                norm_min_b = float(norm_min_all[b].item())
                norm_max_b = float(norm_max_all[b].item())
                scene_size = max(norm_max_b - norm_min_b, 1e-6)

                def _denorm_coords(coord_norm_np):
                    if log_scale:
                        return (np.expm1(
                            (coord_norm_np + 1.0) * 0.5 * np.log1p(scene_size)
                        ) + norm_min_b).astype(np.float32)
                    else:
                        return ((coord_norm_np + 1.0) * 0.5 * scene_size + norm_min_b).astype(np.float32)

                # Input views: always DA3-derived
                input_coord_norm = coord_maps[b, :T_in].detach().cpu().float().numpy()
                input_coord_world = _denorm_coords(input_coord_norm)
                input_coord_mask = np.abs(input_coord_norm).max(axis=1) > 0.01

            # Input views
            for t in range(T_in):
                arrays[f"input_coord_{t}"] = coord_maps[b, t].cpu().float().numpy()
                arrays[f"input_img_{t}"] = processed_01[b, t].cpu().float().clamp(0, 1).numpy()
                if have_tsdf_data:
                    arrays[f"input_coord_world_{t}"] = input_coord_world[t]
                    arrays[f"input_coord_mask_{t}"] = input_coord_mask[t].astype(np.bool_)

            # Predicted views: use model-predicted coords when available,
            # fall back to DA3-derived full_coord_maps otherwise.
            for t in range(T_out):
                if pred_coords is not None:
                    pred_coord_t = pred_coords[b, t].cpu().float().numpy()
                else:
                    pred_coord_t = coord_maps[b, T_in + t].cpu().float().numpy()
                arrays[f"pred_coord_{t}"] = pred_coord_t
                arrays[f"pred_img_{t}"] = preds_flat[b * T_out + t].cpu().float().numpy()
                if have_tsdf_data:
                    pred_coord_world_t = _denorm_coords(pred_coord_t)
                    pred_coord_mask_t = np.abs(pred_coord_t).max(axis=0) > 0.01
                    arrays[f"pred_coord_world_{t}"] = pred_coord_world_t
                    arrays[f"pred_coord_mask_{t}"] = pred_coord_mask_t.astype(np.bool_)

            # GT views: prefer real GT depth from the dataset; fall back to
            # reusing the predicted coord maps when GT depth/poses are absent.
            if use_real_gt_depth:
                depth_b = gt_depths[b, T_in:T_in + T_out].to(self.device).float()        # (T_out, H, W)
                K_b     = gt_intrinsics_raw[b, T_in:T_in + T_out].to(self.device).float() # (T_out, 3, 3)
                w2c_b   = gt_w2c[b, T_in:T_in + T_out].to(self.device).float()            # (T_out, 4, 4)
                coords_world_gt = GeometryModel.unproject_depth_to_world_coords(
                    depth=depth_b, intrinsics=K_b, extrinsics=w2c_b, normalize=False,
                )  # (T_out, 3, H, W)
                mask_gt = (depth_b > 0) & torch.isfinite(depth_b)  # (T_out, H, W)
                rgb_gt = gt_raw_images[b, T_in:T_in + T_out].float() / 255.0  # (T_out, H, W, 3)
                rgb_gt = rgb_gt.permute(0, 3, 1, 2).contiguous()              # (T_out, 3, H, W)
                for t in range(T_out):
                    arrays[f"gt_coord_{t}"] = coord_maps[b, T_in + t].cpu().float().numpy()
                    arrays[f"gt_img_{t}"] = rgb_gt[t].cpu().numpy().astype(np.float32)
                    arrays[f"gt_coord_world_{t}"] = coords_world_gt[t].cpu().numpy().astype(np.float32)
                    arrays[f"gt_coord_mask_{t}"] = mask_gt[t].cpu().numpy().astype(np.bool_)
            else:
                for t in range(T_out):
                    gt_coord_norm_t = coord_maps[b, T_in + t].cpu().float().numpy()
                    arrays[f"gt_coord_{t}"] = gt_coord_norm_t
                    arrays[f"gt_img_{t}"] = gt_flat[b * T_out + t].cpu().float().numpy()
                    if have_tsdf_data:
                        arrays[f"gt_coord_world_{t}"] = _denorm_coords(gt_coord_norm_t)
                        arrays[f"gt_coord_mask_{t}"] = (np.abs(gt_coord_norm_t).max(axis=0) > 0.01).astype(np.bool_)

            if have_tsdf_data:
                arrays["extrinsics"] = extr_all[b].detach().cpu().float().numpy().astype(np.float32)
                arrays["intrinsics"] = intr_all[b].detach().cpu().float().numpy().astype(np.float32)
            arrays["has_tsdf_data"] = np.bool_(have_tsdf_data)

            # Per-image resolutions so consumers (e.g. fit_3dgs_ours.py) can
            # rescale intrinsics to match the saved image size. Inputs and GT
            # are at processed_images resolution; predictions may differ when
            # the model output resolution doesn't match DA3 process_res.
            proc_H, proc_W = processed_01.shape[-2], processed_01.shape[-1]
            pred_H, pred_W = preds_flat.shape[-2], preds_flat.shape[-1]
            arrays["input_img_resolution"] = np.array([proc_H, proc_W], dtype=np.int32)
            arrays["pred_img_resolution"] = np.array([pred_H, pred_W], dtype=np.int32)
            if use_real_gt_depth:
                gt_H_raw = int(rgb_gt.shape[-2])
                gt_W_raw = int(rgb_gt.shape[-1])
                arrays["gt_img_resolution"] = np.array([gt_H_raw, gt_W_raw], dtype=np.int32)
            else:
                arrays["gt_img_resolution"] = np.array([pred_H, pred_W], dtype=np.int32)

            # Held-out GT frames + poses (saved at the input/processed res
            # since they come straight from the dataset and must match a
            # known intrinsics scale). When the dataset returns held-outs at
            # a different resolution than processed_images, store the actual
            # resolution alongside.
            if (
                heldout_raw_images is not None
                and heldout_raw_images[b] is not None
                and heldout_w2c is not None
                and heldout_w2c[b] is not None
                and heldout_intrinsics is not None
                and heldout_intrinsics[b] is not None
            ):
                held_imgs = heldout_raw_images[b]            # (Nh, H, W, 3) uint8
                held_w2c_b = heldout_w2c[b].detach().cpu().float().numpy()       # (Nh, 4, 4)
                held_K_b = heldout_intrinsics[b].detach().cpu().float().numpy()  # (Nh, 3, 3)
                Nh = held_imgs.shape[0]
                held_H = int(held_imgs.shape[1])
                held_W = int(held_imgs.shape[2])
                held_imgs_np = held_imgs.detach().cpu().numpy().astype(np.float32) / 255.0
                # (Nh, H, W, 3) -> (Nh, 3, H, W)
                held_imgs_np = np.transpose(held_imgs_np, (0, 3, 1, 2))
                for i in range(Nh):
                    arrays[f"held_img_{i}"] = held_imgs_np[i].astype(np.float32)
                    arrays[f"held_w2c_{i}"] = held_w2c_b[i].astype(np.float32)
                    arrays[f"held_K_{i}"] = held_K_b[i].astype(np.float32)
                if heldout_depths is not None and heldout_depths[b] is not None:
                    held_d = heldout_depths[b].detach().cpu().float().numpy()
                    for i in range(Nh):
                        arrays[f"held_depth_{i}"] = held_d[i].astype(np.float32)
                if heldout_frame_indices is not None and heldout_frame_indices[b]:
                    arrays["heldout_frame_indices"] = np.array(
                        heldout_frame_indices[b], dtype=np.int64,
                    )
                arrays["num_heldout"] = np.int32(Nh)
                arrays["held_img_resolution"] = np.array([held_H, held_W], dtype=np.int32)
            else:
                arrays["num_heldout"] = np.int32(0)

            # Confidence maps for predicted views (if available)
            has_confidence = False
            if "confidence" in eval_results:
                conf = eval_results["confidence"]  # (B, T_out, 1, H, W)
                coord_H, coord_W = coord_maps.shape[-2:]
                if conf.shape[-2:] != (coord_H, coord_W):
                    conf_b = F.interpolate(
                        conf[b].float(), size=(coord_H, coord_W),
                        mode="bilinear", align_corners=False,
                    )  # (T_out, 1, H, W)
                else:
                    conf_b = conf[b].float()
                for t in range(T_out):
                    arrays[f"pred_conf_{t}"] = conf_b[t].cpu().numpy()  # (1, H, W)
                has_confidence = True

            arrays["num_input_views"] = T_in
            arrays["num_pred_views"] = T_out
            arrays["num_gt_views"] = T_out
            arrays["has_confidence"] = has_confidence

            # Original scene id with slashes preserved, so downstream tools
            # (e.g. fit_3dgs_ours.py video rendering) can locate the source
            # dataset file at <data_path>/<scene_id>.pt without having to
            # reverse the underscore-flattened npz filename.
            arrays["scene_id"] = np.array(str(scene_id))

            safe_name = scene_id.replace("/", "_")
            np.savez(pc_dir / f"{safe_name}.npz", **arrays)


def _build_parser():
    parser = argparse.ArgumentParser(description="Evaluate our model (same metrics as GEN3C eval)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="genrec",
                        help="Model checkpoint: a registry name ('genrec', auto-downloaded "
                             "from HuggingFace), an http(s) URL, or a local .pth path.")
    parser.add_argument("--output_dir", type=str, default="./ours_eval_results", help="Output directory")
    parser.add_argument(
        "--scene_list", type=str, default=None,
        help="Scene list file: one scene id per line (see assets/*_test_scenes_100.txt). "
             "Relative paths resolve against data_path, then the CWD. "
             "Per-dataset defaults are set in genrec/cli/eval_*.py.",
    )
    parser.add_argument("--step_size", type=int, default=10, help="Temporal step size (default: 10)")
    parser.add_argument("--num_steps", type=int, default=15, help="Number of denoising steps (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_visualizations", action="store_true", help="Save pred vs GT images")
    parser.add_argument("--save_pointclouds", action="store_true", help="Save .npz point clouds for visualize_pointclouds.py")
    parser.add_argument("--guide_checkpoint", type=str, default=None, help="Path to older/weaker checkpoint for autoguidance")
    parser.add_argument("--autoguidance_scale", type=float, default=1.5, help="Autoguidance weight w (default: 1.5)")
    parser.add_argument("--autoguidance_scale_unobs", type=float, default=None, help="Autoguidance weight for unobserved regions (default: None = uniform scale)")
    parser.add_argument("--confidence_checkpoint", type=str, default=None, help="Path to trained post-hoc confidence network checkpoint")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="CFG guidance scale (1.0 = no guidance, >1.0 = amplify 3D conditioning)")
    parser.add_argument("--guidance_rescale", type=float, default=0.0,
                        help="CFG Rescale (Lin et al.) to prevent over-exposure (0=off, 0.7-0.8 recommended)")
    parser.add_argument("--guidance_rescale_unobs", type=float, default=None,
                        help="Separate rescale strength for unobserved regions (default: None = same as --guidance_rescale)")
    parser.add_argument("--guidance_unobs", type=float, default=None,
                        help="If set, plain CFG uses this scale in unobserved regions; --guidance remains the observed-region scale. Requires --guidance > 1.0.")
    parser.add_argument("--conf_mask_thresh", type=float, default=0.0,
                        help="DA3 confidence threshold for warp mask (0=disabled, 0.5=match GEN3C)")
    parser.add_argument("--obs_masks_dir", type=str, default=None,
                        help="Observation masks for the observed/unobserved metric split: 'paper' "
                             "(auto-download the paper's masks), 'none' (use internally computed "
                             "warp masks), a .npz archive, or a directory of per-scene .npy files")
    parser.add_argument("--use_gt_poses", action="store_true",
                        help="Use GT camera poses from dataset instead of DA3 predictions (DA3 depth is Umeyama-aligned)")
    parser.add_argument("--skip_coord_metrics", action="store_true",
                        help="Skip coordinate map evaluation metrics")
    parser.add_argument("--save_video", action="store_true",
                        help="Save generated views as an MP4 video per scene under <output_dir>/videos/")
    parser.add_argument("--video_fps", type=int, default=10,
                        help="Frame rate for --save_video output (default: 10)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save individual pred and GT images as PNGs for offline metric scripts")
    parser.add_argument("--depth_filter_thresh", type=float, default=0.5,
                        help="Depth discontinuity filter threshold for forward-warp rendering (default: 0.5, GEN3C-like: 0.05)")
    parser.add_argument("--depth_grad_thresh", type=float, default=0.0,
                        help="Relative depth-gradient edge filter (0 = off; tuned via visualize_pointclouds.py; try 0.01)")
    parser.add_argument("--obs_blend", action="store_true",
                        help="Enable observed-region latent blending (inpainting-style)")
    parser.add_argument("--obs_blend_strength", type=float, default=0.8,
                        help="Blend strength for observed regions (0=off, 1=hard inpaint, default: 0.8)")
    parser.add_argument("--dynamic_threshold", type=float, default=0.0,
                        help="Dynamic thresholding percentile (Imagen). 0=off, 0.995=recommended.")
    parser.add_argument("--recon_step_ratio", type=float, default=0.0,
                        help="Apply recon branch corrections only for the last N%% of denoising steps (0.0=all steps, 0.3=last 30%%)")
    parser.add_argument("--skip_last_step_correction", action="store_true",
                        help="Skip the recon branch pixel correction on the very last denoising step")
    parser.add_argument("--sample_from_end", action="store_true",
                        help="Sample the last (T_in+T_out)*step_size frames instead of the first")
    parser.add_argument("--source_only_da3", action="store_true",
        help="Only pass source views through DA3 (prevents cross-view depth contamination)")
    parser.add_argument("--hires_warp_res", type=int, default=0,
                        help="Recompute obs masks via forward warp at this resolution (longest side), "
                             "then NN-downsample. 0=disabled, 1280=match GEN3C resolution")
    parser.add_argument("--dataset_type", type=str, default="re10k", choices=["re10k", "dl3dv", "mipnerf360"],
                        help="Dataset type to evaluate on (default: re10k)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override data path (required for dl3dv/mipnerf360 or when cfg.data.path is absent)")
    parser.add_argument("--longest_side", type=int, default=None,
                        help="Longest-side resize for aspect-ratio preservation (e.g., 616 for DL3DV)")
    parser.add_argument("--interpolation", action="store_true",
                        help="Use interpolation layout (inputs at trajectory endpoints, targets in between)")
    parser.add_argument("--factor", type=int, default=None,
                        help="Image downsampling factor for mipnerf360 (1/2/4/8; selects images_<factor>/). Dataset default: 4.")
    parser.add_argument("--no_vae_skip", action="store_true",
                        help="Ablation: disable image+depth VAE skip connections at inference")
    parser.add_argument("--num_heldout", type=int, default=0,
                        help="Number of GT held-out frames to save per scene "
                             "(midpoints between consecutive target frames). "
                             "Used by src/fit_3dgs_ours.py for downstream 3DGS "
                             "generalization eval. Requires --use_gt_poses for "
                             "consistent camera coordinates.")
    parser.add_argument("--two_pass_interp", action="store_true",
                        help="Two-pass long-horizon: pass 1 generates 8 sparse anchors "
                             "(T_in=2 + T_out=6); pass 2 runs 7 calls (T_in=4 / T_out=4) "
                             "to interpolate 4 frames between each adjacent anchor pair "
                             "using LERP+SLERP cameras and the source-view point cloud. "
                             "Final stitched output is 36 frames. Pass 1 path is unchanged.")
    parser.add_argument("overrides", nargs="*", help="Config overrides (key=value)")
    return parser


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.guidance_unobs is not None and args.guidance <= 1.0:
        parser.error("--guidance_unobs requires --guidance > 1.0 (plain CFG must be active).")
    return args


def run_eval(dataset_type, defaults=None, argv=None):
    """Entry point for the thin per-dataset eval scripts (eval_re10k / eval_dl3dv / ...).

    Fixes ``--dataset_type`` and applies that dataset's argparse defaults (still
    overridable on the command line), then runs the shared evaluation engine.
    """
    parser = _build_parser()
    if defaults:
        # Any arg we supply a dataset default for is no longer required on the CLI.
        for action in parser._actions:
            if action.dest in defaults:
                action.required = False
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.dataset_type = dataset_type  # fixed for this entry point
    if args.guidance_unobs is not None and args.guidance <= 1.0:
        parser.error("--guidance_unobs requires --guidance > 1.0 (plain CFG must be active).")
    _run(args)


def main():
    args = parse_args()
    _run(args)


def _run(args):
    rank, world_size, device = setup_distributed()

    cfg, model_opts, _ = load_experiment_config(args.config, args.overrides)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        evaluator = OursEvaluator(
            cfg=cfg,
            model_opts=model_opts,
            device=device,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            scene_list=args.scene_list,
            step_size=args.step_size,
            num_inference_steps=args.num_steps,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            save_visualizations=args.save_visualizations,
            save_pointclouds=args.save_pointclouds,
            guide_checkpoint=args.guide_checkpoint,
            autoguidance_scale=args.autoguidance_scale,
            autoguidance_scale_unobs=args.autoguidance_scale_unobs,
            confidence_checkpoint=args.confidence_checkpoint,
            guidance_scale=args.guidance,
            guidance_rescale=args.guidance_rescale,
            guidance_rescale_unobs=args.guidance_rescale_unobs,
            guidance_unobs=args.guidance_unobs,
            conf_mask_thresh=args.conf_mask_thresh,
            obs_masks_dir=args.obs_masks_dir,
            use_gt_poses=args.use_gt_poses,
            skip_coord_metrics=args.skip_coord_metrics,
            save_images=args.save_images,
            save_video=args.save_video,
            video_fps=args.video_fps,
            depth_filter_thresh=args.depth_filter_thresh,
            depth_grad_thresh=args.depth_grad_thresh,
            obs_blend=args.obs_blend,
            obs_blend_strength=args.obs_blend_strength,
            dynamic_threshold_percentile=args.dynamic_threshold,
            recon_step_ratio=args.recon_step_ratio,
            skip_last_step_correction=args.skip_last_step_correction,
            sample_from_end=args.sample_from_end,
            hires_warp_res=args.hires_warp_res,
            source_only_da3=args.source_only_da3,
            dataset_type=args.dataset_type,
            data_path=args.data_path,
            longest_side=args.longest_side,
            interpolation=args.interpolation,
            factor=args.factor,
            no_vae_skip=args.no_vae_skip,
            num_heldout=args.num_heldout,
            two_pass_interp=args.two_pass_interp,
        )
        evaluator.run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Example invocations
# -----------------------------------------------------------------------------
#
# RE10K:
#   python -m genrec.cli.evaluate_ours \
#       --dataset_type re10k \
#       --config configs/re10k.yaml \
#       --checkpoint genrec \
#       --data_path /path/to/RealEstate10K \
#       --output_dir ./ours_eval_results \
#       --scene_list assets/re10k_test_scenes_100.txt \
#       --guidance 1.5 --use_gt_poses --save_visualizations
#
# DL3DV-10K:
#   python -m genrec.cli.evaluate_ours \
#       --dataset_type dl3dv \
#       --config configs/dl3dv.yaml \
#       --checkpoint genrec \
#       --data_path /path/to/DL3DV-10K \
#       --output_dir ./dl3dv_eval_results \
#       --scene_list assets/dl3dv_test_scenes_100.txt \
#       --longest_side 616 --step_size 4 --num_steps 15 \
#       --guidance 1.5 --use_gt_poses --save_visualizations
