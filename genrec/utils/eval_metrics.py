"""
Evaluation metrics for observed/unobserved region analysis.

Standalone masked metric functions, distributional metrics (FID, FD-DINOv2,
CMMD, CLIP-IQA, TSED), and histogram plotting for evaluation scripts.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===========================================================
# Masked metric functions — all return (N,) per-image values
# ===========================================================

def compute_masked_l1(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Masked L1 error.

    Args:
        pred, gt: (N, C, H, W) in [0, 1].
        mask: (N, 1, H, W) soft mask in [0, 1].

    Returns:
        (N,) per-image masked L1.
    """
    C = pred.shape[1]
    diff = (pred - gt).abs() * mask  # (N, C, H, W)
    per_image = diff.sum(dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) * C).clamp(min=1e-8)
    return per_image


def compute_masked_mse(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked MSE. Returns (N,) per-image values."""
    C = pred.shape[1]
    diff = ((pred - gt) ** 2) * mask
    per_image = diff.sum(dim=[1, 2, 3]) / (mask.sum(dim=[1, 2, 3]) * C).clamp(min=1e-8)
    return per_image


def compute_masked_psnr(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked PSNR from masked MSE. Returns (N,) per-image values."""
    mse = compute_masked_mse(pred, gt, mask)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-8))
    return psnr


def compute_masked_ssim(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Masked SSIM via skimage full SSIM map weighted by mask.

    Args:
        pred, gt: (N, C, H, W) in [0, 1].
        mask: (N, 1, H, W) soft mask in [0, 1].

    Returns:
        (N,) per-image masked SSIM.
    """
    from skimage.metrics import structural_similarity as calculate_ssim

    N = pred.shape[0]
    results = []
    for i in range(N):
        pred_np = (pred[i].cpu().float().numpy() * 255.0).astype(np.uint8)
        gt_np = (gt[i].cpu().float().numpy() * 255.0).astype(np.uint8)
        mask_np = mask[i, 0].cpu().float().numpy()  # (H, W)

        # structural_similarity with full=True returns (score, ssim_map)
        _, ssim_map = calculate_ssim(
            pred_np, gt_np,
            channel_axis=0,
            full=True,
            data_range=255,
        )
        # ssim_map shape: (C, H, W), average over channels to get (H, W)
        ssim_map_mean = ssim_map.mean(axis=0)  # (H, W)

        # Weighted average over mask
        weighted = (ssim_map_mean * mask_np).sum()
        count = mask_np.sum()
        results.append(weighted / max(count, 1e-8))

    return torch.tensor(results, dtype=pred.dtype, device=pred.device)


def compute_confidence_weighted_lpips(
    pred: torch.Tensor, gt: torch.Tensor, confidence: torch.Tensor,
    spatial_lpips_net,
) -> torch.Tensor:
    """
    Confidence-weighted LPIPS using spatial distance maps.

    Args:
        pred, gt: (N, C, H, W) in [0, 1].
        confidence: (N, 1, H, W) soft confidence in [0, 1].
        spatial_lpips_net: lpips.LPIPS instance with spatial=True.

    Returns:
        (N,) per-image confidence-weighted LPIPS values.
    """
    # LPIPS expects [-1, 1]
    spatial_map = spatial_lpips_net(pred * 2 - 1, gt * 2 - 1)  # (N, 1, H, W)

    # Resize confidence to match spatial map if needed
    if confidence.shape[-2:] != spatial_map.shape[-2:]:
        confidence = F.interpolate(
            confidence, size=spatial_map.shape[-2:],
            mode='bilinear', align_corners=False,
        )

    # Weighted average: sum(conf * lpips_map) / sum(conf)
    weighted = (confidence * spatial_map).sum(dim=[1, 2, 3])
    denom = confidence.sum(dim=[1, 2, 3]).clamp(min=1e-8)
    return weighted / denom


def compute_masked_lpips(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    lpips_net, lpips_resize: int = 256,
) -> torch.Tensor:
    """
    Masked LPIPS: gray-fill (0.5) non-ROI regions, then compute LPIPS at lpips_resize.

    Args:
        pred, gt: (N, C, H, W) in [0, 1].
        mask: (N, 1, H, W) soft mask in [0, 1].
        lpips_net: kiui.lpips.LPIPS instance.
        lpips_resize: resize resolution for LPIPS computation.

    Returns:
        (N,) per-image LPIPS values.
    """
    # Gray-fill unobserved regions
    pred_masked = pred * mask + 0.5 * (1.0 - mask)
    gt_masked = gt * mask + 0.5 * (1.0 - mask)

    # Resize
    if lpips_resize > 0:
        pred_masked = F.interpolate(
            pred_masked, size=(lpips_resize, lpips_resize),
            mode='bilinear', align_corners=False
        )
        gt_masked = F.interpolate(
            gt_masked, size=(lpips_resize, lpips_resize),
            mode='bilinear', align_corners=False
        )

    # LPIPS expects [-1, 1]
    lpips_vals = lpips_net(pred_masked * 2 - 1, gt_masked * 2 - 1)  # (N, 1, 1, 1) or (N,)
    if lpips_vals.ndim > 1:
        lpips_vals = lpips_vals.squeeze()
    if lpips_vals.ndim == 0:
        lpips_vals = lpips_vals.unsqueeze(0)
    return lpips_vals


# ===========================================================
# Confidence calibration metrics
# ===========================================================

def compute_spearman_rank_correlation(
    confidence: torch.Tensor, error: torch.Tensor, subsample: int = 0
) -> torch.Tensor:
    """
    Per-image Spearman rank correlation between confidence and error.

    Ideal value: -1.0 (high confidence = low error).

    Args:
        confidence: (N, H, W) predicted confidence in [0, 1].
        error: (N, H, W) per-pixel error (e.g. mean absolute error across channels).
        subsample: if > 0, randomly subsample this many pixels per image for efficiency.

    Returns:
        (N,) Spearman correlation per image.
    """
    N, H, W = confidence.shape
    conf_flat = confidence.reshape(N, -1)   # (N, H*W)
    err_flat = error.reshape(N, -1)         # (N, H*W)

    results = torch.zeros(N, device=confidence.device, dtype=confidence.dtype)

    for i in range(N):
        c = conf_flat[i]
        e = err_flat[i]

        if subsample > 0 and subsample < c.shape[0]:
            idx = torch.randperm(c.shape[0], device=c.device)[:subsample]
            c = c[idx]
            e = e[idx]

        # Double-argsort to get ranks
        c_ranks = c.argsort().argsort().float()
        e_ranks = e.argsort().argsort().float()

        # Correlation of ranks
        stacked = torch.stack([c_ranks, e_ranks], dim=0)
        corr_matrix = torch.corrcoef(stacked)
        results[i] = corr_matrix[0, 1]

    return results


def compute_ause(
    confidence: torch.Tensor, error: torch.Tensor, num_bins: int = 20
) -> torch.Tensor:
    """
    Area Under Sparsification Error (AUSE) — uncertainty calibration metric.

    Progressively removes least-confident pixels (model curve) vs highest-error
    pixels (oracle curve) and compares remaining mean error. Lower = better.

    Args:
        confidence: (N, H, W) predicted confidence in [0, 1].
        error: (N, H, W) per-pixel error.
        num_bins: number of fraction steps for sparsification.

    Returns:
        (N,) AUSE values (non-negative, lower = better calibrated).
    """
    N, H, W = confidence.shape
    num_pixels = H * W
    conf_flat = confidence.reshape(N, -1)
    err_flat = error.reshape(N, -1)

    fractions = torch.linspace(0.0, 1.0, num_bins + 1, device=confidence.device)
    results = torch.zeros(N, device=confidence.device, dtype=confidence.dtype)

    for i in range(N):
        c = conf_flat[i]
        e = err_flat[i]

        # Model curve: sort by confidence ascending (remove least confident first)
        model_order = c.argsort()           # least confident first
        e_model_sorted = e[model_order]

        # Oracle curve: sort by error descending (remove highest error first)
        oracle_order = e.argsort(descending=True)  # highest error first
        e_oracle_sorted = e[oracle_order]

        model_curve = torch.zeros(num_bins + 1, device=confidence.device, dtype=confidence.dtype)
        oracle_curve = torch.zeros(num_bins + 1, device=confidence.device, dtype=confidence.dtype)

        for k, frac in enumerate(fractions):
            n_remove = int(frac.item() * num_pixels)
            n_remain = num_pixels - n_remove
            if n_remain <= 0:
                model_curve[k] = 0.0
                oracle_curve[k] = 0.0
            else:
                model_curve[k] = e_model_sorted[n_remove:].mean()
                oracle_curve[k] = e_oracle_sorted[n_remove:].mean()

        # AUSE = area between model and oracle curves (trapezoidal rule)
        diff = model_curve - oracle_curve
        # Normalize by oracle area to get relative AUSE
        oracle_area = torch.trapezoid(oracle_curve, fractions)
        model_area = torch.trapezoid(diff, fractions)
        if oracle_area.abs() > 1e-8:
            results[i] = model_area / oracle_area
        else:
            results[i] = model_area

    return results


# ===========================================================
# Histogram plotting
# ===========================================================

def _compute_bin_means(values: np.ndarray, bin_indices: np.ndarray, num_bins: int) -> np.ndarray:
    """Compute per-bin means for a metric array."""
    bin_means = []
    for b in range(num_bins):
        in_bin = values[bin_indices == b]
        bin_means.append(in_bin.mean() if len(in_bin) > 0 else 0.0)
    return np.array(bin_means)


def plot_metric_histograms(
    obs_percentages: np.ndarray,
    metrics_dict: Dict[str, np.ndarray],
    output_dir: str,
    num_bins: int = 10,
    prefix: str = "",
    baseline_metrics_dict: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Create bar charts of mean metric values binned by observation percentage.

    Args:
        obs_percentages: (N,) observation percentage per sample in [0, 1].
        metrics_dict: {metric_name: (N,) values} — all metrics to plot.
        output_dir: directory to save histogram PNGs.
        num_bins: number of equal-width bins across [0, 100]%.
        prefix: optional filename prefix (e.g., "step_10_denoise_025_").
        baseline_metrics_dict: optional {metric_name: (N,) values} for NN-fill baseline.
            When provided, bars are grouped side-by-side (model vs baseline).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hist_dir = Path(output_dir) / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 * 100  # percentage
    full_bin_width = 100.0 / num_bins * 0.8
    has_baseline = baseline_metrics_dict is not None

    # Assign each sample to a bin
    bin_indices = np.digitize(obs_percentages, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Count samples per bin
    counts = np.array([np.sum(bin_indices == b) for b in range(num_bins)])

    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)

    # Widths and offsets for grouped bars
    if has_baseline:
        half_w = full_bin_width / 2
        offset_model = -half_w / 2
        offset_baseline = half_w / 2
    else:
        half_w = full_bin_width
        offset_model = 0.0

    # Individual plots per metric
    for name, values in metrics_dict.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        bin_means = _compute_bin_means(values, bin_indices, num_bins)
        bars = ax.bar(bin_centers + offset_model, bin_means, width=half_w,
                      color='steelblue', edgecolor='black', alpha=0.8, label='Model')

        # Annotate with sample count
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'n={count}', ha='center', va='bottom', fontsize=8)

        if has_baseline and name in baseline_metrics_dict:
            bl_means = _compute_bin_means(baseline_metrics_dict[name], bin_indices, num_bins)
            ax.bar(bin_centers + offset_baseline, bl_means, width=half_w,
                   color='darkorange', edgecolor='black', alpha=0.8, label='NN-Fill Baseline')
            ax.legend()

        ax.set_xlabel('Observation Percentage (%)')
        ax.set_ylabel(name)
        ax.set_title(f'{name} by Observation Percentage')
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f'{c:.0f}%' for c in bin_centers], rotation=45)
        plt.tight_layout()
        fig.savefig(hist_dir / f'{prefix}{name.lower().replace("/", "_").replace(" ", "_")}.png', dpi=150)
        plt.close(fig)

    # Combined figure with all metrics
    if n_metrics > 0:
        cols = min(n_metrics, 4)
        rows = math.ceil(n_metrics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, (name, values) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            bin_means = _compute_bin_means(values, bin_indices, num_bins)
            bars = ax.bar(bin_centers + offset_model, bin_means, width=half_w,
                          color='steelblue', edgecolor='black', alpha=0.8, label='Model')
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'n={count}', ha='center', va='bottom', fontsize=6)

            if has_baseline and name in baseline_metrics_dict:
                bl_means = _compute_bin_means(baseline_metrics_dict[name], bin_indices, num_bins)
                ax.bar(bin_centers + offset_baseline, bl_means, width=half_w,
                       color='darkorange', edgecolor='black', alpha=0.8, label='NN-Fill Baseline')
                ax.legend(fontsize=6)

            ax.set_xlabel('Obs %')
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.set_xticks(bin_centers)
            ax.set_xticklabels([f'{c:.0f}' for c in bin_centers], rotation=45, fontsize=7)

        # Hide unused axes
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{prefix.rstrip("_")} Metrics by Observation %' if prefix else 'Metrics by Observation %')
        plt.tight_layout()
        fig.savefig(hist_dir / f'{prefix}combined_histograms.png', dpi=150)
        plt.close(fig)


# ===========================================================
# FID Accumulator
# ===========================================================

class FIDAccumulator:
    """
    Incremental FID (Frechet Inception Distance) accumulator.

    Wraps torchmetrics.image.fid.FrechetInceptionDistance to accumulate
    Inception features over the dataset, then compute FID once at the end.

    Memory footprint: ~33MB (running mean + covariance of 2048-dim features),
    independent of dataset size.
    """

    def __init__(self, feature: int = 2048, device: str = "cuda:0"):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self._fid = FrechetInceptionDistance(feature=feature).to(device)
        self._device = device
        self._num_real = 0
        self._num_fake = 0

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to 299x299 and convert float [0,1] -> uint8 [0,255]."""
        x = images.to(self._device)
        if x.shape[-2] != 299 or x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = (x.clamp(0.0, 1.0) * 255).to(torch.uint8)
        return x

    def update_real(self, images: torch.Tensor) -> None:
        """Feed real images (N, C, H, W) float [0,1]."""
        self._fid.update(self._preprocess(images), real=True)
        self._num_real += images.shape[0]

    def update_fake(self, images: torch.Tensor) -> None:
        """Feed generated images (N, C, H, W) float [0,1]."""
        self._fid.update(self._preprocess(images), real=False)
        self._num_fake += images.shape[0]

    def compute(self) -> Optional[float]:
        """Compute FID. Returns None if insufficient samples (after distributed sync)."""
        try:
            return float(self._fid.compute().item())
        except RuntimeError:
            return None

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._fid.reset()
        self._num_real = 0
        self._num_fake = 0


# ===========================================================
# FD-DINOv2 Accumulator
# ===========================================================

class _DINOv2FeatureExtractor(nn.Module):
    """Wraps DINOv2 ViT-B/14 as a feature extractor for FrechetInceptionDistance."""

    num_features = 768  # CLS token dimension; torchmetrics reads this

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Always stay in eval mode
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) float [0, 1]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        features = self.model.forward_features(x)
        return features['x_norm_clstoken']  # (N, 768)


class FDDINOv2Accumulator:
    """
    Incremental FD-DINOv2 (Frechet Distance on DINOv2 features) accumulator.

    Same interface as FIDAccumulator but uses DINOv2 ViT-B/14 CLS token
    features (768-dim) instead of InceptionV3 features (2048-dim).
    """

    def __init__(self, device: str = "cuda:0"):
        from torchmetrics.image.fid import FrechetInceptionDistance
        extractor = _DINOv2FeatureExtractor()
        self._fd = FrechetInceptionDistance(feature=extractor, normalize=True).to(device)
        self._device = device
        self._num_real = 0
        self._num_fake = 0

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Clamp to [0,1] float32 and move to device."""
        return images.to(self._device).float().clamp(0.0, 1.0)

    def update_real(self, images: torch.Tensor) -> None:
        """Feed real images (N, C, H, W) float [0,1]."""
        self._fd.update(self._preprocess(images), real=True)
        self._num_real += images.shape[0]

    def update_fake(self, images: torch.Tensor) -> None:
        """Feed generated images (N, C, H, W) float [0,1]."""
        self._fd.update(self._preprocess(images), real=False)
        self._num_fake += images.shape[0]

    def compute(self) -> Optional[float]:
        """Compute FD-DINOv2. Returns None if insufficient samples (after distributed sync)."""
        try:
            return float(self._fd.compute().item())
        except RuntimeError:
            return None

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self._fd.reset()
        self._num_real = 0
        self._num_fake = 0


# ===========================================================
# CMMD (CLIP Maximum Mean Discrepancy) Accumulator
# ===========================================================

class CMMDAccumulator:
    """
    CLIP Maximum Mean Discrepancy accumulator.

    Measures distributional similarity between real and generated images using
    CLIP embeddings with Gaussian RBF kernel MMD. Lower is better.
    Reference: https://github.com/sayakpaul/cmmd-pytorch
    """

    def __init__(self, device: str = "cuda:0"):
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        model_id = "openai/clip-vit-large-patch14-336"
        self._model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
        self._processor = CLIPImageProcessor.from_pretrained(model_id)
        self._device = device
        self._real_embeds: list = []
        self._fake_embeds: list = []

    @torch.no_grad()
    def _extract(self, images: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized CLIP image embeddings (N, 768)."""
        images = images.float().clamp(0.0, 1.0)
        # Resize to 336x336 and normalize with CLIP constants
        images = F.interpolate(images, size=(336, 336), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        mean = torch.tensor(self._processor.image_mean, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(self._processor.image_std, device=self._device).view(1, 3, 1, 1)
        images = (images - mean) / std
        embeds = self._model(pixel_values=images).image_embeds  # (N, 768)
        embeds = F.normalize(embeds, dim=-1)
        return embeds.cpu()

    def update_real(self, images: torch.Tensor) -> None:
        self._real_embeds.append(self._extract(images.to(self._device)))

    def update_fake(self, images: torch.Tensor) -> None:
        self._fake_embeds.append(self._extract(images.to(self._device)))

    @staticmethod
    def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0, scale: float = 1000.0) -> float:
        gamma = 1.0 / (2.0 * sigma * sigma)
        x_sqnorms = (x * x).sum(dim=1)
        y_sqnorms = (y * y).sum(dim=1)
        dist_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2.0 * x @ x.T
        dist_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2.0 * y @ y.T
        dist_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2.0 * x @ y.T
        k_xx = torch.exp(-gamma * dist_xx).mean()
        k_yy = torch.exp(-gamma * dist_yy).mean()
        k_xy = torch.exp(-gamma * dist_xy).mean()
        return float(scale * (k_xx + k_yy - 2.0 * k_xy))

    def sync_distributed(self):
        """Gather embeddings from all ranks. Must be called by ALL ranks."""
        import torch.distributed as dist
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        device = torch.device('cuda')

        for attr in ('_real_embeds', '_fake_embeds'):
            embeds_list = getattr(self, attr)
            local = torch.cat(embeds_list, dim=0) if embeds_list else torch.empty(0, 768)

            # Exchange sizes (ranks may have different counts)
            local_size = torch.tensor([local.shape[0]], dtype=torch.long, device=device)
            all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            max_size = max(s.item() for s in all_sizes)

            if max_size == 0:
                setattr(self, attr, [])
                continue

            # Pad, gather, trim
            padded = torch.zeros(int(max_size), 768, device=device)
            padded[:local.shape[0]] = local.to(device)

            gathered = [torch.zeros_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded)

            all_embeds = []
            for i in range(world_size):
                n = int(all_sizes[i].item())
                if n > 0:
                    all_embeds.append(gathered[i][:n].cpu())

            setattr(self, attr, all_embeds)

    def compute(self) -> Optional[float]:
        if not self._real_embeds or not self._fake_embeds:
            return None
        real = torch.cat(self._real_embeds, dim=0)
        fake = torch.cat(self._fake_embeds, dim=0)
        if real.shape[0] < 2 or fake.shape[0] < 2:
            return None
        return self._mmd_rbf(real, fake)

    def reset(self) -> None:
        self._real_embeds.clear()
        self._fake_embeds.clear()


# ===========================================================
# CLIP-IQA Accumulator (no-reference image quality)
# ===========================================================

class CLIPIQAAccumulator:
    """
    Incremental CLIP-IQA accumulator for no-reference image quality assessment.

    Computes per-image scores for each prompt and returns dataset-level means.
    Unlike FID/FD-DINOv2 (distributional metrics), CLIP-IQA is per-image,
    so we accumulate scores and average at the end.
    """

    PROMPTS = ("quality", "sharpness", "natural")

    def __init__(self, device: str = "cuda:0"):
        from torchmetrics.multimodal import CLIPImageQualityAssessment
        self._metric = CLIPImageQualityAssessment(prompts=self.PROMPTS).to(device)
        self._device = device
        self._sums = {p: 0.0 for p in self.PROMPTS}
        self._count = 0

    def update(self, images: torch.Tensor) -> Dict[str, float]:
        """Feed generated images (N, C, H, W) float [0,1]. Returns per-image mean scores."""
        x = images.to(self._device).float().clamp(0.0, 1.0)
        scores = self._metric(x)  # dict {prompt: tensor}
        batch_means = {}
        for p in self.PROMPTS:
            # torchmetrics key format varies by version: "clip_iqa_{p}" or just "{p}"
            key = f"clip_iqa_{p}" if f"clip_iqa_{p}" in scores else p
            val = scores[key].mean().item()
            self._sums[p] += scores[key].sum().item()
            batch_means[p] = val
        self._count += x.shape[0]
        return batch_means

    def compute(self) -> Optional[Dict[str, float]]:
        """Return mean scores per prompt. None if no samples."""
        if self._count == 0:
            return None
        return {p: self._sums[p] / self._count for p in self.PROMPTS}

    def reset(self):
        self._sums = {p: 0.0 for p in self.PROMPTS}
        self._count = 0


# ===========================================================
# TSED (Thresholded Symmetric Epipolar Distance)
# From "3D-aware Multi-Class Image-to-Image Translation with NeRFs"
# (Frieder et al., CVPR 2023, arXiv:2304.10700)
# ===========================================================

def compute_fundamental_matrix(
    K1: np.ndarray, K2: np.ndarray,
    w2c1: np.ndarray, w2c2: np.ndarray,
) -> np.ndarray:
    """Compute fundamental matrix F from known intrinsics and w2c poses.

    F = K2^{-T} [t]_x R K1^{-1}  where R, t are the relative pose from cam1 to cam2.

    Args:
        K1, K2: (3, 3) intrinsic matrices.
        w2c1, w2c2: (4, 4) world-to-camera extrinsics.

    Returns:
        F: (3, 3) fundamental matrix.
    """
    # Relative pose: cam2 = R_rel @ cam1_world + t_rel
    # w2c = [R|t], so c2w = inv(w2c)
    c2w1 = np.linalg.inv(w2c1)
    # Relative: w2c2 @ c2w1 = [R_rel | t_rel; 0 0 0 1]
    rel = w2c2 @ c2w1
    R_rel = rel[:3, :3]
    t_rel = rel[:3, 3]

    # Skew-symmetric matrix of t
    tx = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0],
    ], dtype=np.float64)

    # Essential matrix E = [t]_x R
    E = tx @ R_rel

    # F = K2^{-T} E K1^{-1}
    K2_inv_T = np.linalg.inv(K2).T
    K1_inv = np.linalg.inv(K1)
    F = K2_inv_T @ E @ K1_inv
    return F


def compute_symmetric_epipolar_distance(
    F: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Compute Symmetric Epipolar Distance for point correspondences.

    SED = 0.5 * (d(p2, F p1)^2 / ||F p1||^2_{1:2} + d(p1, F^T p2)^2 / ||F^T p2||^2_{1:2})
    where d is the signed distance from a point to the epipolar line.

    Args:
        F: (3, 3) fundamental matrix.
        pts1, pts2: (N, 2) matched keypoint coordinates.

    Returns:
        sed: (N,) symmetric epipolar distances in pixels.
    """
    N = pts1.shape[0]
    # Homogeneous coordinates
    ones = np.ones((N, 1), dtype=np.float64)
    p1h = np.hstack([pts1, ones])  # (N, 3)
    p2h = np.hstack([pts2, ones])  # (N, 3)

    # Epipolar lines: l2 = F @ p1, l1 = F^T @ p2
    l2 = (F @ p1h.T).T  # (N, 3)
    l1 = (F.T @ p2h.T).T  # (N, 3)

    # Signed distance: p^T l / ||l_{1:2}||
    # d(p2, l2) = |p2^T l2| / sqrt(l2[0]^2 + l2[1]^2)
    d2 = np.abs(np.sum(p2h * l2, axis=1)) / (np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8)
    d1 = np.abs(np.sum(p1h * l1, axis=1)) / (np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8)

    sed = 0.5 * (d1 + d2)
    return sed


def compute_tsed_for_scene(
    images: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    match_threshold: int = 10,
    sed_threshold: float = 2.0,
    lowe_ratio: float = 0.75,
    sift_edge_threshold: int = 30,
) -> Dict[str, float]:
    """Compute TSED for a sequence of predicted images with known poses.

    For each consecutive pair: extract SIFT keypoints, match with BFMatcher +
    Lowe ratio test, compute F from known poses, compute median SED, threshold.

    Args:
        images: (T, H, W, 3) uint8 predicted images.
        intrinsics: (T, 3, 3) camera intrinsics (at image resolution).
        extrinsics: (T, 4, 4) world-to-camera extrinsics.
        match_threshold: minimum number of matches for a pair to count.
        sed_threshold: maximum median SED (px) for a pair to be "consistent".
        lowe_ratio: Lowe ratio test threshold for SIFT matching.
        sift_edge_threshold: SIFT edge threshold (COLMAP default = 30).

    Returns:
        Dict with 'tsed', 'median_sed', 'n_matches', 'n_consistent', 'n_pairs'.
    """
    import cv2

    T = images.shape[0]
    if T < 2:
        return {"tsed": 0.0, "median_sed": 0.0, "n_matches": 0.0, "n_consistent": 0, "n_pairs": 0}

    sift = cv2.SIFT_create(edgeThreshold=sift_edge_threshold)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    n_pairs = T - 1
    n_consistent = 0
    all_median_seds = []
    all_n_matches = []

    for i in range(n_pairs):
        img1 = images[i]
        img2 = images[i + 1]

        # Convert to grayscale for SIFT
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            all_median_seds.append(float("inf"))
            all_n_matches.append(0)
            continue

        # BFMatcher with knn=2 for Lowe ratio test
        raw_matches = bf.knnMatch(des1, des2, k=2)

        # Lowe ratio test
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < lowe_ratio * n.distance:
                    good_matches.append(m)

        n_matches = len(good_matches)
        all_n_matches.append(n_matches)

        if n_matches < match_threshold:
            all_median_seds.append(float("inf"))
            continue

        # Extract matched points
        pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float64)

        # Compute F from known poses
        K1 = intrinsics[i].astype(np.float64)
        K2 = intrinsics[i + 1].astype(np.float64)
        w2c1 = extrinsics[i].astype(np.float64)
        w2c2 = extrinsics[i + 1].astype(np.float64)

        F = compute_fundamental_matrix(K1, K2, w2c1, w2c2)
        sed = compute_symmetric_epipolar_distance(F, pts1, pts2)
        median_sed = float(np.median(sed))
        all_median_seds.append(median_sed)

        if median_sed < sed_threshold:
            n_consistent += 1

    tsed = n_consistent / n_pairs if n_pairs > 0 else 0.0
    mean_median_sed = float(np.mean([s for s in all_median_seds if np.isfinite(s)])) if any(np.isfinite(s) for s in all_median_seds) else 0.0
    mean_n_matches = float(np.mean(all_n_matches)) if all_n_matches else 0.0

    return {
        "tsed": tsed,
        "median_sed": mean_median_sed,
        "n_matches": mean_n_matches,
        "n_consistent": n_consistent,
        "n_pairs": n_pairs,
    }


class TSEDAccumulator:
    """Accumulates per-scene TSED scores across batches.

    Usage:
        acc = TSEDAccumulator()
        for batch in loader:
            acc.update(images, intrinsics, extrinsics)
        result = acc.compute()  # {'tsed': ..., 'median_sed': ..., 'n_matches': ...}
    """

    def __init__(
        self,
        match_threshold: int = 10,
        sed_threshold: float = 2.0,
        lowe_ratio: float = 0.75,
        sift_edge_threshold: int = 30,
    ):
        self.match_threshold = match_threshold
        self.sed_threshold = sed_threshold
        self.lowe_ratio = lowe_ratio
        self.sift_edge_threshold = sift_edge_threshold
        self.reset()

    def update(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Dict[str, float]:
        """Add a batch of scenes.

        Args:
            images: (B, T, C, H, W) predicted images in [0, 1] float.
            intrinsics: (B, T, 3, 3) camera intrinsics at image resolution.
            extrinsics: (B, T, 4, 4) world-to-camera extrinsics.

        Returns:
            Per-batch mean metrics dict.
        """
        B = images.shape[0]
        batch_tsed = []
        batch_median_sed = []
        batch_n_matches = []

        for b in range(B):
            # Convert (T, C, H, W) float [0,1] -> (T, H, W, 3) uint8
            imgs_np = (images[b].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            K_np = intrinsics[b].cpu().numpy()
            E_np = extrinsics[b].cpu().numpy()

            result = compute_tsed_for_scene(
                imgs_np, K_np, E_np,
                match_threshold=self.match_threshold,
                sed_threshold=self.sed_threshold,
                lowe_ratio=self.lowe_ratio,
                sift_edge_threshold=self.sift_edge_threshold,
            )

            self._tsed_scores.append(result["tsed"])
            self._median_seds.append(result["median_sed"])
            self._n_matches.append(result["n_matches"])
            self._n_consistent += result["n_consistent"]
            self._n_pairs += result["n_pairs"]

            batch_tsed.append(result["tsed"])
            batch_median_sed.append(result["median_sed"])
            batch_n_matches.append(result["n_matches"])

        return {
            "tsed": float(np.mean(batch_tsed)) if batch_tsed else 0.0,
            "median_sed": float(np.mean(batch_median_sed)) if batch_median_sed else 0.0,
            "n_matches": float(np.mean(batch_n_matches)) if batch_n_matches else 0.0,
        }

    def compute(self) -> Optional[Dict[str, float]]:
        """Return aggregated TSED metrics. None if no samples."""
        if not self._tsed_scores:
            return None
        return {
            "tsed": float(np.mean(self._tsed_scores)),
            "median_sed": float(np.mean(self._median_seds)),
            "n_matches": float(np.mean(self._n_matches)),
            "consistency_rate": self._n_consistent / self._n_pairs if self._n_pairs > 0 else 0.0,
        }

    def reset(self):
        self._tsed_scores = []
        self._median_seds = []
        self._n_matches = []
        self._n_consistent = 0
        self._n_pairs = 0


