"""
Evaluation metrics for predicted 3D scene coordinate maps.

Five categories:
1. Per-pixel coordinate error (L1/L2) in observed regions
2. Depth metrics (abs_rel, sq_rel, rmse, delta thresholds) derived from world coords
3. Pose estimation via PnP+RANSAC
4. Reprojection consistency across output views
5. Multi-view 3D consistency across output views

All metrics operate in denormalized (real-world) coordinate space.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Category 1: Per-Pixel Coordinate Error
# ---------------------------------------------------------------------------

def compute_coord_l1(
    pred: torch.Tensor,   # (N, 3, H, W)
    gt: torch.Tensor,     # (N, 3, H, W)
    mask: torch.Tensor,   # (N, 1, H, W)
) -> Dict[str, float]:
    mask_sum = mask.sum().clamp(min=1)
    diff = (pred - gt).abs()  # (N, 3, H, W)
    per_axis = (diff * mask).sum(dim=(0, 2, 3)) / mask_sum  # (3,)
    combined = (diff.mean(dim=1, keepdim=True) * mask).sum() / mask_sum
    return {
        "coord_l1": combined.item(),
        "coord_l1_x": per_axis[0].item(),
        "coord_l1_y": per_axis[1].item(),
        "coord_l1_z": per_axis[2].item(),
    }


def compute_coord_l2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    mask_sum = mask.sum().clamp(min=1)
    diff_sq = (pred - gt) ** 2  # (N, 3, H, W)
    mse = (diff_sq.mean(dim=1, keepdim=True) * mask).sum() / mask_sum
    rmse = mse.sqrt()
    return {
        "coord_mse": mse.item(),
        "coord_rmse": rmse.item(),
    }


# ---------------------------------------------------------------------------
# Category 2: Depth Metrics
# ---------------------------------------------------------------------------

def world_coords_to_camera_depth(
    world_coords: torch.Tensor,  # (N, 3, H, W)
    w2c: torch.Tensor,           # (N, 4, 4)
) -> torch.Tensor:
    """Transform world coords to camera space, return Z (depth). Shape (N, H, W)."""
    N, _, H, W = world_coords.shape
    pts = world_coords.reshape(N, 3, -1)  # (N, 3, HW)
    ones = torch.ones(N, 1, H * W, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)  # (N, 4, HW)
    cam_pts = torch.bmm(w2c.float(), pts_h.float())  # (N, 4, HW)
    depth = cam_pts[:, 2, :].reshape(N, H, W)  # Z-forward (OpenCV)
    return depth


def compute_depth_metrics(
    pred_depth: torch.Tensor,  # (N, H, W)
    gt_depth: torch.Tensor,    # (N, H, W)
    mask: torch.Tensor,         # (N, 1, H, W)
) -> Dict[str, float]:
    mask_2d = mask[:, 0]  # (N, H, W)
    # Only evaluate where depth is positive in both pred and GT
    valid = mask_2d.bool() & (pred_depth > 0) & (gt_depth > 0)
    n_valid = valid.sum().clamp(min=1)

    pred_v = pred_depth[valid]
    gt_v = gt_depth[valid]

    thresh = torch.max(pred_v / gt_v, gt_v / pred_v)
    delta_1 = (thresh < 1.25).float().mean().item()
    delta_2 = (thresh < 1.25 ** 2).float().mean().item()
    delta_3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_diff = (pred_v - gt_v).abs()
    abs_rel = (abs_diff / gt_v.clamp(min=1e-8)).mean().item()
    sq_rel = ((abs_diff ** 2) / gt_v.clamp(min=1e-8)).mean().item()
    rmse = abs_diff.pow(2).mean().sqrt().item()

    log_pred = pred_v.clamp(min=1e-8).log()
    log_gt = gt_v.clamp(min=1e-8).log()
    rmse_log = (log_pred - log_gt).pow(2).mean().sqrt().item()

    return {
        "depth/abs_rel": abs_rel,
        "depth/sq_rel": sq_rel,
        "depth/rmse": rmse,
        "depth/rmse_log": rmse_log,
        "depth/delta_1": delta_1,
        "depth/delta_2": delta_2,
        "depth/delta_3": delta_3,
    }


# ---------------------------------------------------------------------------
# Category 3: Pose Estimation via PnP+RANSAC
# ---------------------------------------------------------------------------

def estimate_pose_pnp(
    pred_coords: torch.Tensor,   # (B, T_out, 3, H, W)
    intrinsics: torch.Tensor,    # (B, T_out, 3, 3)
    gt_extrinsics: torch.Tensor, # (B, T_out, 4, 4)
    mask: torch.Tensor,          # (B, T_out, 1, H, W)
) -> Dict[str, float]:
    try:
        import cv2
    except ImportError:
        return {}

    B, T_out, _, H, W = pred_coords.shape
    rot_errors = []
    trans_errors = []
    inlier_ratios = []

    for b in range(B):
        for t in range(T_out):
            m = mask[b, t, 0].cpu().numpy() > 0.5  # (H, W)
            if m.sum() < 10:
                continue

            ys, xs = m.nonzero()
            # Subsample to ~5000 points
            if len(ys) > 5000:
                idx = torch.randperm(len(ys))[:5000].numpy()
                ys, xs = ys[idx], xs[idx]

            ys_t = torch.from_numpy(ys).long()
            xs_t = torch.from_numpy(xs).long()
            coords_3d = pred_coords[b, t, :, ys_t, xs_t].T.float().cpu().numpy().astype("float64")  # (N, 3)
            pts_2d = torch.stack([xs_t.float(), ys_t.float()], dim=1).numpy().astype("float64")  # (N, 2)

            K = intrinsics[b, t].cpu().numpy().astype("float64")

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                coords_3d, pts_2d, K, None,
                flags=cv2.SOLVEPNP_SQPNP,
                reprojectionError=8.0,
                iterationsCount=1000,
            )
            if not success or inliers is None:
                continue

            inlier_ratios.append(len(inliers) / len(ys))

            R_pred, _ = cv2.Rodrigues(rvec)
            t_pred = tvec.flatten()

            # GT w2c
            gt_w2c = gt_extrinsics[b, t].cpu().numpy().astype("float64")
            R_gt = gt_w2c[:3, :3]
            t_gt = gt_w2c[:3, 3]

            # Rotation error in degrees
            R_rel = R_pred @ R_gt.T
            trace_val = max(min(R_rel.trace(), 3.0), -1.0)
            import math
            rot_err = math.degrees(math.acos((trace_val - 1.0) / 2.0))
            rot_errors.append(rot_err)

            # Translation error (normalized coord space)
            trans_err = float(((t_pred - t_gt) ** 2).sum() ** 0.5)
            trans_errors.append(trans_err)

    if not rot_errors:
        return {"pnp_n_valid": 0}

    import numpy as np
    rot_arr = np.array(rot_errors)
    trans_arr = np.array(trans_errors)

    return {
        "pnp_rot_err_mean": float(rot_arr.mean()),
        "pnp_rot_err_median": float(np.median(rot_arr)),
        "pnp_trans_err_mean": float(trans_arr.mean()),
        "pnp_trans_err_median": float(np.median(trans_arr)),
        "pnp_5deg_5unit": float(((rot_arr < 5) & (trans_arr < 0.05)).mean()),
        "pnp_10deg_10unit": float(((rot_arr < 10) & (trans_arr < 0.10)).mean()),
        "pnp_inlier_ratio_mean": float(np.mean(inlier_ratios)),
        "pnp_n_valid": len(rot_errors),
    }


# ---------------------------------------------------------------------------
# Category 4: Reprojection Consistency (T_out >= 2)
# ---------------------------------------------------------------------------

def _project_to_pixel(
    world_pts: torch.Tensor,   # (N, 3, H, W)
    w2c: torch.Tensor,         # (N, 4, 4)
    intrinsic: torch.Tensor,   # (N, 3, 3)
) -> torch.Tensor:
    """Project world coords to pixel coords. Returns (N, 2, H, W) [x, y]."""
    N, _, H, W = world_pts.shape
    pts = world_pts.reshape(N, 3, -1)  # (N, 3, HW)
    ones = torch.ones(N, 1, H * W, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=1)  # (N, 4, HW)
    cam = torch.bmm(w2c.float(), pts_h.float())[:, :3]  # (N, 3, HW)
    proj = torch.bmm(intrinsic.float(), cam)  # (N, 3, HW)
    z = proj[:, 2:3].clamp(min=1e-8)
    px = (proj[:, :2] / z).reshape(N, 2, H, W)  # (N, 2, H, W)
    return px


def compute_reprojection_consistency(
    pred_coords: torch.Tensor,   # (N, 3, H, W)
    gt_coords: torch.Tensor,     # (N, 3, H, W)
    extrinsics: torch.Tensor,    # (N, 4, 4)
    intrinsics: torch.Tensor,    # (N, 3, 3)
    mask: torch.Tensor,          # (N, 1, H, W)
) -> Dict[str, float]:
    N, _, H, W = pred_coords.shape
    if N < 2:
        return {}

    errors = []
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            m_i = mask[i, 0] > 0.5  # (H, W)
            if m_i.sum() < 10:
                continue

            # Project view i's coords into view j's image plane
            pred_px = _project_to_pixel(
                pred_coords[i:i+1], extrinsics[j:j+1], intrinsics[j:j+1],
            )[0]  # (2, H, W)
            gt_px = _project_to_pixel(
                gt_coords[i:i+1], extrinsics[j:j+1], intrinsics[j:j+1],
            )[0]  # (2, H, W)

            # Check that projected points are within bounds
            in_bounds = (
                (pred_px[0] >= 0) & (pred_px[0] < W) &
                (pred_px[1] >= 0) & (pred_px[1] < H) &
                (gt_px[0] >= 0) & (gt_px[0] < W) &
                (gt_px[1] >= 0) & (gt_px[1] < H)
            )
            valid = m_i & in_bounds

            if valid.sum() < 10:
                continue

            err = ((pred_px - gt_px) ** 2).sum(dim=0).sqrt()  # (H, W) pixel distance
            errors.append(err[valid])
            n_pairs += 1

    if not errors:
        return {"reproj_n_pairs": 0}

    all_err = torch.cat(errors)
    return {
        "reproj_mean": all_err.mean().item(),
        "reproj_median": all_err.median().item(),
        "reproj_n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
# Category 5: Multi-View 3D Consistency (T_out >= 2)
# ---------------------------------------------------------------------------

def compute_multiview_3d_consistency(
    pred_coords: torch.Tensor,   # (N, 3, H, W)
    extrinsics: torch.Tensor,    # (N, 4, 4)
    intrinsics: torch.Tensor,    # (N, 3, 3)
    mask: torch.Tensor,          # (N, 1, H, W)
) -> Dict[str, float]:
    N, _, H, W = pred_coords.shape
    if N < 2:
        return {}

    l1_errors = []
    crossreproj_errors = []
    n_pairs = 0

    for i in range(N):
        for j in range(i + 1, N):
            m_i = mask[i, 0] > 0.5
            m_j = mask[j, 0] > 0.5

            # Project view i's 3D coords into view j's image
            px_ij = _project_to_pixel(
                pred_coords[i:i+1], extrinsics[j:j+1], intrinsics[j:j+1],
            )[0]  # (2, H, W)

            # Check bounds
            in_bounds = (
                (px_ij[0] >= 0) & (px_ij[0] < W - 1) &
                (px_ij[1] >= 0) & (px_ij[1] < H - 1)
            )
            valid = m_i & in_bounds

            if valid.sum() < 10:
                continue

            # Normalize pixel coords to [-1, 1] for grid_sample
            grid_x = px_ij[0:1] / (W - 1) * 2 - 1  # (1, H, W)
            grid_y = px_ij[1:2] / (H - 1) * 2 - 1
            grid = torch.stack([grid_x[0], grid_y[0]], dim=-1).unsqueeze(0)  # (1, H, W, 2)

            # Sample view j's coords at projected locations
            sampled_j = F.grid_sample(
                pred_coords[j:j+1].float(), grid.float(),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )[0]  # (3, H, W)

            # Also check that sampled location in j is observed
            sampled_mask_j = F.grid_sample(
                mask[j:j+1].float(), grid.float(),
                mode="nearest", padding_mode="zeros", align_corners=True,
            )[0, 0] > 0.5  # (H, W)

            both_valid = valid & sampled_mask_j

            if both_valid.sum() < 10:
                continue

            # 3D discrepancy: L1 between view i coords and sampled view j coords
            l1 = (pred_coords[i] - sampled_j).abs().mean(dim=0)  # (H, W)
            l1_errors.append(l1[both_valid])

            # Cross-view reprojection: project sampled j coords back to view i
            sampled_j_full = sampled_j.unsqueeze(0)  # (1, 3, H, W)
            px_ji = _project_to_pixel(
                sampled_j_full, extrinsics[i:i+1], intrinsics[i:i+1],
            )[0]  # (2, H, W)

            # Original pixel grid for view i
            grid_y_i, grid_x_i = torch.meshgrid(
                torch.arange(H, device=pred_coords.device, dtype=torch.float32),
                torch.arange(W, device=pred_coords.device, dtype=torch.float32),
                indexing="ij",
            )
            orig_px = torch.stack([grid_x_i, grid_y_i], dim=0)  # (2, H, W)

            reproj_err = ((px_ji - orig_px) ** 2).sum(dim=0).sqrt()
            crossreproj_errors.append(reproj_err[both_valid])
            n_pairs += 1

    if not l1_errors:
        return {"mv_n_pairs": 0}

    all_l1 = torch.cat(l1_errors)
    all_reproj = torch.cat(crossreproj_errors)
    return {
        "mv_3d_l1_mean": all_l1.mean().item(),
        "mv_3d_l1_median": all_l1.median().item(),
        "mv_crossreproj_mean": all_reproj.mean().item(),
        "mv_crossreproj_median": all_reproj.median().item(),
        "mv_n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
# Convenience Wrapper
# ---------------------------------------------------------------------------

def _unnormalize_coords(
    coords: torch.Tensor,   # (..., 3, H, W) in [-1, 1]
    norm_min: torch.Tensor,  # scalar or (B,) — padded bounds (world space)
    norm_max: torch.Tensor,  # scalar or (B,)
    log_scale_coords: bool = False,
) -> torch.Tensor:
    """Invert the [-1, 1] normalization back to world coordinates.

    norm_min/norm_max are padded bounds in world space.
    """
    size = (norm_max - norm_min).clamp(min=1e-6)
    if log_scale_coords:
        log_max = torch.log1p(size)
        coords_log = (coords + 1.0) / 2.0 * log_max  # [0, log_max]
        return torch.expm1(coords_log) + norm_min
    return (coords + 1.0) / 2.0 * size + norm_min


def compute_all_coord_metrics(
    pred_coords: torch.Tensor,          # (B, T_out, 3, H, W)
    gt_coord_maps: torch.Tensor,        # (B, T, 3, H_gt, W_gt)
    extrinsics: torch.Tensor,           # (B, T, 4, 4)
    intrinsics: torch.Tensor,           # (B, T, 3, 3)
    warp_masks: torch.Tensor,           # (B, T_out, 1, H_gt, W_gt)
    T_in: int,
    T_out: int,
    coord_norm_min: torch.Tensor = None,  # (B,) per-scene normalization min
    coord_norm_max: torch.Tensor = None,  # (B,) per-scene normalization max
    log_scale_coords: bool = False,
) -> Dict[str, float]:
    """Compute all coordinate map metrics, handling resolution alignment internally."""
    B = pred_coords.shape[0]
    pred_H, pred_W = pred_coords.shape[-2:]

    # Extract output-view data
    gt_out = gt_coord_maps[:, T_in:]      # (B, T_out, 3, H_gt, W_gt)
    ext_out = extrinsics[:, T_in:]        # (B, T_out, 4, 4)
    intr_out = intrinsics[:, T_in:].clone()  # (B, T_out, 3, 3)

    gt_H, gt_W = gt_out.shape[-2:]

    # Resize GT coords to match pred resolution
    if (gt_H, gt_W) != (pred_H, pred_W):
        gt_flat = gt_out.reshape(B * T_out, 3, gt_H, gt_W)
        gt_flat = F.interpolate(gt_flat.float(), size=(pred_H, pred_W), mode="bilinear", align_corners=False)
        gt_out = gt_flat.reshape(B, T_out, 3, pred_H, pred_W)

        # Rescale intrinsics
        intr_out[:, :, 0, :] *= pred_W / gt_W
        intr_out[:, :, 1, :] *= pred_H / gt_H

    # Resize warp masks to match pred resolution (may differ from both gt and pred)
    mask_H, mask_W = warp_masks.shape[-2:]
    if (mask_H, mask_W) != (pred_H, pred_W):
        mask_flat = warp_masks.reshape(B * T_out, 1, mask_H, mask_W)
        mask_flat = F.interpolate(mask_flat.float(), size=(pred_H, pred_W), mode="nearest")
        warp_masks = mask_flat.reshape(B, T_out, 1, pred_H, pred_W)

    # Move to same device
    device = pred_coords.device
    gt_out = gt_out.to(device)
    warp_masks = warp_masks.to(device)
    ext_out = ext_out.to(device)
    intr_out = intr_out.to(device)

    # Flatten batch and time for per-pixel metrics (normalized coords)
    pred_flat = pred_coords.reshape(B * T_out, 3, pred_H, pred_W).float()
    gt_flat = gt_out.reshape(B * T_out, 3, pred_H, pred_W).float()
    mask_flat = warp_masks.reshape(B * T_out, 1, pred_H, pred_W).float()
    ext_flat = ext_out.reshape(B * T_out, 4, 4)
    intr_flat = intr_out.reshape(B * T_out, 3, 3)

    metrics = {}

    # Unnormalize coords for all metrics (Categories 1-5)
    # These need real-world coords to be consistent with extrinsics/intrinsics
    if coord_norm_min is not None and coord_norm_max is not None:
        # Reshape scalars for broadcasting: (B,) -> (B, 1, 1, 1, 1)
        nm = coord_norm_min.to(device).float().view(B, 1, 1, 1, 1)
        nx = coord_norm_max.to(device).float().view(B, 1, 1, 1, 1)
        pred_world = _unnormalize_coords(pred_coords.float(), nm, nx, log_scale_coords=log_scale_coords)  # (B, T_out, 3, H, W)
        gt_world = _unnormalize_coords(gt_out.float(), nm, nx, log_scale_coords=log_scale_coords)         # (B, T_out, 3, H, W)
    else:
        pred_world = pred_coords.float()
        gt_world = gt_out.float()

    pred_world_flat = pred_world.reshape(B * T_out, 3, pred_H, pred_W)
    gt_world_flat = gt_world.reshape(B * T_out, 3, pred_H, pred_W)

    # Category 1: Per-pixel coord error (denormalized coords)
    metrics.update(compute_coord_l1(pred_world_flat, gt_world_flat, mask_flat))
    metrics.update(compute_coord_l2(pred_world_flat, gt_world_flat, mask_flat))

    # Category 2: Depth metrics (need real-world coords for correct camera-space depth)
    pred_depth = world_coords_to_camera_depth(pred_world_flat, ext_flat)
    gt_depth = world_coords_to_camera_depth(gt_world_flat, ext_flat)
    metrics.update(compute_depth_metrics(pred_depth, gt_depth, mask_flat))

    # Category 3: PnP (uses real-world coords for 3D-2D correspondences)
    metrics.update(estimate_pose_pnp(pred_world, intr_out, ext_out, warp_masks))

    # Categories 4 & 5: Multi-view (need T_out >= 2, use real-world coords)
    if T_out >= 2:
        reproj_all = []
        mv3d_all = []
        for b in range(B):
            reproj_all.append(compute_reprojection_consistency(
                pred_world[b], gt_world[b],
                ext_out[b], intr_out[b], warp_masks[b],
            ))
            mv3d_all.append(compute_multiview_3d_consistency(
                pred_world[b], ext_out[b], intr_out[b], warp_masks[b],
            ))

        # Aggregate reprojection
        reproj_means = [r["reproj_mean"] for r in reproj_all if r.get("reproj_n_pairs", 0) > 0]
        reproj_medians = [r["reproj_median"] for r in reproj_all if r.get("reproj_n_pairs", 0) > 0]
        total_pairs = sum(r.get("reproj_n_pairs", 0) for r in reproj_all)
        if reproj_means:
            import numpy as np
            metrics["reproj_mean"] = float(np.mean(reproj_means))
            metrics["reproj_median"] = float(np.mean(reproj_medians))
        metrics["reproj_n_pairs"] = total_pairs

        # Aggregate multi-view 3D
        mv_l1 = [r["mv_3d_l1_mean"] for r in mv3d_all if r.get("mv_n_pairs", 0) > 0]
        mv_l1_med = [r["mv_3d_l1_median"] for r in mv3d_all if r.get("mv_n_pairs", 0) > 0]
        mv_reproj = [r["mv_crossreproj_mean"] for r in mv3d_all if r.get("mv_n_pairs", 0) > 0]
        mv_reproj_med = [r["mv_crossreproj_median"] for r in mv3d_all if r.get("mv_n_pairs", 0) > 0]
        mv_pairs = sum(r.get("mv_n_pairs", 0) for r in mv3d_all)
        if mv_l1:
            import numpy as np
            metrics["mv_3d_l1_mean"] = float(np.mean(mv_l1))
            metrics["mv_3d_l1_median"] = float(np.mean(mv_l1_med))
            metrics["mv_crossreproj_mean"] = float(np.mean(mv_reproj))
            metrics["mv_crossreproj_median"] = float(np.mean(mv_reproj_med))
        metrics["mv_n_pairs"] = mv_pairs

    return metrics
