# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GPU forward-warp rendering utilities for GenRec.

Ported from GEN3C (cosmos_predict1/diffusion/inference/forward_warp_utils_pytorch.py).
Provides bilinear splatting with depth-weighted blending, camera-ray back-face
culling, triangle-mesh foreground occlusion masking, and reliable depth filtering.

Extrinsic convention: **world-to-camera (w2c)** matrices throughout.
"""

from typing import Optional, Tuple
import warnings

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_grid(b: int, h: int, w: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Dense (x, y) coordinate grid of shape ``(b, 2, h, w)``."""
    x = torch.arange(0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    y = torch.arange(0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([x, y], dim=1)


def get_max_exponent_for_dtype(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 80.0
    elif dtype == torch.float16:
        return 10.0
    elif dtype == torch.float32:
        return 80.0
    elif dtype == torch.float64:
        return 700.0
    return 80.0


def inverse_with_conversion(mtx: torch.Tensor) -> torch.Tensor:
    """Matrix inverse cast through float32 for numerical safety."""
    return torch.linalg.inv(mtx.to(torch.float32)).to(mtx.dtype)


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def get_camera_rays(h: int, w: int, intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Back-project pixel grid into normalized camera-space ray directions.

    Args:
        h, w: image resolution.
        intrinsic: ``(b, 3, 3)`` camera intrinsic matrices.

    Returns:
        ``(b, h, w, 3)`` normalized ray directions.
    """
    device = intrinsic.device
    dtype = intrinsic.dtype
    x = torch.arange(0, w, device=device, dtype=dtype)[None].repeat(h, 1)      # (h, w)
    y = torch.arange(0, h, device=device, dtype=dtype)[:, None].repeat(1, w)    # (h, w)
    ones = torch.ones(h, w, device=device, dtype=dtype)
    pos_homo = torch.stack([x, y, ones], dim=2)[None, :, :, :, None]            # (1, h, w, 3, 1)

    K_inv = inverse_with_conversion(intrinsic)[:, None, None]                    # (b, 1, 1, 3, 3)
    unnorm = torch.matmul(K_inv, pos_homo).squeeze(-1)                           # (b, h, w, 3)
    norm = torch.norm(unnorm, dim=-1, keepdim=True).clamp(min=1e-8)
    return unnorm / norm


def compute_transformed_points(
    depth1: torch.Tensor,            # (b, 1, h, w)
    w2c1: torch.Tensor,              # (b, 4, 4)
    w2c2: torch.Tensor,              # (b, 4, 4)
    intrinsic1: torch.Tensor,        # (b, 3, 3)
    is_depth: bool = True,
    intrinsic2: Optional[torch.Tensor] = None,
    return_cam_points: bool = False,
) -> torch.Tensor:
    """
    Un-project depth from view-1, transform to view-2 camera space, project.

    Returns ``(b, h, w, 3, 1)`` projected (homogeneous pixel) coordinates.
    If *return_cam_points* is True, also returns ``(b, h, w, 3)`` camera-space
    3-D points in view-2.
    """
    b, _, h, w = depth1.shape
    device, dtype = depth1.device, depth1.dtype
    if intrinsic2 is None:
        intrinsic2 = intrinsic1.clone()

    # Relative transform: view-1 → view-2
    transformation = torch.bmm(w2c2, inverse_with_conversion(w2c1))  # (b, 4, 4)

    x = torch.arange(0, w, device=device, dtype=dtype)[None].repeat(h, 1)
    y = torch.arange(0, h, device=device, dtype=dtype)[:, None].repeat(1, w)
    ones = torch.ones(h, w, device=device, dtype=dtype)
    ones_4d = ones[None, :, :, None, None].repeat(b, 1, 1, 1, 1)
    pos_homo = torch.stack([x, y, ones], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

    K1_inv = inverse_with_conversion(intrinsic1)[:, None, None]
    K2_4d = intrinsic2[:, None, None]
    depth_4d = depth1[:, 0, :, :, None, None]                         # (b, h, w, 1, 1)
    trans_4d = transformation[:, None, None]

    unnorm_pos = torch.matmul(K1_inv, pos_homo)                       # (b, h, w, 3, 1)
    if is_depth:
        world_pts = depth_4d * unnorm_pos
    else:
        direction = unnorm_pos / torch.norm(unnorm_pos, dim=-2, keepdim=True).clamp(min=1e-8)
        world_pts = depth_4d * direction

    world_pts_h = torch.cat([world_pts, ones_4d], dim=3)              # (b, h, w, 4, 1)
    trans_world = torch.matmul(trans_4d, world_pts_h)[:, :, :, :3]    # (b, h, w, 3, 1)
    projected = torch.matmul(K2_4d, trans_world)                      # (b, h, w, 3, 1)

    if return_cam_points:
        return projected, trans_world.squeeze(-1)
    return projected


def project_points(
    world_points: torch.Tensor,  # (b, h, w, 3)
    w2c: torch.Tensor,           # (b, 4, 4)
    intrinsic: torch.Tensor,     # (b, 3, 3)
    return_cam_points: bool = False,
) -> torch.Tensor:
    """Project world points → pixel coords ``(b, h, w, 3, 1)``."""
    pts = world_points.unsqueeze(-1)                                  # (b, h, w, 3, 1)
    b, h, w, _, _ = pts.shape
    ones = torch.ones(b, h, w, 1, 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=3)                             # (b, h, w, 4, 1)

    cam_pts = torch.matmul(w2c[:, None, None], pts_h)[:, :, :, :3]   # (b, h, w, 3, 1)
    proj = torch.matmul(intrinsic[:, None, None], cam_pts)            # (b, h, w, 3, 1)

    if return_cam_points:
        return proj, cam_pts.squeeze(-1)
    return proj


# ---------------------------------------------------------------------------
# Bilinear splatting
# ---------------------------------------------------------------------------

def bilinear_splatting(
    frame1: torch.Tensor,                   # (b, c, h, w)
    mask1: Optional[torch.Tensor],          # (b, 1, h, w)
    depth1: torch.Tensor,                   # (b, 1, h, w)
    flow12: torch.Tensor,                   # (b, 2, h, w)
    flow12_mask: Optional[torch.Tensor],    # (b, 1, h, w)
    is_image: bool = False,
    n_views: int = 1,
    depth_weight_scale: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Depth-weighted bilinear forward splatting.

    When ``n_views > 1`` the batch dimension is assumed to contain *n_views*
    repeats of each target; the accumulation sums over the view axis so that
    all source views blend into a single output per target.
    """
    b, c, h, w = frame1.shape
    device, dtype = frame1.device, frame1.dtype

    if mask1 is None:
        mask1 = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    if flow12_mask is None:
        flow12_mask = torch.ones(b, 1, h, w, device=device, dtype=dtype)

    grid = create_grid(b, h, w, device=device, dtype=dtype)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = torch.floor(trans_pos_offset).long()
    trans_pos_ceil = torch.ceil(trans_pos_offset).long()

    trans_pos_offset = torch.stack([
        trans_pos_offset[:, 0].clamp(0, w + 1),
        trans_pos_offset[:, 1].clamp(0, h + 1),
    ], dim=1)
    trans_pos_floor = torch.stack([
        trans_pos_floor[:, 0].clamp(0, w + 1),
        trans_pos_floor[:, 1].clamp(0, h + 1),
    ], dim=1)
    trans_pos_ceil = torch.stack([
        trans_pos_ceil[:, 0].clamp(0, w + 1),
        trans_pos_ceil[:, 1].clamp(0, h + 1),
    ], dim=1)

    # Bilinear proximity weights
    prox_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
              (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
              (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
    prox_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
              (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
    prox_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
              (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

    # Depth-based inverse weighting
    clamped_depth = torch.clamp(depth1, min=0)
    log_depth = torch.log1p(clamped_depth)
    exponent = log_depth / (log_depth.max() + 1e-7) * depth_weight_scale
    max_exp = get_max_exponent_for_dtype(dtype)
    depth_weights = torch.exp(exponent.clamp(max=max_exp)) + 1e-7

    # Combine masks + depth weighting → per-corner weights   (b, h, w, 1) layout
    def _to_hw1(t):
        return torch.moveaxis(t, [0, 1, 2, 3], [0, 3, 1, 2])

    shared_mask = mask1 * flow12_mask / depth_weights
    w_nw = _to_hw1(prox_nw * shared_mask)
    w_sw = _to_hw1(prox_sw * shared_mask)
    w_ne = _to_hw1(prox_ne * shared_mask)
    w_se = _to_hw1(prox_se * shared_mask)

    # Accumulate into padded buffer
    warped = torch.zeros(b, h + 2, w + 2, c, dtype=dtype, device=device)
    weights = torch.zeros(b, h + 2, w + 2, 1, dtype=dtype, device=device)

    frame_hw = _to_hw1(frame1)   # (b, h, w, c)
    bi = torch.arange(b, device=device)[:, None, None]

    for corner_y, corner_x, corner_w in [
        (trans_pos_floor[:, 1], trans_pos_floor[:, 0], w_nw),
        (trans_pos_ceil[:, 1],  trans_pos_floor[:, 0], w_sw),
        (trans_pos_floor[:, 1], trans_pos_ceil[:, 0],  w_ne),
        (trans_pos_ceil[:, 1],  trans_pos_ceil[:, 0],  w_se),
    ]:
        warped.index_put_((bi, corner_y, corner_x), frame_hw * corner_w, accumulate=True)
        weights.index_put_((bi, corner_y, corner_x), corner_w, accumulate=True)

    # Multi-view reduction
    if n_views > 1:
        warped = warped.reshape(b // n_views, n_views, h + 2, w + 2, c).sum(1)
        weights = weights.reshape(b // n_views, n_views, h + 2, w + 2, 1).sum(1)

    # Back to channel-first, crop padding
    warped = torch.moveaxis(warped, [0, 1, 2, 3], [0, 2, 3, 1])[:, :, 1:-1, 1:-1]
    weights = torch.moveaxis(weights, [0, 1, 2, 3], [0, 2, 3, 1])[:, :, 1:-1, 1:-1]
    weights = torch.nan_to_num(weights, nan=1000.0)

    valid = weights > 0
    fill = -1.0 if is_image else 0.0
    out = torch.where(valid, warped / weights, torch.tensor(fill, dtype=dtype, device=device))
    mask_out = valid.to(dtype)

    if is_image:
        out = out.clamp(-1, 1)

    return out, mask_out


# ---------------------------------------------------------------------------
# Mesh construction & foreground masking helpers
# ---------------------------------------------------------------------------

def points_to_mesh(
    points: torch.Tensor,   # (H, W, 3)
    mask: torch.Tensor,     # (H, W)
    resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dense depth-grid of 3-D points into a triangle mesh.

    Returns ``(vertices (N, 3), faces (M, 3))`` — may be empty if no valid
    patches exist.
    """
    H, W = points.shape[:2]

    if resolution is not None:
        new_H, new_W = resolution
        points = F.interpolate(
            points.permute(2, 0, 1).unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0), size=(new_H, new_W), mode="nearest",
        ).squeeze(0).squeeze(0).bool()
        H, W = new_H, new_W

    verts_idx = torch.arange(H * W, device=points.device).reshape(H, W)

    # 2×2 patch validity — any corner in mask
    valid = mask[:-1, :-1] | mask[:-1, 1:] | mask[1:, :-1] | mask[1:, 1:]
    vh, vw = torch.where(valid)
    n = len(vh)
    if n == 0:
        empty3 = torch.empty(0, 3, device=points.device)
        return empty3, empty3.long()

    tl = verts_idx[vh, vw]
    tr = verts_idx[vh, vw + 1]
    bl = verts_idx[vh + 1, vw]
    br = verts_idx[vh + 1, vw + 1]

    faces = torch.cat([
        torch.stack([tl, tr, bl], dim=1),
        torch.stack([tr, br, bl], dim=1),
    ], dim=0)

    flat_pts = points.reshape(-1, 3)
    used = torch.unique(faces.flatten())
    remap = torch.full((H * W,), -1, dtype=torch.long, device=points.device)
    remap[used] = torch.arange(len(used), device=points.device)

    return flat_pts[used], remap[faces.flatten()].reshape(-1, 3)


def reliable_depth_mask_range_batch(
    depth: torch.Tensor,
    window_size: int = 5,
    ratio_thresh: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Local depth-variation filter.

    Returns a boolean mask ``(B, 1, H, W)`` that is True where the depth is
    "reliable" (local range / local mean < *ratio_thresh*).
    """
    assert window_size % 2 == 1
    if depth.dim() == 3:
        d = depth.unsqueeze(1)
    elif depth.dim() == 4:
        d = depth
    else:
        raise ValueError("depth must be (B, H, W) or (B, 1, H, W)")
    pad = window_size // 2
    local_max = F.max_pool2d(d, window_size, stride=1, padding=pad)
    local_min = -F.max_pool2d(-d, window_size, stride=1, padding=pad)
    local_mean = F.avg_pool2d(d, window_size, stride=1, padding=pad)
    ratio = (local_max - local_min) / (local_mean + eps)
    return (ratio < ratio_thresh) & (d > 0)


def depth_gradient_mask_batch_relative(
    depth: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mask pixels at depth discontinuities using a *relative* gradient threshold.

    Computes central-difference spatial gradients of ``depth`` and rejects
    pixels where ``grad_mag / depth > threshold``. The threshold is
    dimensionless (max fractional depth change per pixel), so the same value
    is meaningful across scenes with different metric scales.

    Mirrors ``scripts/visualize_pointclouds.py::depth_edge_mask_relative`` so a
    threshold tuned interactively in the visualizer transfers 1:1 here.

    Args:
        depth: (B, H, W) or (B, 1, H, W) per-pixel camera-Z depth.
        threshold: dimensionless max fractional depth change per pixel.

    Returns:
        Boolean mask (B, 1, H, W), True = keep.
    """
    if depth.dim() == 3:
        d = depth.unsqueeze(1)
    elif depth.dim() == 4:
        d = depth
    else:
        raise ValueError("depth must be (B, H, W) or (B, 1, H, W)")
    grad_y = torch.zeros_like(d)
    grad_x = torch.zeros_like(d)
    grad_y[..., 1:-1, :] = (d[..., 2:, :] - d[..., :-2, :]) / 2.0
    grad_x[..., :, 1:-1] = (d[..., :, 2:] - d[..., :, :-2]) / 2.0
    grad_y[..., 0, :] = d[..., 1, :] - d[..., 0, :]
    grad_y[..., -1, :] = d[..., -1, :] - d[..., -2, :]
    grad_x[..., :, 0] = d[..., :, 1] - d[..., :, 0]
    grad_x[..., :, -1] = d[..., :, -1] - d[..., :, -2]
    grad_mag = torch.sqrt(grad_y * grad_y + grad_x * grad_x)
    rel_grad = grad_mag / torch.clamp_min(d, eps)
    return (rel_grad <= threshold) & (d > 0)


# ---------------------------------------------------------------------------
# Ray-triangle intersection — soft dependency on NVIDIA Warp
# ---------------------------------------------------------------------------

_rti_func = None
_rti_checked = False


def _get_ray_triangle_func():
    global _rti_func, _rti_checked
    if _rti_checked:
        return _rti_func
    _rti_checked = True
    try:
        from genrec.utils.ray_triangle_intersection import ray_triangle_intersection_warp
        _rti_func = ray_triangle_intersection_warp
    except Exception:
        warnings.warn(
            "NVIDIA Warp not available — foreground occlusion masking will be skipped.",
            stacklevel=2,
        )
    return _rti_func


def ray_triangle_intersection(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
) -> "torch.Tensor | None":
    """Thin wrapper; returns ``None`` if Warp is unavailable."""
    fn = _get_ray_triangle_func()
    if fn is None:
        return None
    return fn(ray_origins, ray_directions, vertices, faces, device)


# ---------------------------------------------------------------------------
# Full forward-warp pipeline
# ---------------------------------------------------------------------------

def precompute_source_view_dirs(
    depth1: torch.Tensor,       # (T_in, 1, h, w)
    w2c1: torch.Tensor,         # (T_in, 4, 4)
    intrinsic1: torch.Tensor,   # (T_in, 3, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute source-side quantities for viewing-angle reliability.

    Call once per scene, reuse across all target views.

    Returns:
        ``(v_src, world_pts)`` — normalized source view directions and
        world-space 3D points, both ``(T_in, h, w, 3)``.
    """
    b, _, h, w = depth1.shape
    device, dtype = depth1.device, depth1.dtype

    c2w1 = inverse_with_conversion(w2c1)        # (b, 4, 4)
    K_inv = inverse_with_conversion(intrinsic1)  # (b, 3, 3)

    x = torch.arange(w, device=device, dtype=dtype)
    y = torch.arange(h, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ones = torch.ones(h, w, device=device, dtype=dtype)
    pixel_coords = torch.stack([xx, yy, ones], dim=-1)  # (h, w, 3)

    cam_pts = torch.matmul(K_inv[:, None, None], pixel_coords[None, :, :, :, None]).squeeze(-1)
    cam_pts = cam_pts * depth1[:, 0, :, :, None]

    R1 = c2w1[:, :3, :3]
    t1 = c2w1[:, :3, 3]
    world_pts = torch.matmul(R1[:, None, None], cam_pts.unsqueeze(-1)).squeeze(-1) + t1[:, None, None]

    cam_center1 = c2w1[:, :3, 3]
    v_src = world_pts - cam_center1[:, None, None]
    v_src = v_src / v_src.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return v_src, world_pts


def _compute_viewing_angle_cosine_fast(
    v_src: torch.Tensor,        # (b, h, w, 3) — precomputed
    world_pts: torch.Tensor,    # (b, h, w, 3) — precomputed
    w2c2: torch.Tensor,         # (b, 4, 4)
) -> torch.Tensor:
    """
    Per-pixel viewing angle cosine using precomputed source directions.

    Returns ``(b, 1, h, w)`` values in [0, 1].
    """
    c2w2 = inverse_with_conversion(w2c2)
    cam_center2 = c2w2[:, :3, 3]

    v_tgt = world_pts - cam_center2[:, None, None]
    v_tgt = v_tgt / v_tgt.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    cos_angle = (v_src * v_tgt).sum(dim=-1).clamp(min=0.0)
    return cos_angle.unsqueeze(1)


def forward_warp(
    frame1: torch.Tensor,                       # (b, c, h, w)
    mask1: Optional[torch.Tensor],              # (b, 1, h, w)
    depth1: Optional[torch.Tensor],             # (b, 1, h, w)
    w2c1: Optional[torch.Tensor],               # (b, 4, 4)
    w2c2: torch.Tensor,                         # (b, 4, 4)
    intrinsic1: Optional[torch.Tensor],         # (b, 3, 3)
    intrinsic2: Optional[torch.Tensor],         # (b, 3, 3)
    is_image: bool = True,
    cameraray_filtering: bool = False,
    foreground_masking: bool = False,
    boundary_mask: Optional[torch.Tensor] = None,
    n_views: int = 1,
    precomputed_reliability: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    depth_weight_scale: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Warp *frame1* from view-1 to view-2 via depth-weighted bilinear splatting.

    Args:
        frame1: Source frame(s).
        mask1: Validity mask for *frame1* (1=known, 0=unknown).
        depth1: Source depth.
        w2c1, w2c2: World-to-camera extrinsics for source / target.
        intrinsic1, intrinsic2: Camera intrinsics.
        is_image: Clip output to [-1, 1].
        cameraray_filtering: Back-face culling via camera rays.
        foreground_masking: Occlude warped content behind source mesh.
        boundary_mask: Required if *foreground_masking* is True.
        n_views: If > 1, batch dim contains repeated target views that are summed.
        precomputed_reliability: ``(v_src, world_pts)`` from
            :func:`precompute_source_view_dirs`.  If provided, a reliability
            map is splatted alongside the frame.

    Returns:
        ``(warped_frame, mask, warped_depth_or_None, flow, reliability_or_None)``
    """
    device = frame1.device
    b, c, h, w = frame1.shape
    dtype = frame1.dtype

    if mask1 is None:
        mask1 = torch.ones(b, 1, h, w, device=device, dtype=dtype)
    if intrinsic2 is None:
        assert intrinsic1 is not None
        intrinsic2 = intrinsic1.clone()

    # Compute optical flow via depth + extrinsics
    depth1 = torch.nan_to_num(depth1, nan=1e6).clamp(0, 1e6)
    if foreground_masking:
        trans_pts, cam_pts_tgt = compute_transformed_points(
            depth1, w2c1, w2c2, intrinsic1, True, intrinsic2, return_cam_points=True,
        )
    else:
        trans_pts = compute_transformed_points(
            depth1, w2c1, w2c2, intrinsic1, True, intrinsic2,
        )

    # Mask behind-camera points
    mask1 = mask1 * (trans_pts[:, :, :, 2, 0].unsqueeze(1) > 0)
    trans_coords = trans_pts[:, :, :, :2, 0] / (trans_pts[:, :, :, 2:3, 0] + 1e-7)
    trans_coords = trans_coords.permute(0, 3, 1, 2)  # (b, 2, h, w)

    grid = create_grid(b, h, w, device=device, dtype=dtype)
    flow12 = trans_coords - grid

    # Camera-ray back-face culling
    if cameraray_filtering:
        camera_rays = get_camera_rays(h, w, intrinsic1)  # (b, h, w, 3)
        rel_rot = torch.bmm(w2c2, inverse_with_conversion(w2c1))
        rel_rot[:, :3, 3] = 0
        inv_vec = torch.tensor([-1, -1, -1], dtype=dtype, device=device).view(1, 1, 1, 3, 1)

        normals = camera_rays.unsqueeze(-1)              # (b, h, w, 3, 1)
        ones_pad = torch.ones(b, h, w, 1, 1, device=device, dtype=dtype)
        normals_h = torch.cat([normals * inv_vec, ones_pad], dim=3)

        rot_normals = torch.matmul(rel_rot[:, None, None], normals_h).squeeze(-1)[..., :3]
        dot = (rot_normals * camera_rays).sum(-1)
        mask1 = mask1 * (dot > 0).unsqueeze(1)

    # Splat
    trans_depth = trans_pts[:, :, :, 2, 0].unsqueeze(1)
    warped_frame, mask2 = bilinear_splatting(
        frame1, mask1, trans_depth, flow12, None, is_image=is_image, n_views=n_views,
        depth_weight_scale=depth_weight_scale,
    )

    # Reliability: splat viewing-angle cosine as an extra channel
    reliability_map = None
    if precomputed_reliability is not None:
        v_src, world_pts = precomputed_reliability
        cos_angle = _compute_viewing_angle_cosine_fast(v_src, world_pts, w2c2)  # (b, 1, h, w)
        reliability_map = bilinear_splatting(
            cos_angle, mask1, trans_depth, flow12, None, is_image=False, n_views=n_views,
            depth_weight_scale=depth_weight_scale,
        )[0]  # (b_out, 1, h, w) — blended by depth-weighted splatting

    warped_depth = None
    if foreground_masking:
        warped_depth = bilinear_splatting(
            trans_depth, mask1, trans_depth, flow12, None, is_image=False, n_views=n_views,
            depth_weight_scale=depth_weight_scale,
        )[0][:, 0]

        for bi in range(b if n_views == 1 else b // n_views):
            if boundary_mask is None:
                continue
            mesh_mask = boundary_mask[bi]
            ds = 4
            verts, faces = points_to_mesh(
                cam_pts_tgt[bi], mesh_mask, resolution=(h // ds, w // ds),
            )
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                continue

            scaled_K = intrinsic2[bi:bi + 1].clone()
            rays = get_camera_rays(h, w, scaled_K)[0]   # (h, w, 3)
            origins = torch.zeros(h, w, 3, device=device, dtype=dtype)

            mesh_depth = ray_triangle_intersection(origins, rays, verts, faces, device)
            if mesh_depth is None:
                continue

            mesh_z = mesh_depth * rays[:, :, 2]
            mesh_z = F.interpolate(
                mesh_z.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear",
            ).squeeze(0).squeeze(0)

            mesh_valid = mesh_z > 0
            mesh_closer = ((mesh_z + 0.02) < warped_depth[bi]) & mesh_valid

            mask2[bi, 0] *= (~mesh_closer).float()
            fill = -1.0 if is_image else 0.0
            warped_frame[bi] = (warped_frame[bi] - fill) * (~mesh_closer).unsqueeze(0).float() + fill
            warped_depth[bi] *= (~mesh_closer).unsqueeze(0).float()

    return warped_frame, mask2, warped_depth, flow12, reliability_map
