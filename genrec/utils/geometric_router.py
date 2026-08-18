"""
Geometric Router: depth-based sparse correspondence mapping for multi-view attention.

Given source depths and camera parameters, computes for each target latent pixel
the K nearest source latent tokens via depth unprojection/reprojection. This
enables geometry-guided sparse attention (O(N*K)) instead of all-to-all (O(N^2)).

Optimized: batched latent-resolution reprojection, vectorized neighborhood
expansion, and sort-based K-nearest selection (no per-cell Python loops).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from genrec.utils.forward_warp import inverse_with_conversion


@torch.no_grad()
def replace_with_3d_distances(
    geo_index_map: 'GeometricIndexMap',
    source_world_pos: torch.Tensor,   # (B, T_src, 3, H, W) normalized [-1,1]
    target_world_pos: torch.Tensor,   # (B, T_out, 3, H, W) normalized [-1,1]
    T_out: int,
    vae_scale_factor: int = 1,
) -> 'GeometricIndexMap':
    """Replace 2D projected distances with 3D world-space Euclidean distances.

    For each (target_pixel, source_pixel) pair in the K-NN index map,
    compute the 3D distance between the corresponding world-space positions
    from full_coord_maps. Replaces weights with normalized inverse 3D distance.

    Args:
        geo_index_map: Existing index map with 2D distances.
        source_world_pos: (B, T_src, 3, H, W) world coords of source views.
        target_world_pos: (B, T_out, 3, H, W) world coords of target views.
        T_out: Number of target views.
        vae_scale_factor: Spatial downscale factor (1 for pixel, 8 for latent).

    Returns:
        New GeometricIndexMap with 3D distance-based weights.
    """
    B = source_world_pos.shape[0]
    T_src = source_world_pos.shape[1]
    H_img, W_img = source_world_pos.shape[3], source_world_pos.shape[4]
    H = H_img // vae_scale_factor
    W = W_img // vae_scale_factor
    N = H * W
    K = geo_index_map.indices.shape[-1]

    # Downsample world positions to match index map resolution
    if vae_scale_factor > 1:
        src_pos = torch.nn.functional.avg_pool2d(
            source_world_pos.reshape(B * T_src, 3, H_img, W_img),
            kernel_size=vae_scale_factor, stride=vae_scale_factor,
        ).reshape(B, T_src, 3, H, W)
        tgt_pos = torch.nn.functional.avg_pool2d(
            target_world_pos.reshape(B * T_out, 3, H_img, W_img),
            kernel_size=vae_scale_factor, stride=vae_scale_factor,
        ).reshape(B, T_out, 3, H, W)
    else:
        src_pos = source_world_pos
        tgt_pos = target_world_pos

    # Flatten source: (B, T_src*H*W, 3)
    src_flat = src_pos.permute(0, 1, 3, 4, 2).reshape(B, T_src * N, 3)

    # Flatten target: (B, T_out, H*W, 3)
    tgt_flat = tgt_pos.permute(0, 1, 3, 4, 2).reshape(B, T_out, N, 3)

    indices = geo_index_map.indices  # (B, T_out, N, K)
    valid = geo_index_map.valid_mask  # (B, T_out, N, K)

    new_distances = torch.zeros_like(geo_index_map.distances)
    new_weights = torch.zeros_like(geo_index_map.weights)

    for b in range(B):
        for j in range(T_out):
            idx = indices[b, j]  # (N, K) into T_src*N
            v = valid[b, j]     # (N, K)

            # Gather source 3D positions: (N, K, 3)
            idx_clamped = idx.clamp(0, src_flat.shape[1] - 1)
            src_gathered = src_flat[b][idx_clamped.reshape(-1)].reshape(N, K, 3)

            # Target 3D positions: (N, 1, 3) -> broadcast
            tgt_pts = tgt_flat[b, j].unsqueeze(1)  # (N, 1, 3)

            # 3D Euclidean distance
            dist_3d = (src_gathered - tgt_pts).norm(dim=-1)  # (N, K)
            dist_3d = dist_3d * v.float()  # zero out invalid

            new_distances[b, j] = dist_3d

            # Inverse distance weights, normalized per pixel
            inv_dist = 1.0 / (dist_3d + 1e-6)
            inv_dist = inv_dist * v.float()
            row_sum = inv_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            new_weights[b, j] = inv_dist / row_sum

    return GeometricIndexMap(
        indices=geo_index_map.indices,
        distances=new_distances,
        weights=new_weights,
        valid_mask=geo_index_map.valid_mask,
        proj_xy=geo_index_map.proj_xy,
        proj_depth=geo_index_map.proj_depth,
    )


@dataclass
class GeometricIndexMap:
    """Sparse correspondence map from target to source latent tokens."""
    indices: torch.Tensor       # (T_out, H_lat*W_lat, K) — flat index into (T_in*H_lat*W_lat)
    distances: torch.Tensor     # (T_out, H_lat*W_lat, K) — 2D pixel distances (latent coords)
    weights: torch.Tensor       # (T_out, H_lat*W_lat, K) — normalized inverse-distance weights
    valid_mask: torch.Tensor    # (T_out, H_lat*W_lat, K) — bool, True if slot filled
    proj_xy: Optional[torch.Tensor] = None       # (T_out, T_src*H_img*W_img, 2) — only for visualization
    proj_depth: Optional[torch.Tensor] = None    # (T_out, T_src*H_img*W_img) — only for visualization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reproject_latent_batched(
    source_depth: torch.Tensor,        # (N_pairs, H_img, W_img)
    source_w2c: torch.Tensor,          # (N_pairs, 4, 4)
    target_w2c: torch.Tensor,          # (N_pairs, 4, 4)
    source_intrinsics: torch.Tensor,   # (N_pairs, 3, 3)
    target_intrinsics: torch.Tensor,   # (N_pairs, 3, 3)
    vae_scale_factor: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reproject source depth at LATENT resolution into target cameras.

    Center-pixel samples depth, builds an image-space pixel grid at those
    centers, then does unproject -> transform -> reproject entirely at
    (H_lat, W_lat) resolution (64x fewer pixels than full-res).

    Returns:
        xy_lat: (N_pairs, H_lat, W_lat, 2) projected positions in latent coords.
        z:      (N_pairs, H_lat, W_lat)     depth in target camera.
    """
    N = source_depth.shape[0]
    H_img, W_img = source_depth.shape[1], source_depth.shape[2]
    H_lat = H_img // vae_scale_factor
    W_lat = W_img // vae_scale_factor
    device = source_depth.device
    dtype = source_depth.dtype
    center_offset = vae_scale_factor // 2  # 4

    # Latent cell centers in image pixel coordinates
    sy = torch.arange(H_lat, device=device) * vae_scale_factor + center_offset
    sx = torch.arange(W_lat, device=device) * vae_scale_factor + center_offset
    sy = sy.clamp(0, H_img - 1)
    sx = sx.clamp(0, W_img - 1)

    # Sample depth at center pixels: (N, H_lat, W_lat)
    depth_lat = source_depth[:, sy[:, None], sx[None, :]]

    # Image-space pixel grid at center positions
    x_img = sx[None, :].expand(H_lat, W_lat).to(dtype)   # (H_lat, W_lat)
    y_img = sy[:, None].expand(H_lat, W_lat).to(dtype)   # (H_lat, W_lat)
    ones = torch.ones(H_lat, W_lat, device=device, dtype=dtype)
    # (1, H_lat, W_lat, 3, 1)
    pos_homo = torch.stack([x_img, y_img, ones], dim=2)[None, :, :, :, None]

    # Camera matrices
    K1_inv = inverse_with_conversion(source_intrinsics)[:, None, None]  # (N,1,1,3,3)
    K2 = target_intrinsics[:, None, None]                                # (N,1,1,3,3)
    transformation = torch.bmm(
        target_w2c, inverse_with_conversion(source_w2c)
    )  # (N, 4, 4)
    trans_4d = transformation[:, None, None]                             # (N,1,1,4,4)

    # Unproject -> world
    unnorm_pos = torch.matmul(K1_inv, pos_homo)                          # (N,H,W,3,1)
    world_pts = depth_lat[:, :, :, None, None] * unnorm_pos              # (N,H,W,3,1)

    ones_4d = torch.ones(N, H_lat, W_lat, 1, 1, device=device, dtype=dtype)
    world_pts_h = torch.cat([world_pts, ones_4d], dim=3)                 # (N,H,W,4,1)
    trans_world = torch.matmul(trans_4d, world_pts_h)[:, :, :, :3]       # (N,H,W,3,1)
    projected = torch.matmul(K2, trans_world)                            # (N,H,W,3,1)

    z = projected[:, :, :, 2, 0]                                         # (N,H,W)
    xy_img = projected[:, :, :, :2, 0] / z[:, :, :, None].clamp(min=1e-7)

    # Image coords -> latent coords
    xy_lat = xy_img.clone()
    xy_lat[..., 0] *= (W_lat / W_img)
    xy_lat[..., 1] *= (H_lat / H_img)

    return xy_lat, z


def _expand_neighborhoods_vectorized(
    valid_xy: torch.Tensor,     # (N_valid, 2)  projected positions in latent coords
    valid_z: torch.Tensor,      # (N_valid,)
    valid_idx: torch.Tensor,    # (N_valid,)    source flat indices
    search_radius: float,
    H_lat: int,
    W_lat: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand each valid source token to all neighbouring target cells within
    *search_radius*.  Fully vectorized — no Python offset loop.

    Returns (all filtered by in-bounds & radius):
        cell_flat: (M,) target cell flat index
        src_idx:   (M,) source flat index
        z:         (M,) depth in target camera
        dist:      (M,) 2-D distance to cell center
    """
    device = valid_xy.device
    r_int = int(search_radius + 0.5)

    # Offset grid, e.g. 3x3 for radius 1.5
    dy_range = torch.arange(-r_int, r_int + 1, device=device)
    dx_range = torch.arange(-r_int, r_int + 1, device=device)
    oy, ox = torch.meshgrid(dy_range, dx_range, indexing='ij')
    offsets_y = oy.reshape(-1)   # (n_off,)
    offsets_x = ox.reshape(-1)   # (n_off,)
    n_off = offsets_y.shape[0]
    N_valid = valid_xy.shape[0]

    # Base cell per source token
    base_cx = valid_xy[:, 0].long()  # (N_valid,)
    base_cy = valid_xy[:, 1].long()

    # Broadcast: (N_valid, n_off)
    cy_all = base_cy[:, None] + offsets_y[None, :]
    cx_all = base_cx[:, None] + offsets_x[None, :]

    # Cell centers
    tcx = cx_all.float() + 0.5
    tcy = cy_all.float() + 0.5

    # Distance
    d = torch.sqrt(
        (valid_xy[:, 0:1] - tcx) ** 2 +
        (valid_xy[:, 1:2] - tcy) ** 2
    )  # (N_valid, n_off)

    # Keep mask
    in_bounds = (cx_all >= 0) & (cx_all < W_lat) & (cy_all >= 0) & (cy_all < H_lat)
    keep = in_bounds & (d < search_radius)

    # Flatten & filter
    keep_flat = keep.reshape(-1)
    keep_indices = keep_flat.nonzero(as_tuple=True)[0]

    cell_flat = (cy_all * W_lat + cx_all).reshape(-1)[keep_indices]
    src_idx = valid_idx[:, None].expand(-1, n_off).reshape(-1)[keep_indices]
    z_out = valid_z[:, None].expand(-1, n_off).reshape(-1)[keep_indices]
    dist = d.reshape(-1)[keep_indices]

    return cell_flat, src_idx, z_out, dist


def _vectorized_knearest(
    cell_flat: torch.Tensor,   # (M,) target cell index
    src_idx: torch.Tensor,     # (M,) source flat index
    z_vals: torch.Tensor,      # (M,) depth in target camera
    dist: torch.Tensor,        # (M,) 2-D distance to cell center
    N_tgt: int,
    K: int,
    occlusion_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized per-cell z-buffer + dedup + K-nearest selection.

    Replaces the Python ``for ci, cell in enumerate(unique_cells)`` loop with
    scatter-reduce and sort-based binning.

    Returns:
        indices:   (N_tgt, K)
        distances: (N_tgt, K)
        weights:   (N_tgt, K)
        valid:     (N_tgt, K)
    """
    # --- Phase A: Z-buffer filtering ---
    # Ensure consistent dtypes (inputs may arrive as bf16 from autocast)
    z_vals = z_vals.to(dtype)
    dist = dist.to(dtype)
    z_min_buf = torch.full((N_tgt,), float('inf'), device=device, dtype=dtype)
    z_min_buf.scatter_reduce_(0, cell_flat, z_vals, reduce='amin')

    z_threshold = z_min_buf[cell_flat] * (1.0 + occlusion_eps)
    z_ok = z_vals <= z_threshold
    keep = z_ok.nonzero(as_tuple=True)[0]

    if keep.numel() == 0:
        return (torch.zeros(N_tgt, K, dtype=torch.long, device=device),
                torch.zeros(N_tgt, K, dtype=dtype, device=device),
                torch.zeros(N_tgt, K, dtype=dtype, device=device),
                torch.zeros(N_tgt, K, dtype=torch.bool, device=device))

    cell_flat = cell_flat[keep]
    src_idx = src_idx[keep]
    dist = dist[keep]

    # --- Phase B: Deduplication (same src in same cell → keep min dist) ---
    N_src_max = int(src_idx.max().item()) + 1
    composite_key = cell_flat * N_src_max + src_idx
    unique_keys, inverse = composite_key.unique(return_inverse=True)

    best_dist = torch.full((unique_keys.numel(),), float('inf'), device=device, dtype=dtype)
    best_dist.scatter_reduce_(0, inverse, dist, reduce='amin')

    dedup_cell = unique_keys // N_src_max
    dedup_src = unique_keys % N_src_max
    dedup_dist = best_dist
    M = unique_keys.numel()

    # --- Phase C: K-nearest via two-pass stable sort ---
    # Sort by distance (stable), then by cell (stable) → lexicographic (cell, dist)
    order1 = dedup_dist.argsort(stable=True)
    order2 = dedup_cell[order1].argsort(stable=True)
    sorted_order = order1[order2]

    s_cell = dedup_cell[sorted_order]
    s_src = dedup_src[sorted_order]
    s_dist = dedup_dist[sorted_order]

    # Compute within-cell rank via boundary detection
    boundaries = torch.ones(M, dtype=torch.bool, device=device)
    boundaries[1:] = s_cell[1:] != s_cell[:-1]
    boundary_pos = boundaries.nonzero(as_tuple=True)[0]           # starts of each group
    group_ids = boundaries.long().cumsum(0) - 1                    # 0,0,0,1,1,2,...
    starts = boundary_pos[group_ids]                               # start pos per element
    within_rank = torch.arange(M, device=device) - starts         # rank within cell

    # Keep first K per cell
    topk_mask = within_rank < K
    final_cell = s_cell[topk_mask]
    final_src = s_src[topk_mask]
    final_dist = s_dist[topk_mask]
    final_rank = within_rank[topk_mask]

    # --- Phase D: Scatter into output tensors ---
    out_indices = torch.zeros(N_tgt, K, dtype=torch.long, device=device)
    out_distances = torch.zeros(N_tgt, K, dtype=dtype, device=device)
    out_valid = torch.zeros(N_tgt, K, dtype=torch.bool, device=device)

    flat_pos = final_cell * K + final_rank
    out_indices.view(-1).scatter_(0, flat_pos, final_src)
    out_distances.view(-1).scatter_(0, flat_pos, final_dist)
    out_valid.view(-1).scatter_(0, flat_pos, torch.ones_like(flat_pos, dtype=torch.bool))

    # Weights: normalized inverse distance (zero for invalid slots)
    inv_dist = 1.0 / (out_distances + 1e-6)
    inv_dist = inv_dist * out_valid.float()
    row_sum = inv_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    out_weights = inv_dist / row_sum

    return out_indices, out_distances, out_weights, out_valid


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_geometric_index_map(
    source_depth: torch.Tensor,        # (T_in, H_img, W_img)
    source_w2c: torch.Tensor,          # (T_in, 4, 4)
    source_intrinsics: torch.Tensor,   # (T_in, 3, 3)
    target_w2c: torch.Tensor,          # (T_out, 4, 4)
    target_intrinsics: torch.Tensor,   # (T_out, 3, 3)
    K: int = 8,
    vae_scale_factor: int = 8,
    occlusion_eps: float = 0.02,
    search_radius: float = 1.5,
    store_projections: bool = False,
) -> GeometricIndexMap:
    """
    Compute sparse geometric correspondence map from target to source latent tokens.

    For each target latent pixel, finds the K nearest source latent tokens by:
    1. Unprojecting source depth pixels to 3D, reprojecting into target view
    2. Radius search + z-buffer occlusion filtering + K-nearest selection

    All reprojection is done at latent resolution (center-pixel sampling) and
    batched across all (target, source) view pairs for efficiency.

    Args:
        source_depth: Depth maps at original image resolution.
        source_w2c: World-to-camera extrinsics for source views.
        source_intrinsics: Camera intrinsics at original image resolution.
        target_w2c: World-to-camera extrinsics for target views.
        target_intrinsics: Camera intrinsics at original image resolution.
        K: Number of nearest neighbors per target pixel.
        vae_scale_factor: Latent downscale factor (typically 8).
        occlusion_eps: Relative z-buffer tolerance for occlusion filtering.
        search_radius: Search radius in latent pixels around target pixel center.
        store_projections: If True, compute and store full-res proj_xy/proj_depth
            (needed only for visualization, not training).
    """
    T_in = source_depth.shape[0]
    H_img, W_img = source_depth.shape[1], source_depth.shape[2]
    T_out = target_w2c.shape[0]
    device = source_depth.device
    dtype = torch.float32  # geometry ops need float32 precision (inputs may be bf16)

    H_lat = H_img // vae_scale_factor
    W_lat = W_img // vae_scale_factor
    N_tgt = H_lat * W_lat

    # --- Phase 1: Batched latent-resolution reprojection ---
    # Build all T_out * T_in view pairs
    N_pairs = T_out * T_in
    pair_depth = source_depth.float()[None, :].expand(T_out, -1, -1, -1).reshape(N_pairs, H_img, W_img)
    pair_src_w2c = source_w2c.float()[None, :].expand(T_out, -1, -1, -1).reshape(N_pairs, 4, 4)
    pair_src_K = source_intrinsics.float()[None, :].expand(T_out, -1, -1, -1).reshape(N_pairs, 3, 3)
    pair_tgt_w2c = target_w2c.float()[:, None].expand(-1, T_in, -1, -1).reshape(N_pairs, 4, 4)
    pair_tgt_K = target_intrinsics.float()[:, None].expand(-1, T_in, -1, -1).reshape(N_pairs, 3, 3)

    xy_lat, z = _reproject_latent_batched(
        pair_depth, pair_src_w2c, pair_tgt_w2c, pair_src_K, pair_tgt_K, vae_scale_factor,
    )  # (N_pairs, H_lat, W_lat, 2), (N_pairs, H_lat, W_lat)

    # Reshape to (T_out, T_in, H_lat, W_lat, ...)
    xy_lat = xy_lat.reshape(T_out, T_in, H_lat, W_lat, 2)
    z = z.reshape(T_out, T_in, H_lat, W_lat)

    # Optional full-res projections for visualization
    proj_xy = None
    proj_depth = None
    if store_projections:
        from genrec.utils.forward_warp import compute_transformed_points
        proj_xy = torch.zeros(T_out, T_in * H_img * W_img, 2, dtype=dtype, device=device)
        proj_depth = torch.zeros(T_out, T_in * H_img * W_img, dtype=dtype, device=device)
        for j in range(T_out):
            for i in range(T_in):
                depth_i = source_depth[i:i+1].unsqueeze(1)
                projected = compute_transformed_points(
                    depth_i, source_w2c[i:i+1], target_w2c[j:j+1],
                    source_intrinsics[i:i+1], is_depth=True,
                    intrinsic2=target_intrinsics[j:j+1],
                )
                pz = projected[0, :, :, 2, 0]
                pxy = projected[0, :, :, :2, 0] / pz.unsqueeze(-1).clamp(min=1e-7)
                s = i * H_img * W_img
                e = s + H_img * W_img
                proj_xy[j, s:e] = pxy.reshape(-1, 2)
                proj_depth[j, s:e] = pz.reshape(-1)

    # --- Phase 2: Per target view — vectorized KNN ---
    all_indices = torch.zeros(T_out, N_tgt, K, dtype=torch.long, device=device)
    all_distances = torch.zeros(T_out, N_tgt, K, dtype=dtype, device=device)
    all_weights = torch.zeros(T_out, N_tgt, K, dtype=dtype, device=device)
    all_valid = torch.zeros(T_out, N_tgt, K, dtype=torch.bool, device=device)

    for j in range(T_out):
        # Flatten all source tokens for this target view
        all_src_xy = xy_lat[j].reshape(-1, 2)       # (T_in*H_lat*W_lat, 2)
        all_src_z = z[j].reshape(-1)                  # (T_in*H_lat*W_lat,)
        all_src_idx = torch.arange(T_in * N_tgt, device=device)

        # Validity filter
        valid = (all_src_z > 0) & \
                (all_src_xy[:, 0] >= 0) & (all_src_xy[:, 0] < W_lat) & \
                (all_src_xy[:, 1] >= 0) & (all_src_xy[:, 1] < H_lat)
        valid_mask = valid.nonzero(as_tuple=True)[0]
        if valid_mask.numel() == 0:
            continue

        valid_xy = all_src_xy[valid_mask]
        valid_z = all_src_z[valid_mask]
        valid_idx = all_src_idx[valid_mask]

        # Vectorized neighborhood expansion
        cell_flat, src_idx, z_exp, dist = _expand_neighborhoods_vectorized(
            valid_xy, valid_z, valid_idx, search_radius, H_lat, W_lat,
        )
        if cell_flat.numel() == 0:
            continue

        # Vectorized K-nearest selection
        idx_j, dist_j, w_j, v_j = _vectorized_knearest(
            cell_flat, src_idx, z_exp, dist, N_tgt, K, occlusion_eps, device, dtype,
        )
        all_indices[j] = idx_j
        all_distances[j] = dist_j
        all_weights[j] = w_j
        all_valid[j] = v_j

    return GeometricIndexMap(
        indices=all_indices,
        distances=all_distances,
        weights=all_weights,
        valid_mask=all_valid,
        proj_xy=proj_xy,
        proj_depth=proj_depth,
    )


def compute_batched_geometric_index_map(
    source_depth: torch.Tensor,        # (B, T_in, H_img, W_img)
    source_w2c: torch.Tensor,          # (B, T_in, 4, 4)
    source_intrinsics: torch.Tensor,   # (B, T_in, 3, 3)
    target_w2c: torch.Tensor,          # (B, T_out, 4, 4)
    target_intrinsics: torch.Tensor,   # (B, T_out, 3, 3)
    K: int = 8,
    vae_scale_factor: int = 8,
    occlusion_eps: float = 0.02,
    search_radius: float = 1.5,
    store_projections: bool = False,
) -> GeometricIndexMap:
    """
    Batched wrapper for compute_geometric_index_map.

    Returns GeometricIndexMap with indices shape (B, T_out, H_lat*W_lat, K), etc.
    """
    B = source_depth.shape[0]

    results = []
    for b in range(B):
        result = compute_geometric_index_map(
            source_depth=source_depth[b],
            source_w2c=source_w2c[b],
            source_intrinsics=source_intrinsics[b],
            target_w2c=target_w2c[b],
            target_intrinsics=target_intrinsics[b],
            K=K,
            vae_scale_factor=vae_scale_factor,
            occlusion_eps=occlusion_eps,
            search_radius=search_radius,
            store_projections=store_projections,
        )
        results.append(result)

    return GeometricIndexMap(
        indices=torch.stack([r.indices for r in results], dim=0),
        distances=torch.stack([r.distances for r in results], dim=0),
        weights=torch.stack([r.weights for r in results], dim=0),
        valid_mask=torch.stack([r.valid_mask for r in results], dim=0),
        proj_xy=torch.stack([r.proj_xy for r in results], dim=0) if store_projections else None,
        proj_depth=torch.stack([r.proj_depth for r in results], dim=0) if store_projections else None,
    )


@torch.no_grad()
def compute_output_to_output_geometric_index_map(
    output_depth: torch.Tensor,        # (T_out, H_img, W_img)
    output_w2c: torch.Tensor,          # (T_out, 4, 4)
    output_intrinsics: torch.Tensor,   # (T_out, 3, 3)
    K: int = 8,
    vae_scale_factor: int = 8,
    occlusion_eps: float = 0.02,
    search_radius: float = 1.5,
    store_projections: bool = False,
) -> GeometricIndexMap:
    """
    Compute sparse geometric correspondence map between output views.

    Same algorithm as compute_geometric_index_map but source = target = output
    views. A view does NOT attend to itself (skip i == j). Index space is flat
    into T_out * H_lat * W_lat.
    """
    T_out = output_depth.shape[0]
    H_img, W_img = output_depth.shape[1], output_depth.shape[2]
    device = output_depth.device
    dtype = output_depth.dtype

    H_lat = H_img // vae_scale_factor
    W_lat = W_img // vae_scale_factor
    N_tgt = H_lat * W_lat

    # --- Phase 1: Batched reprojection for all (j, i) pairs where i != j ---
    # Build pair indices, excluding self-pairs
    pair_j = []
    pair_i = []
    for j in range(T_out):
        for i in range(T_out):
            if i != j:
                pair_j.append(j)
                pair_i.append(i)

    if not pair_j:
        # T_out <= 1: no cross-view pairs
        return GeometricIndexMap(
            indices=torch.zeros(T_out, N_tgt, K, dtype=torch.long, device=device),
            distances=torch.zeros(T_out, N_tgt, K, dtype=dtype, device=device),
            weights=torch.zeros(T_out, N_tgt, K, dtype=dtype, device=device),
            valid_mask=torch.zeros(T_out, N_tgt, K, dtype=torch.bool, device=device),
            proj_xy=None,
            proj_depth=None,
        )

    pair_j_t = torch.tensor(pair_j, device=device)
    pair_i_t = torch.tensor(pair_i, device=device)
    N_pairs = pair_j_t.shape[0]

    xy_lat, z = _reproject_latent_batched(
        output_depth[pair_i_t],         # source depth
        output_w2c[pair_i_t],           # source w2c
        output_w2c[pair_j_t],           # target w2c
        output_intrinsics[pair_i_t],    # source K
        output_intrinsics[pair_j_t],    # target K
        vae_scale_factor,
    )  # (N_pairs, H_lat, W_lat, 2), (N_pairs, H_lat, W_lat)

    # Optional full-res projections
    proj_xy = None
    proj_depth = None
    if store_projections:
        from genrec.utils.forward_warp import compute_transformed_points
        proj_xy = torch.zeros(T_out, T_out * H_img * W_img, 2, dtype=dtype, device=device)
        proj_depth_out = torch.zeros(T_out, T_out * H_img * W_img, dtype=dtype, device=device)
        for p in range(N_pairs):
            j, i = pair_j[p], pair_i[p]
            depth_i = output_depth[i:i+1].unsqueeze(1)
            projected = compute_transformed_points(
                depth_i, output_w2c[i:i+1], output_w2c[j:j+1],
                output_intrinsics[i:i+1], is_depth=True,
                intrinsic2=output_intrinsics[j:j+1],
            )
            pz = projected[0, :, :, 2, 0]
            pxy = projected[0, :, :, :2, 0] / pz.unsqueeze(-1).clamp(min=1e-7)
            s = i * H_img * W_img
            e = s + H_img * W_img
            proj_xy[j, s:e] = pxy.reshape(-1, 2)
            proj_depth_out[j, s:e] = pz.reshape(-1)
        proj_depth = proj_depth_out

    # --- Phase 2: Per target view — gather pairs & vectorized KNN ---
    all_indices = torch.zeros(T_out, N_tgt, K, dtype=torch.long, device=device)
    all_distances = torch.zeros(T_out, N_tgt, K, dtype=dtype, device=device)
    all_weights = torch.zeros(T_out, N_tgt, K, dtype=dtype, device=device)
    all_valid = torch.zeros(T_out, N_tgt, K, dtype=torch.bool, device=device)

    # Group pairs by target view
    for j in range(T_out):
        # Gather all source views for target j
        pair_mask = (pair_j_t == j)
        pair_positions = pair_mask.nonzero(as_tuple=True)[0]
        source_views = pair_i_t[pair_positions]  # which source views
        n_src = source_views.shape[0]

        if n_src == 0:
            continue

        # Collect projected positions and z for all sources of this target
        src_xy_parts = []
        src_z_parts = []
        src_idx_parts = []

        for si, p_idx in enumerate(pair_positions):
            src_view = source_views[si].item()
            xy_p = xy_lat[p_idx].reshape(-1, 2)   # (H_lat*W_lat, 2)
            z_p = z[p_idx].reshape(-1)              # (H_lat*W_lat,)
            # Flat index into T_out*H_lat*W_lat using original source view index
            base_idx = src_view * N_tgt
            flat_idx = base_idx + torch.arange(N_tgt, device=device)
            src_xy_parts.append(xy_p)
            src_z_parts.append(z_p)
            src_idx_parts.append(flat_idx)

        all_src_xy = torch.cat(src_xy_parts, dim=0)
        all_src_z = torch.cat(src_z_parts, dim=0)
        all_src_idx = torch.cat(src_idx_parts, dim=0)

        # Validity filter
        valid = (all_src_z > 0) & \
                (all_src_xy[:, 0] >= 0) & (all_src_xy[:, 0] < W_lat) & \
                (all_src_xy[:, 1] >= 0) & (all_src_xy[:, 1] < H_lat)
        valid_mask = valid.nonzero(as_tuple=True)[0]
        if valid_mask.numel() == 0:
            continue

        valid_xy_v = all_src_xy[valid_mask]
        valid_z_v = all_src_z[valid_mask]
        valid_idx_v = all_src_idx[valid_mask]

        # Vectorized neighborhood expansion
        cell_flat, src_idx_exp, z_exp, dist = _expand_neighborhoods_vectorized(
            valid_xy_v, valid_z_v, valid_idx_v, search_radius, H_lat, W_lat,
        )
        if cell_flat.numel() == 0:
            continue

        # Vectorized K-nearest selection
        idx_j, dist_j, w_j, v_j = _vectorized_knearest(
            cell_flat, src_idx_exp, z_exp, dist, N_tgt, K, occlusion_eps, device, dtype,
        )
        all_indices[j] = idx_j
        all_distances[j] = dist_j
        all_weights[j] = w_j
        all_valid[j] = v_j

    return GeometricIndexMap(
        indices=all_indices,
        distances=all_distances,
        weights=all_weights,
        valid_mask=all_valid,
        proj_xy=proj_xy,
        proj_depth=proj_depth,
    )


def compute_batched_output_to_output_index_map(
    output_depth: torch.Tensor,        # (B, T_out, H_img, W_img)
    output_w2c: torch.Tensor,          # (B, T_out, 4, 4)
    output_intrinsics: torch.Tensor,   # (B, T_out, 3, 3)
    K: int = 8,
    vae_scale_factor: int = 8,
    occlusion_eps: float = 0.02,
    search_radius: float = 1.5,
    store_projections: bool = False,
) -> GeometricIndexMap:
    """
    Batched wrapper for compute_output_to_output_geometric_index_map.

    Returns GeometricIndexMap with indices shape (B, T_out, H_lat*W_lat, K)
    into T_out*H_lat*W_lat.
    """
    B = output_depth.shape[0]

    results = []
    for b in range(B):
        result = compute_output_to_output_geometric_index_map(
            output_depth=output_depth[b],
            output_w2c=output_w2c[b],
            output_intrinsics=output_intrinsics[b],
            K=K,
            vae_scale_factor=vae_scale_factor,
            occlusion_eps=occlusion_eps,
            search_radius=search_radius,
            store_projections=store_projections,
        )
        results.append(result)

    return GeometricIndexMap(
        indices=torch.stack([r.indices for r in results], dim=0),
        distances=torch.stack([r.distances for r in results], dim=0),
        weights=torch.stack([r.weights for r in results], dim=0),
        valid_mask=torch.stack([r.valid_mask for r in results], dim=0),
        proj_xy=torch.stack([r.proj_xy for r in results], dim=0) if store_projections else None,
        proj_depth=torch.stack([r.proj_depth for r in results], dim=0) if store_projections else None,
    )
