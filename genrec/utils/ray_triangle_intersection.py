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
Warp-based GPU ray-triangle intersection for foreground occlusion masking.

Ported from GEN3C (cosmos_predict1/diffusion/inference/ray_triangle_intersection_warp.py).
Requires NVIDIA Warp (soft dependency — if unavailable, the public API returns None).
"""

import os
import torch

_warp_available = False
_warp_initialized = False
_kernels_compiled = False

try:
    import warp as wp
    _warp_available = True
except ImportError:
    pass


def _ensure_warp():
    """Lazy Warp initialization (safe for multi-GPU DDP — called once per rank)."""
    global _warp_initialized, _kernels_compiled
    if not _warp_available:
        return False
    if not _warp_initialized:
        local_rank = os.getenv("LOCAL_RANK", "0")
        print(f"[ray_triangle_intersection] Initializing Warp (local_rank={local_rank})...")
        wp.init()
        _warp_initialized = True
        print(f"[ray_triangle_intersection] Warp initialized (local_rank={local_rank})")
    return True


# ---------------------------------------------------------------------------
# Warp kernels — defined at module level so they are compiled once.
# They are only evaluated if warp was successfully imported.
# ---------------------------------------------------------------------------
if _warp_available:

    @wp.kernel
    def _ray_tri_kernel(
        ray_origins: wp.array2d(dtype=wp.float32),
        ray_directions: wp.array2d(dtype=wp.float32),
        vertices: wp.array2d(dtype=wp.float32),
        faces: wp.array2d(dtype=wp.int32),
        depth_map: wp.array(dtype=wp.float32),
        num_triangles: wp.int32,
        epsilon: wp.float32,
    ):
        """Each thread: one ray vs *all* triangles (Moller-Trumbore)."""
        ray_idx = wp.tid()

        ro = wp.vec3(ray_origins[ray_idx, 0], ray_origins[ray_idx, 1], ray_origins[ray_idx, 2])
        rd = wp.vec3(ray_directions[ray_idx, 0], ray_directions[ray_idx, 1], ray_directions[ray_idx, 2])

        min_t = wp.float32(1e10)

        for tri in range(num_triangles):
            i0 = faces[tri, 0]
            i1 = faces[tri, 1]
            i2 = faces[tri, 2]

            v0 = wp.vec3(vertices[i0, 0], vertices[i0, 1], vertices[i0, 2])
            v1 = wp.vec3(vertices[i1, 0], vertices[i1, 1], vertices[i1, 2])
            v2 = wp.vec3(vertices[i2, 0], vertices[i2, 1], vertices[i2, 2])

            edge1 = v1 - v0
            edge2 = v2 - v0
            h = wp.cross(rd, edge2)
            a = wp.dot(edge1, h)

            if wp.abs(a) < epsilon:
                continue

            f = 1.0 / a
            s = ro - v0
            u = f * wp.dot(s, h)
            if u < 0.0 or u > 1.0:
                continue

            q = wp.cross(s, edge1)
            v = f * wp.dot(rd, q)
            if v < 0.0 or (u + v) > 1.0:
                continue

            t = f * wp.dot(edge2, q)
            if t > epsilon and t < min_t:
                min_t = t

        if min_t < 1e10:
            depth_map[ray_idx] = min_t
        else:
            depth_map[ray_idx] = 0.0

    @wp.kernel
    def _ray_tri_tiled_kernel(
        ray_origins: wp.array2d(dtype=wp.float32),
        ray_directions: wp.array2d(dtype=wp.float32),
        vertices: wp.array2d(dtype=wp.float32),
        faces: wp.array2d(dtype=wp.int32),
        depth_map: wp.array(dtype=wp.float32),
        tri_start: wp.int32,
        tri_end: wp.int32,
        epsilon: wp.float32,
    ):
        """Tiled variant — processes a *range* of triangles, uses atomic_min."""
        ray_idx = wp.tid()

        ro = wp.vec3(ray_origins[ray_idx, 0], ray_origins[ray_idx, 1], ray_origins[ray_idx, 2])
        rd = wp.vec3(ray_directions[ray_idx, 0], ray_directions[ray_idx, 1], ray_directions[ray_idx, 2])

        min_t = depth_map[ray_idx]
        if min_t == 0.0:
            min_t = wp.float32(1e10)

        for tri in range(tri_start, tri_end):
            i0 = faces[tri, 0]
            i1 = faces[tri, 1]
            i2 = faces[tri, 2]

            v0 = wp.vec3(vertices[i0, 0], vertices[i0, 1], vertices[i0, 2])
            v1 = wp.vec3(vertices[i1, 0], vertices[i1, 1], vertices[i1, 2])
            v2 = wp.vec3(vertices[i2, 0], vertices[i2, 1], vertices[i2, 2])

            edge1 = v1 - v0
            edge2 = v2 - v0
            h = wp.cross(rd, edge2)
            a = wp.dot(edge1, h)

            if wp.abs(a) < epsilon:
                continue

            f = 1.0 / a
            s = ro - v0
            u = f * wp.dot(s, h)
            if u < 0.0 or u > 1.0:
                continue

            q = wp.cross(s, edge1)
            v = f * wp.dot(rd, q)
            if v < 0.0 or (u + v) > 1.0:
                continue

            t = f * wp.dot(edge2, q)
            if t > epsilon and t < min_t:
                min_t = t

        if min_t < 1e10:
            wp.atomic_min(depth_map, ray_idx, min_t)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ray_triangle_intersection_warp(
    ray_origins: torch.Tensor,       # (H, W, 3)
    ray_directions: torch.Tensor,    # (H, W, 3)
    vertices: torch.Tensor,          # (N, 3)
    faces: torch.Tensor,             # (M, 3)
    device: torch.device,
) -> "torch.Tensor | None":
    """
    Compute ray-triangle intersections using NVIDIA Warp GPU kernels.

    Returns (H, W) depth map (0 where no intersection), or ``None`` if Warp
    is not available.
    """
    if not _ensure_warp():
        return None

    H, W = ray_origins.shape[:2]
    num_rays = H * W
    num_triangles = faces.shape[0]

    ray_origins_flat = ray_origins.reshape(-1, 3).contiguous().float()
    ray_dirs_flat = ray_directions.reshape(-1, 3).contiguous().float()

    wp_ro = wp.from_torch(ray_origins_flat, dtype=wp.float32)
    wp_rd = wp.from_torch(ray_dirs_flat, dtype=wp.float32)
    wp_verts = wp.from_torch(vertices.contiguous().float(), dtype=wp.float32)
    wp_faces = wp.from_torch(faces.int().contiguous(), dtype=wp.int32)

    depth_flat = torch.zeros(num_rays, device=device, dtype=torch.float32)
    wp_depth = wp.from_torch(depth_flat, dtype=wp.float32)

    cuda_device = f"cuda:{device.index}" if device.index is not None else "cuda:0"

    if num_triangles < 10_000:
        wp.launch(
            kernel=_ray_tri_kernel,
            dim=num_rays,
            inputs=[wp_ro, wp_rd, wp_verts, wp_faces, wp_depth, num_triangles, 1e-8],
            device=cuda_device,
        )
    else:
        tile_size = 10_000
        depth_flat.fill_(float("inf"))
        for tri_start in range(0, num_triangles, tile_size):
            tri_end = min(tri_start + tile_size, num_triangles)
            wp.launch(
                kernel=_ray_tri_tiled_kernel,
                dim=num_rays,
                inputs=[wp_ro, wp_rd, wp_verts, wp_faces, wp_depth, tri_start, tri_end, 1e-8],
                device=cuda_device,
            )
        depth_flat[depth_flat == float("inf")] = 0.0

    wp.synchronize()
    return depth_flat.reshape(H, W)
