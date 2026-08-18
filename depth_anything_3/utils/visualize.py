# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib
import numpy as np
import torch
from einops import rearrange
import plotly.graph_objects as go

from depth_anything_3.utils.logger import logger


def visualize_depth(
    depth: np.ndarray,
    depth_min=None,
    depth_max=None,
    percentile=2,
    ret_minmax=False,
    ret_type=np.uint8,
    cmap="Spectral",
):
    """
    Visualize a depth map using a colormap.

    Args:
        depth: Input depth map array
        depth_min: Minimum depth value for normalization. If None, uses percentile
        depth_max: Maximum depth value for normalization. If None, uses percentile
        percentile: Percentile for min/max computation if not provided
        ret_minmax: Whether to return min/max depth values
        ret_type: Return array type (uint8 or float)
        cmap: Matplotlib colormap name to use

    Returns:
        Colored depth visualization as numpy array
        If ret_minmax=True, also returns depth_min and depth_max
    """
    depth = depth.copy()
    depth.copy()
    valid_mask = depth > 0
    depth[valid_mask] = 1 / depth[valid_mask]
    if depth_min is None:
        if valid_mask.sum() <= 10:
            depth_min = 0
        else:
            depth_min = np.percentile(depth[valid_mask], percentile)
    if depth_max is None:
        if valid_mask.sum() <= 10:
            depth_max = 0
        else:
            depth_max = np.percentile(depth[valid_mask], 100 - percentile)
    if depth_min == depth_max:
        depth_min = depth_min - 1e-6
        depth_max = depth_max + 1e-6
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - depth_min) / (depth_max - depth_min)).clip(0, 1)
    depth = 1 - depth
    img_colored_np = cm(depth[None], bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    if ret_type == np.uint8:
        img_colored_np = (img_colored_np[0] * 255.0).astype(np.uint8)
    elif ret_type == np.float32 or ret_type == np.float64:
        img_colored_np = img_colored_np[0]
    else:
        raise ValueError(f"Invalid return type: {ret_type}")
    if ret_minmax:
        return img_colored_np, depth_min, depth_max
    else:
        return img_colored_np


# GS video rendering visulization function, since it operates in Tensor space...


def vis_depth_map_tensor(
    result: torch.Tensor,  # "*batch height width"
    color_map: str = "Spectral",
) -> torch.Tensor:  # "*batch 3 height with"
    """
    Color-map the depth map.
    """
    far = result.reshape(-1)[:16_000_000].float().quantile(0.99).log().to(result)
    try:
        near = result[result > 0][:16_000_000].float().quantile(0.01).log().to(result)
    except (RuntimeError, ValueError) as e:
        logger.error(f"No valid depth values found. Reason: {e}")
        near = torch.zeros_like(far)
    result = result.log()
    result = (result - near) / (far - near)
    return apply_color_map_to_image(result, color_map)


def apply_color_map(
    x: torch.Tensor,  # " *batch"
    color_map: str = "inferno",
) -> torch.Tensor:  # "*batch 3"
    cmap = matplotlib.cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.float().detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: torch.Tensor,  # "*batch height width"
    color_map: str = "inferno",
) -> torch.Tensor:  # "*batch 3 height with"
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")

def get_camera_frustum(c2w, frustum_length=0.3, color='blue'):
    """
    Generates the lines for a camera pyramid (frustum).
    """
    # Define the 5 points of the pyramid in Camera Space
    # Origin, TopLeft, TopRight, BottomRight, BottomLeft
    # Assuming standard OpenCV (Z+ forward) or OpenGL (Z- forward) - logic works for wireframe
    # Adjust z-sign below if cameras point the wrong way
    z = frustum_length
    w, h = frustum_length * 0.5, frustum_length * 0.35
    
    # Points in camera coordinate system
    # (0,0,0) is the camera center
    points_cam = np.array([
        [0, 0, 0],   # Center
        [-w, -h, z], # TL
        [ w, -h, z], # TR
        [ w,  h, z], # BR
        [-w,  h, z]  # BL
    ]).T # Shape (3, 5)

    # Convert to World Space: R * points + T
    # Or simply: C2W @ [x,y,z,1]
    points_cam_hom = np.vstack((points_cam, np.ones((1, 5))))
    points_world = (c2w @ points_cam_hom)[:3, :]

    # Create the lines for the pyramid
    # Connections: Center->TL->TR->Center->BL->BR->Center, then TR->BR, TL->BL
    x = points_world[0]
    y = points_world[1]
    z = points_world[2]
    
    # Order to draw lines efficiently
    # Center -> TL -> TR -> Center -> BL -> BR -> Center -> TL (close loop) ...
    # Easier way: Manually define segments or use mesh3d. 
    # Here is a simple wireframe sequence:
    
    # Lines connecting center to image plane corners
    lines_x = [x[0], x[1], None, x[0], x[2], None, x[0], x[3], None, x[0], x[4], None]
    lines_y = [y[0], y[1], None, y[0], y[2], None, y[0], y[3], None, y[0], y[4], None]
    lines_z = [z[0], z[1], None, z[0], z[2], None, z[0], z[3], None, z[0], z[4], None]
    
    # Image plane rectangle (TL-TR-BR-BL-TL)
    lines_x.extend([x[1], x[2], x[3], x[4], x[1], None])
    lines_y.extend([y[1], y[2], y[3], y[4], y[1], None])
    lines_z.extend([z[1], z[2], z[3], z[4], z[1], None])

    return go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color=color, width=2),
        showlegend=False
    )

def visualize_cameras_plotly(poses1, poses2):
    """
    poses1: (N, 4, 4) numpy array
    poses2: (N, 4, 4) numpy array
    """
    data = []
    
    # Add first set of cameras (Red)
    # Subsample with [::5] if you have too many cameras
    for c2w in poses1[::1]: 
        data.append(get_camera_frustum(c2w, color='red'))
        
    # Add second set of cameras (Blue)
    for c2w in poses2[::1]:
        data.append(get_camera_frustum(c2w, color='blue'))

    # Add a dummy trace for legend
    data.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='lines', line=dict(color='red', width=4), name='Set 1'))
    data.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='lines', line=dict(color='blue', width=4), name='Set 2'))

    # Set up layout
    layout = go.Layout(
        title="Camera Extrinsics Visualization",
        scene=dict(
            aspectmode='data', # Keeps proportions correct
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
        )
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.show()