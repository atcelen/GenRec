import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import glob
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
from depth_anything_3.api import DepthAnything3
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
from depth_anything_3.utils.visualize import vis_depth_map_tensor, visualize_cameras_plotly
from depth_anything_3.utils.layout_helpers import hcat, vcat



from genrec.utils.typing import *
from genrec.utils.forward_warp import (
    forward_warp,
    reliable_depth_mask_range_batch,
    depth_gradient_mask_batch_relative,
    precompute_source_view_dirs,
)
from dataclasses import dataclass


def _align_depth_scale_multiview(
    depth: torch.Tensor,       # (T, H, W)
    extrinsics: torch.Tensor,  # (T, 4, 4) w2c
    intrinsics: torch.Tensor,  # (T, 3, 3)
    T_in: int,
) -> float:
    """Compute scalar to align depth maps with camera extrinsic translation scale.

    For each pair of input views (i, j):
      1. Unproject pixels from view i at depth d_i to 3D world points.
      2. Project those 3D points into view j.
      3. Compare projected depth with view j's actual depth.
      4. Collect median(d_j / d_proj) as scale ratio.

    Returns the median ratio across all pairs.
    """
    device = depth.device
    _, H, W = depth.shape
    ratios = []

    # Precompute pixel grids and c2w once
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )  # each (H, W)

    for i in range(T_in):
        d_i = depth[i]  # (H, W)
        K_i = intrinsics[i]  # (3, 3)
        w2c_i = extrinsics[i]  # (4, 4)
        c2w_i = torch.inverse(w2c_i)  # (4, 4)

        # Unproject view i pixels to 3D world coords
        fx_i, fy_i = K_i[0, 0], K_i[1, 1]
        cx_i, cy_i = K_i[0, 2], K_i[1, 2]
        x_cam = (u - cx_i) / fx_i * d_i  # (H, W)
        y_cam = (v - cy_i) / fy_i * d_i  # (H, W)
        pts_cam = torch.stack([x_cam, y_cam, d_i, torch.ones_like(d_i)], dim=-1)  # (H, W, 4)
        pts_world = (c2w_i @ pts_cam.reshape(-1, 4).T).T.reshape(H, W, 4)  # (H, W, 4)

        for j in range(T_in):
            if i == j:
                continue
            d_j = depth[j]  # (H, W)
            K_j = intrinsics[j]  # (3, 3)
            w2c_j = extrinsics[j]  # (4, 4)

            # Project world pts into view j camera frame
            pts_camj = (w2c_j @ pts_world.reshape(-1, 4).T).T.reshape(H, W, 4)  # (H, W, 4)
            d_proj = pts_camj[..., 2]  # (H, W) — projected depth in view j

            # Sample view j's actual depth at projected pixel locations
            fx_j, fy_j = K_j[0, 0], K_j[1, 1]
            cx_j, cy_j = K_j[0, 2], K_j[1, 2]
            u_proj = pts_camj[..., 0] / d_proj.clamp(min=1e-6) * fx_j + cx_j
            v_proj = pts_camj[..., 1] / d_proj.clamp(min=1e-6) * fy_j + cy_j

            # Validity mask: positive depth, within image bounds
            valid = (
                (d_i > 1e-6)
                & (d_proj > 1e-6)
                & (u_proj >= 0) & (u_proj < W - 1)
                & (v_proj >= 0) & (v_proj < H - 1)
            )

            if valid.sum() < 100:
                continue

            # Sample d_j at projected locations via grid_sample
            u_norm = u_proj / (W - 1) * 2 - 1  # [-1, 1]
            v_norm = v_proj / (H - 1) * 2 - 1  # [-1, 1]
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
            d_j_sampled = F.grid_sample(
                d_j.unsqueeze(0).unsqueeze(0), grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze()  # (H, W)

            # Compute ratio d_j_sampled / d_proj at valid pixels
            ratio_map = d_j_sampled[valid] / d_proj[valid].clamp(min=1e-6)
            # Filter outliers
            ratio_map = ratio_map[(ratio_map > 0.1) & (ratio_map < 10.0)]
            if len(ratio_map) > 0:
                ratios.append(ratio_map.median().item())

    if len(ratios) == 0:
        return 1.0
    return float(torch.tensor(ratios).median())


@dataclass
class BatchedPrediction:
    """Container for batched geometry predictions."""
    processed_images: Optional[torch.Tensor]  # (B, T, H, W, 3) uint8
    depth: torch.Tensor  # (B, T, H, W)
    conf: torch.Tensor  # (B, T, H, W)
    extrinsics: torch.Tensor  # (B, T, 4, 4)
    intrinsics: torch.Tensor  # (B, T, 3, 3)
    plucker_rays: Optional[torch.Tensor]  # (B, T, 6, H, W)
    pc_renders: Optional[torch.Tensor]  # (B, T, 3, H, W)
    coord_maps: torch.Tensor  # (B, T, 3, H, W) - world coordinate maps rendered from input depth (loo_pred)
    full_coord_maps: torch.Tensor  # (B, T, 3, H, W) - coord maps from ALL views (for supervision)
    warp_masks: torch.Tensor  # (B, T, 1, H, W) - forward-warp coverage masks
    reliability_maps: Optional[torch.Tensor] = None  # (B, T, 1, H, W) - viewing-angle reliability [0, 1]
    gaussians: Optional[Any] = None  # GS world parameters if infer_gs=True
    scales: Optional[torch.Tensor] = None  # (B,) scale factors
    coord_norm_min: Optional[torch.Tensor] = None  # (B,) per-scene coord normalization min
    coord_norm_max: Optional[torch.Tensor] = None  # (B,) per-scene coord normalization max
    depth_proc: Optional[torch.Tensor] = None       # (B, T, H_proc, W_proc) depth at DA3 process res
    conf_proc: Optional[torch.Tensor] = None         # (B, T, H_proc, W_proc) confidence at DA3 process res
    intrinsics_proc: Optional[torch.Tensor] = None    # (B, T, 3, 3) DA3 intrinsics at process res
    proc_res_hw: Optional[Tuple[int, int]] = None     # (H_proc, W_proc)
    umeyama_scales: Optional[torch.Tensor] = None    # (B,) Umeyama scale: metric / RE10K
    log_scale_coords: bool = False                    # Whether log(1+x) normalization was applied


def da3_gs_head(
    self,
    feats: list[torch.Tensor],
    H: int,
    W: int,
    output: Dict[str, torch.Tensor],
    in_images: torch.Tensor,
    extrinsics: torch.Tensor | None = None,
    intrinsics: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Process 3DGS parameters estimation if 3DGS head is available."""
    if self.gs_head is None or self.gs_adapter is None:
        return output
    assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

    # The depth is defined in the DA3 model's camera space,
    # so even with provided GT camera poses,
    # we instead use the predicted camera poses for better alignment.
    ctx_extr = output.get("extrinsics", None)
    ctx_intr = output.get("intrinsics", None)
    assert (
        ctx_extr is not None and ctx_intr is not None
    ), "must process camera info first if GT is not available"

    gt_extr = extrinsics
    # homo the extr if needed
    ctx_extr = as_homogeneous(ctx_extr)
    if gt_extr is not None:
        gt_extr = as_homogeneous(gt_extr)

    # forward through the gs_dpt head to get 'camera space' parameters
    gs_outs = self.gs_head(
        feats=feats,
        H=H,
        W=W,
        patch_start_idx=0,
        images=in_images,
    )
    raw_gaussians = gs_outs.raw_gs
    densities = gs_outs.raw_gs_conf

    # Surgery on the Gaussians to disclude the first view
    raw_gaussians = raw_gaussians[:, :-1]
    densities = densities[:, :-1]

    # convert to 'world space' 3DGS parameters; ready to export and render
    # gt_extr could be None, and will be used to align the pose scale if available
    gs_world = self.gs_adapter(
        extrinsics=ctx_extr[:, :-1],
        intrinsics=ctx_intr[:, :-1],
        depths=output.depth[:, :-1],
        opacities=map_pdf_to_opacity(densities),
        raw_gaussians=raw_gaussians,
        image_shape=(H, W),
        gt_extrinsics=gt_extr,
    )
    output.gaussians = gs_world
    
    return output


def _normalize_extrinsics_for_da3(extrinsics, device):
    """Normalize extrinsics: relative to first camera, scaled by median distance."""
    ext = extrinsics.clone().float().to(device)
    if ext.shape[-2] == 3:
        ext = as_homogeneous(ext)
    transform = affine_inverse(ext[:, :1])  # (B, 1, 4, 4)
    ext = ext @ transform
    c2ws = affine_inverse(ext)
    dists = c2ws[..., :3, 3].norm(dim=-1)
    median_dist = torch.clamp(torch.median(dists), min=1e-1)
    ext[..., :3, 3] /= median_dist
    return ext


class GeometryModel(DepthAnything3):
    """
    Wrapper around DepthAnything3 that adds specific data loading 
    and preprocessing capabilities for COLMAP datasets.
    """
    
    def __init__(self, model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE", **kwargs):
        # Pass arguments to the parent DepthAnything3 class
        super().__init__(model_name=model_name, **kwargs)
        # Override the gs head with our custom implementation
        self.model.da3._process_gs_head = da3_gs_head.__get__(self.model.da3, self.model.da3.__class__)
        self.model.da3_metric._process_gs_head = da3_gs_head.__get__(self.model.da3_metric, self.model.da3_metric.__class__)

    @staticmethod
    def load_colmap_data(root_path: str | Path, image_step: int = 1, cutoff: int = 8):
        """
        Parses COLMAP transforms.json and images, performing the specific 
        coordinate system transformations required by your workflow.
        """
        root_path = Path(root_path)
        images_path = root_path / "images"
        transforms_path = root_path / "transforms.json"

        if not transforms_path.exists():
            raise FileNotFoundError(f"transforms.json not found at {transforms_path}")

        # 1. Load Images list
        # Using sorted glob to match your original logic
        images_full_list = sorted(glob.glob(os.path.join(images_path, "*.png")))

        images = images_full_list[:cutoff * image_step:image_step]
        
        print(f"Found {len(images)} images (step={image_step})")

        # 2. Load Transforms
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)

        frame_extrinsics = []
        frame_intrinsics = []
        render_exts = []
        render_ixts = []

        # 3. Calculate Global Transform (Normalize scene based on first frame)
        # Logic copied exactly from your script
        first_frame = transforms['frames'][0]
        first_frame_c2w = np.array(first_frame['transform_matrix'])
        
        # Invert Z axis
        first_frame_c2w[2, :] *= -1 
        # Swap X and Y axes
        first_frame_c2w = first_frame_c2w[np.array([1, 0, 2, 3]), :] 
        # Invert Y and Z rotation columns
        first_frame_c2w[0:3, 1:3] *= -1 
        
        transform_mat = np.linalg.inv(first_frame_c2w)

        # 4. Process all frames
        # Helper to extract intrinsics
        fx, fy = transforms['fl_x'], transforms['fl_y']
        cx, cy = transforms['cx'], transforms['cy']
        base_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        for frame in transforms['frames']:
            # Apply same transforms to current frame
            c2w = np.array(frame['transform_matrix'])
            c2w[2, :] *= -1
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[0:3, 1:3] *= -1
            
            # Apply global normalization
            c2w = transform_mat @ c2w
            
            # Convert to OpenCV W2C (Extrinsics)
            w2c_opencv = np.linalg.inv(c2w)
            
            render_exts.append(w2c_opencv)
            render_ixts.append(base_intrinsics)

            # Check if this frame is in our selected image list
            frame_filename = os.path.basename(frame['file_path'])
            full_path = str(images_path / frame_filename)
            
            # Simple string matching to see if this frame is in our processed list
            if full_path in images:        
                frame_extrinsics.append(w2c_opencv)
                frame_intrinsics.append(base_intrinsics)

        # 5. Stack into Numpy Arrays
        return {
            "images": images,
            "extrinsics": np.stack(frame_extrinsics),
            "intrinsics": np.stack(frame_intrinsics),
            "render_exts": np.stack(render_exts),
            "render_ixts": np.stack(render_ixts),
            "original_h": transforms['h'],
            "original_w": transforms['w']
        }
        
    
    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
        skip_alignment: bool = False,
    ) -> Prediction:
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction, None
        if skip_alignment:
            prediction.intrinsics = intrinsics.numpy()
            return prediction, None
        prediction.intrinsics = intrinsics.numpy()
        r, t, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        # visualize_cameras_plotly(aligned_extrinsics, prediction.extrinsics)

        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction, scale
    
    @torch.no_grad()
    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        skip_alignment: bool = False,
        infer_gs: bool = False,
        process_res: int = 616,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: Sequence[int] | None = None,
        # GLB export parameters
        conf_thresh_percentile: float = 20.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: Optional[dict] = {},
    ) -> Prediction:
        """
        Run inference on input images.

        Args:
            image: List of input images (numpy arrays, PIL Images, or file paths)
            extrinsics: Camera extrinsics (N, 4, 4)
            intrinsics: Camera intrinsics (N, 3, 3)
            align_to_input_ext_scale: whether to align the input pose scale to the prediction
            infer_gs: Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)
            render_exts: Optional render extrinsics for Gaussian video export
            render_ixts: Optional render intrinsics for Gaussian video export
            render_hw: Optional render resolution for Gaussian video export
            process_res: Processing resolution
            process_res_method: Resize method for processing
            export_feat_layers: Layer indices to export intermediate features from
            conf_thresh_percentile: [GLB] Lower percentile for adaptive confidence threshold (default: 40.0) # noqa: E501
            num_max_points: [GLB] Maximum number of points in the point cloud (default: 1,000,000)
            show_cameras: [GLB] Show camera wireframes in the exported scene (default: True)
            feat_vis_fps: [FEAT_VIS] Frame rate for output video (default: 15)
            export_kwargs: additional arguments to export functions.

        Returns:
            Prediction object containing depth maps and camera parameters
        """

        # Preprocess images
        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )

        # Prepare tensors for model
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        # Run model forward pass
        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self._run_model_forward(imgs, ex_t_norm, in_t, export_feat_layers, infer_gs)

        # Convert raw output to prediction
        prediction = self._convert_to_prediction(raw_output)

        # Align prediction to extrinsincs
        prediction, scale = self._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction, align_to_input_ext_scale,
            skip_alignment=skip_alignment,
        )

        # Add processed images for visualization
        prediction = self._add_processed_images(prediction, imgs_cpu)

        return prediction, scale

    @staticmethod
    @torch.no_grad()
    def render_views_gpu(
        source_depth: torch.Tensor,       # (T_in, H, W)
        source_w2c: torch.Tensor,         # (T_in, 4, 4)
        source_intrinsics: torch.Tensor,  # (T_in, 3, 3)
        target_w2c: torch.Tensor,         # (T_target, 4, 4)
        target_intrinsics: torch.Tensor,  # (T_target, 3, 3)
        source_images: Optional[torch.Tensor] = None,   # (T_in, 3, H, W) float [0,1]
        source_coords: Optional[torch.Tensor] = None,   # (T_in, 3, H, W) float
        mask: Optional[torch.Tensor] = None,             # (T_in, 1, H, W)
        foreground_masking: bool = True,
        cameraray_filtering: bool = False,
        depth_filter_thresh: float = 0.5,
        depth_grad_thresh: float = 0.0,
        depth_weight_scale: float = 50.0,
        compute_reliability: bool = False,
    ) -> dict:
        """
        GPU forward-warp rendering of source views into target camera poses.

        Concatenates source_images and source_coords into a single multi-channel
        frame, warps via bilinear splatting with depth-weighted blending, then
        splits the result back.

        Returns dict with keys:
            "images":       (T_target, C_img, H, W)  or None
            "coords":       (T_target, C_coord, H, W) or None
            "masks":        (T_target, 1, H, W)
            "reliability":  (T_target, 1, H, W) or None — viewing-angle reliability
        """
        T_in = source_depth.shape[0]
        H, W = source_depth.shape[1], source_depth.shape[2]
        device = source_depth.device
        dtype = source_depth.dtype

        # --- Build multi-channel frame ---
        channels = []
        c_img = 0
        if source_images is not None:
            channels.append(source_images)  # (T_in, 3, H, W)
            c_img = source_images.shape[1]
        c_coord = 0
        if source_coords is not None:
            channels.append(source_coords)  # (T_in, 3, H, W)
            c_coord = source_coords.shape[1]

        assert len(channels) > 0, "At least one of source_images or source_coords must be provided"
        frame = torch.cat(channels, dim=1)  # (T_in, C, H, W)
        C = frame.shape[1]

        # --- Build validity mask ---
        depth_mask = (source_depth > 0) & torch.isfinite(source_depth)
        depth_mask = depth_mask.unsqueeze(1).float()  # (T_in, 1, H, W)

        # Optional reliable-depth edge filter (remove depth discontinuities)
        if depth_filter_thresh > 0:
            reliable = reliable_depth_mask_range_batch(
                source_depth, window_size=5, ratio_thresh=depth_filter_thresh,
            ).float()  # (T_in, 1, H, W)
            depth_mask = depth_mask * reliable

        # Optional relative depth-gradient edge filter (mirrors the visualizer's
        # depth_edge_mask_relative; threshold transfers 1:1 from there).
        if depth_grad_thresh > 0:
            grad_ok = depth_gradient_mask_batch_relative(
                source_depth, threshold=depth_grad_thresh,
            ).float()  # (T_in, 1, H, W)
            depth_mask = depth_mask * grad_ok

        if mask is not None:
            depth_mask = depth_mask * mask.float()

        depth_4d = source_depth.unsqueeze(1)  # (T_in, 1, H, W)

        T_target = target_w2c.shape[0]
        out_images_list = []
        out_coords_list = []
        out_masks_list = []
        out_reliability_list = []

        # Precompute source-side view directions once (reused across all targets)
        precomp_rel = None
        if compute_reliability:
            precomp_rel = precompute_source_view_dirs(
                depth_4d, source_w2c, source_intrinsics,
            )

        for j in range(T_target):
            # Expand source tensors: each source view paired with same target
            tgt_w2c_j = target_w2c[j:j+1].expand(T_in, -1, -1)       # (T_in, 4, 4)
            tgt_K_j = target_intrinsics[j:j+1].expand(T_in, -1, -1)   # (T_in, 3, 3)

            warped, wmask, _, _, reliability = forward_warp(
                frame1=frame,
                mask1=depth_mask,
                depth1=depth_4d,
                w2c1=source_w2c,
                w2c2=tgt_w2c_j,
                intrinsic1=source_intrinsics,
                intrinsic2=tgt_K_j,
                is_image=False,
                cameraray_filtering=cameraray_filtering,
                foreground_masking=foreground_masking,
                boundary_mask=depth_mask[:, 0].bool() if foreground_masking else None,
                n_views=T_in,
                precomputed_reliability=precomp_rel,
                depth_weight_scale=depth_weight_scale,
            )
            # warped: (1, C, H, W),  wmask: (1, 1, H, W)
            out_masks_list.append(wmask[0])  # (1, H, W)

            if reliability is not None:
                out_reliability_list.append(reliability[0])  # (1, H, W)

            if c_img > 0:
                out_images_list.append(warped[0, :c_img])   # (c_img, H, W)
            if c_coord > 0:
                out_coords_list.append(warped[0, c_img:])   # (c_coord, H, W)

        result = {
            "images": torch.stack(out_images_list, dim=0) if out_images_list else None,
            "coords": torch.stack(out_coords_list, dim=0) if out_coords_list else None,
            "masks":  torch.stack(out_masks_list, dim=0),  # (T_target, 1, H, W)
            "reliability": torch.stack(out_reliability_list, dim=0) if out_reliability_list else None,
        }
        return result

    @torch.no_grad()
    def batched_forward(
        self,
        images: torch.Tensor,  # (B, T, H, W, 3) - raw images, uint8 or float [0, 255]
        process_res: int = 616,
        process_res_method: str = "upper_bound_resize",
        infer_gs: bool = True,
        T_in: int = 3,
        T_out: int = 1,
        conf_mask_thresh: float = 0.0,
        gt_extrinsics: torch.Tensor = None,  # (B, T, 4, 4) GT w2c matrices
        gt_intrinsics: torch.Tensor = None,  # (B, T, 3, 3) GT intrinsics at original image res
        depth_filter_thresh: float = 0.5,
        depth_grad_thresh: float = 0.0,
        depth_weight_scale: float = 50.0,
        dyn_masks: torch.Tensor = None,  # (B, T, 1, H, W) 1=dynamic 0=static
        precomputed_depths: torch.Tensor = None,  # (B, T, H, W) precomputed depth maps
        classify_only: bool = False,  # skip plucker rays, pc renders, image processing
        source_only_da3: bool = False,  # only pass source views through DA3 (prevents cross-view contamination)
        compute_reliability: bool = False,  # compute viewing-angle reliability maps
        log_scale_coords: bool = False,  # use log(1+x) normalization for scene coordinates
        normalize_with_all_views: bool = False,  # bounds from input + output views (GT at eval)
    ) -> BatchedPrediction:
        """
        Process multiple scenes in a batched fashion for efficient GPU utilization.
        
        This method processes all scenes in parallel through the depth/geometry backbone,
        then performs per-scene pose estimation and rendering.
        
        Args:
            images: Raw images tensor of shape (B, T, H, W, 3) where:
                    B = batch size (number of scenes)
                    T = number of views per scene (T_in + T_out)
                    H, W = image dimensions
                    3 = RGB channels
                    Values should be in [0, 255] range (uint8 or float)
            process_res: Processing resolution for depth estimation
            process_res_method: Resize method for processing
            infer_gs: Whether to infer 3D Gaussians
            T_in: Number of input views
            T_out: Number of output views
            
        Returns:
            BatchedPrediction containing all geometry outputs for the batch
        """
        B, T, H_orig, W_orig, C = images.shape
        device = images.device
        umeyama_scale_list = None  # populated by DA3 Umeyama alignment (not precomputed path)

        if precomputed_depths is not None:
            # --- Precomputed depths path: skip DA3 entirely ---
            assert gt_extrinsics is not None and gt_intrinsics is not None, \
                "GT poses required when using precomputed depths"

            depth = precomputed_depths.float().to(device)       # (B, T, H_orig, W_orig)
            conf = torch.ones_like(depth)                       # full confidence
            pred_extrinsics = gt_extrinsics.float().to(device)  # (B, T, 4, 4)
            pred_intrinsics = gt_intrinsics.float().to(device)  # (B, T, 3, 3) at orig res

            # Align depth scale to GT pose scale via multi-view reprojection
            for b in range(B):
                scale = _align_depth_scale_multiview(
                    depth[b], pred_extrinsics[b], pred_intrinsics[b], T_in
                )
                if math.isnan(scale) or math.isinf(scale):
                    print(f"[NaN DEBUG] NaN/Inf depth alignment scale for batch {b}, falling back to 1.0")
                    scale = 1.0
                elif scale < 0.5 or scale > 2.0:
                    print(f"[WARNING] Depth alignment scale = {scale:.4f} (expected ~1.0)")
                depth[b] *= scale

            # Build raw images tensor at original resolution
            raw_imgs_chw = (images.float().to(device) / 255.0).permute(0, 1, 4, 2, 3).reshape(B * T, 3, H_orig, W_orig)

            # No process-res data available (DA3 was skipped)
            depth_at_proc = None
            conf_at_proc = None
            da3_intrinsics_proc = None
        else:
            # --- DA3 path: GPU-native input processing ---
            images_flat = images.view(B * T, H_orig, W_orig, C)
            if images_flat.dtype != torch.uint8:
                images_flat = images_flat.clamp(0, 255).to(torch.uint8)

            # (B*T, 3, H, W) float [0, 1]
            imgs_chw = images_flat.float().permute(0, 3, 1, 2) / 255.0

            # Upper-bound resize: scale longest side to process_res, preserve aspect ratio
            PATCH_SIZE_X4 = 56  # DA3 requires divisibility by 14*4
            longest = max(H_orig, W_orig)
            scale = process_res / longest
            H_scaled = int(round(H_orig * scale))
            W_scaled = int(round(W_orig * scale))
            # Round to nearest multiple of 56
            H_proc = max(PATCH_SIZE_X4, ((H_scaled + PATCH_SIZE_X4 // 2) // PATCH_SIZE_X4) * PATCH_SIZE_X4)
            W_proc = max(PATCH_SIZE_X4, ((W_scaled + PATCH_SIZE_X4 // 2) // PATCH_SIZE_X4) * PATCH_SIZE_X4)

            imgs_resized = F.interpolate(
                imgs_chw.to(device), size=(H_proc, W_proc),
                mode='bicubic' if scale > 1.0 else 'area',
                align_corners=False if scale > 1.0 else None,
            )

            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            imgs_tensor = (imgs_resized - mean) / std

            # Reshape for DA3: (B, T, 3, H_proc, W_proc)
            imgs_batched = imgs_tensor.view(B, T, 3, H_proc, W_proc)

            # Prepare camera conditioning for DA3 if GT poses available
            # (matches DA3's _normalize_extrinsics + intrinsics scaling in api.py)
            da3_ext = None
            da3_int = None
            if gt_extrinsics is not None and gt_intrinsics is not None:
                # Scale GT intrinsics from orig_res to process_res
                da3_int = gt_intrinsics.clone().float().to(device)
                da3_int[..., 0, :] *= W_proc / W_orig
                da3_int[..., 1, :] *= H_proc / H_orig

                # Normalize GT extrinsics (same as DA3's _normalize_extrinsics)
                da3_ext = _normalize_extrinsics_for_da3(gt_extrinsics, device)

            autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            # Build raw images tensor at original resolution (bypass ImageNet norm round-trip)
            raw_imgs_chw = (images.float().to(device) / 255.0).permute(0, 1, 4, 2, 3).reshape(B * T, 3, H_orig, W_orig)

            if source_only_da3:
                # Two-pass approach: avoids target views contaminating source depth
                # via DA3's multi-view attention.
                assert gt_extrinsics is not None, \
                    "source_only_da3 requires gt_extrinsics"

                # Pass 1: all views (pose estimation only, for Umeyama scale)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    raw_output_all = self.model(imgs_batched, da3_ext, da3_int, [], infer_gs)
                pred_extrinsics_all = as_homogeneous(raw_output_all["extrinsics"])  # (B, T, 4, 4)

                # Umeyama alignment to get per-scene scale
                umeyama_scale_list = []
                for b in range(B):
                    _, _, s = align_poses_umeyama(
                        pred_extrinsics_all[b].cpu().float().numpy()[:, :3, :],
                        gt_extrinsics[b].cpu().float().numpy()[:, :3, :],
                    )
                    umeyama_scale_list.append(s)
                del raw_output_all, pred_extrinsics_all  # free GPU memory before pass 2

                # Pass 2: source views only (uncontaminated depth)
                imgs_src = imgs_batched[:, :T_in]
                da3_ext_src = _normalize_extrinsics_for_da3(gt_extrinsics[:, :T_in], device) if da3_ext is not None else None
                da3_int_src = da3_int[:, :T_in] if da3_int is not None else None
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    raw_output_src = self.model(imgs_src, da3_ext_src, da3_int_src, [], infer_gs)

                # Extract source-only outputs
                depth_src = raw_output_src["depth"]       # (B, T_in, H, W)
                conf_src = raw_output_src["depth_conf"]   # (B, T_in, H, W)
                del raw_output_src

                # Pad with zeros for target views
                T_out = T - T_in
                depth = F.pad(depth_src, (0, 0, 0, 0, 0, T_out))   # (B, T, H, W)
                conf = F.pad(conf_src, (0, 0, 0, 0, 0, T_out))     # (B, T, H, W)

                # Scale GT pose translations to metric space (depth is already metric)
                pred_extrinsics = gt_extrinsics.clone().float().to(device)
                for b in range(B):
                    pred_extrinsics[b, :, :3, 3] *= umeyama_scale_list[b]

                gt_intrinsics_proc = gt_intrinsics.clone().float()
                gt_intrinsics_proc[..., 0, :] *= W_proc / W_orig
                gt_intrinsics_proc[..., 1, :] *= H_proc / H_orig
                pred_intrinsics = gt_intrinsics_proc.to(device)

            else:
                # Standard single-pass: DA3 on all views
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    raw_output = self.model(imgs_batched, da3_ext, da3_int, [], infer_gs)

                # Extract outputs - shapes are (B, T, ...)
                depth = raw_output["depth"]           # (B, T, H, W)
                conf = raw_output["depth_conf"]       # (B, T, H, W)
                pred_extrinsics = raw_output["extrinsics"]  # (B, T, 3, 4)
                pred_intrinsics = raw_output["intrinsics"]  # (B, T, 3, 3)

                # Make extrinsics homogeneous (B, T, 4, 4)
                pred_extrinsics = as_homogeneous(pred_extrinsics)

                # Replace predicted poses with GT if provided
                if gt_extrinsics is not None:
                    gt_intrinsics_proc = gt_intrinsics.clone().float()
                    gt_intrinsics_proc[..., 0, :] *= W_proc / W_orig
                    gt_intrinsics_proc[..., 1, :] *= H_proc / H_orig

                    # Per-scene Umeyama alignment to get scale, then rescale depth
                    umeyama_scale_list = []
                    for b in range(B):
                        pred_ext_np = pred_extrinsics[b].cpu().float().numpy()
                        gt_ext_np = gt_extrinsics[b].cpu().float().numpy()
                        _, _, scale = align_poses_umeyama(pred_ext_np[:, :3, :], gt_ext_np[:, :3, :])
                        depth[b] /= scale
                        umeyama_scale_list.append(scale)

                    pred_extrinsics = gt_extrinsics.float().to(device)
                    pred_intrinsics = gt_intrinsics_proc.to(device)

            del imgs_tensor, imgs_resized  # Free DA3-resolution tensors

            # Save intrinsics at process_res for GEN3C cache
            # When --use_gt_poses: GT intrinsics at process_res (matches GEN3C's behavior
            # where _align_to_input_extrinsics_intrinsics replaces prediction.intrinsics with GT)
            # Otherwise: DA3's predicted intrinsics at process_res
            da3_intrinsics_proc = pred_intrinsics.clone()  # (B, T, 3, 3) at process_res

            # Save DA3 process-res data before downsample (for GEN3C renderer alignment)
            depth_at_proc = depth.clone()
            conf_at_proc = conf.clone()

            # Downsample depth/conf/intrinsics from DA3 process_res back to raw image resolution
            if H_proc != H_orig or W_proc != W_orig:
                depth = F.interpolate(
                    depth.view(B * T, 1, H_proc, W_proc), size=(H_orig, W_orig),
                    mode="bilinear", align_corners=False,
                ).view(B, T, H_orig, W_orig)
                conf = F.interpolate(
                    conf.view(B * T, 1, H_proc, W_proc), size=(H_orig, W_orig),
                    mode="bilinear", align_corners=False,
                ).view(B, T, H_orig, W_orig)
                pred_intrinsics = pred_intrinsics.clone()
                pred_intrinsics[..., 0, :] *= W_orig / W_proc  # fx, cx, tx
                pred_intrinsics[..., 1, :] *= H_orig / H_proc  # fy, cy, ty

        # Compute plucker rays and pc renders for each scene
        if not classify_only:
            from genrec.utils.cam_ops import get_plucker_rays, get_ray_directions
            import einops

        all_plucker_rays = []
        all_pc_renders = []
        all_coord_maps = []
        all_warp_masks = []
        all_reliability_maps = []
        all_full_coord_maps = []
        all_scales = []
        all_norm_min = []
        all_norm_max = []

        for b in range(B):
            # Get scene-specific data
            scene_extrinsics = pred_extrinsics[b]  # (T, 4, 4)
            scene_intrinsics = pred_intrinsics[b]  # (T, 3, 3)
            scene_depth = depth[b]  # (T, H, W)
            scene_conf = conf[b]  # (T, H, W)
            
            if not classify_only:
                # Compute ray directions for each view
                ray_directions = torch.stack([
                    get_ray_directions(
                        H=H_orig,
                        W=W_orig,
                        focal=(scene_intrinsics[t, 0, 0].item(), scene_intrinsics[t, 1, 1].item()),
                        principal=(scene_intrinsics[t, 0, 2].item(), scene_intrinsics[t, 1, 2].item()),
                    ) for t in range(T)
                ], dim=0)  # (T, H, W, 3)

                # Camera-to-world
                c2w = torch.inverse(scene_extrinsics.float())  # (T, 4, 4)

                # Ensure ray_directions is on the same device as c2w
                ray_directions = ray_directions.to(c2w.device)

                # Compute plucker rays
                plucker_rays = get_plucker_rays(
                    directions=ray_directions,
                    c2w=c2w,
                    keepdim=True,
                )  # (T, H, W, 6)
                plucker_rays = einops.rearrange(plucker_rays, "T H W C -> T C H W")
                all_plucker_rays.append(plucker_rays)

                # Scale factor (kept for BatchedPrediction.scales output)
                scale = scene_depth.max().item() / 10.0
                scale = max(scale, 1.0)
                all_scales.append(scale)

            # Use UNSCALED extrinsics for forward warp — bilinear splatting
            # handles sub-pixel displacement natively, so the old CPU-renderer
            # hack of scaling translations is no longer needed.
            w2c_float = scene_extrinsics.float()  # (T, 4, 4)

            # --- Compute input-view world coords (unnormalized) ---
            input_coords_raw = self.unproject_depth_to_world_coords(
                depth=scene_depth[:T_in],
                intrinsics=scene_intrinsics[:T_in],
                extrinsics=scene_extrinsics[:T_in],
                normalize=False,
            )  # (T_in, 3, H, W)

            # Bounds: input views only by default; optionally extend to output views.
            if normalize_with_all_views and T_out > 0:
                output_coords_raw = self.unproject_depth_to_world_coords(
                    depth=scene_depth[T_in:],
                    intrinsics=scene_intrinsics[T_in:],
                    extrinsics=scene_extrinsics[T_in:],
                    normalize=False,
                )  # (T_out, 3, H, W)
                norm_min = torch.minimum(input_coords_raw.min(), output_coords_raw.min())
                norm_max = torch.maximum(input_coords_raw.max(), output_coords_raw.max())
            else:
                norm_min = input_coords_raw.min()   # scalar
                norm_max = input_coords_raw.max()   # scalar
            scene_size = (norm_max - norm_min).clamp(min=1e-6)
            if scene_size.item() < 1e-4:
                print(f"[NaN DEBUG] Degenerate scene {b}: coord range={scene_size.item():.8f}, "
                      f"min={norm_min.item():.4f}, max={norm_max.item():.4f}")

            all_norm_min.append(norm_min)
            all_norm_max.append(norm_max)

            if log_scale_coords:
                coords_shifted = input_coords_raw - norm_min  # >= 0
                input_coords_norm = torch.log1p(coords_shifted) / torch.log1p(scene_size) * 2.0 - 1.0
            else:
                input_coords_norm = (input_coords_raw - norm_min) / scene_size * 2.0 - 1.0
            input_coords_norm = input_coords_norm.clamp(-1.0, 1.0)

            # --- Source images as float [0,1] on GPU ---
            if not classify_only:
                start_idx = b * T
                source_imgs = raw_imgs_chw[start_idx:start_idx + T_in]  # (T_in, 3, H_orig, W_orig)

            # --- GPU forward-warp rendering (input views → all views) ---
            conf_mask = None
            if conf_mask_thresh > 0:
                conf_mask = (scene_conf[:T_in] > conf_mask_thresh).unsqueeze(1).float()

            # NOTE: dyn_masks are NOT applied to conf_mask for forward warp.
            # Dynamic pixels should still contribute to PC renders (the model
            # needs to see them as conditioning). dyn_masks are only used below
            # for coord map masking (supervision targets).

            warp_result = self.render_views_gpu(
                source_depth=scene_depth[:T_in],
                source_w2c=w2c_float[:T_in],
                source_intrinsics=scene_intrinsics[:T_in].float(),
                target_w2c=w2c_float,
                target_intrinsics=scene_intrinsics.float(),
                source_images=None if classify_only else source_imgs.float(),
                source_coords=input_coords_norm.float(),
                mask=conf_mask,
                foreground_masking=False,
                cameraray_filtering=False,
                depth_filter_thresh=depth_filter_thresh,
                depth_grad_thresh=depth_grad_thresh,
                depth_weight_scale=depth_weight_scale,
                compute_reliability=compute_reliability,
            )
            coord_maps_tensor = warp_result["coords"]   # (T, 3, H, W)
            warp_masks_tensor = warp_result["masks"]     # (T, 1, H, W)

            if not classify_only:
                pc_renders_tensor = warp_result["images"]   # (T, 3, H, W)
                all_pc_renders.append(pc_renders_tensor)
            all_coord_maps.append(coord_maps_tensor)
            all_warp_masks.append(warp_masks_tensor)
            if warp_result["reliability"] is not None:
                all_reliability_maps.append(warp_result["reliability"])

            # --- Full coord maps: dense unprojection for ALL views, same bounds ---
            # Mask dynamic pixels in depth before unprojecting to prevent incorrect
            # 3D coordinate supervision targets in dynamic regions
            if dyn_masks is not None:
                scene_dyn = dyn_masks[b, :T]  # (T, 1, H_orig, W_orig)
                if scene_dyn.shape[-2:] != scene_depth.shape[-2:]:
                    scene_dyn = F.interpolate(
                        scene_dyn, size=scene_depth.shape[-2:], mode="nearest"
                    )
                masked_depth = scene_depth * (1.0 - scene_dyn[:, 0])  # zero dynamic pixels
            else:
                masked_depth = scene_depth
            all_views_coord = self.unproject_depth_to_world_coords(
                depth=masked_depth,
                intrinsics=scene_intrinsics,
                extrinsics=scene_extrinsics,
                normalize=True,
                norm_min=norm_min,
                norm_max=norm_max,
                log_scale_coords=log_scale_coords,
            )  # (T, 3, H, W)
            all_full_coord_maps.append(all_views_coord)
        
        # Stack all results
        coord_maps_batched = torch.stack(all_coord_maps, dim=0).to(device)  # (B, T, 3, H, W)
        warp_masks_batched = torch.stack(all_warp_masks, dim=0).to(device)  # (B, T, 1, H, W)
        reliability_batched = (
            torch.stack(all_reliability_maps, dim=0).to(device)  # (B, T, 1, H, W)
            if all_reliability_maps else None
        )
        full_coord_maps_batched = torch.stack(all_full_coord_maps, dim=0).to(device)  # (B, T, 3, H, W)

        if not classify_only:
            plucker_rays_batched = torch.stack(all_plucker_rays, dim=0)  # (B, T, 6, H, W)
            pc_renders_batched = torch.stack(all_pc_renders, dim=0).to(device)  # (B, T, 3, H, W)
            scales_tensor = torch.tensor(all_scales, device=device)

            # Get processed images from raw inputs at original resolution
            processed_imgs = raw_imgs_chw.view(B, T, 3, H_orig, W_orig)
            processed_imgs = (processed_imgs.clamp(0, 1) * 255).to(torch.uint8)
            processed_imgs = processed_imgs.permute(0, 1, 3, 4, 2)  # (B, T, H_orig, W_orig, 3)

            # Gaussians not used in this pipeline
            gaussians = None
        else:
            plucker_rays_batched = None
            pc_renders_batched = None
            scales_tensor = None
            processed_imgs = None
            gaussians = None

        # proc_res_hw only defined in DA3 path
        proc_res_hw = None
        if precomputed_depths is None:
            proc_res_hw = (H_proc, W_proc)

        return BatchedPrediction(
            processed_images=processed_imgs,
            depth=depth,
            conf=conf,
            extrinsics=pred_extrinsics,
            intrinsics=pred_intrinsics,
            plucker_rays=plucker_rays_batched.to(device) if plucker_rays_batched is not None else None,
            pc_renders=pc_renders_batched,
            coord_maps=coord_maps_batched,
            full_coord_maps=full_coord_maps_batched,
            warp_masks=warp_masks_batched,
            reliability_maps=reliability_batched,
            gaussians=gaussians,
            scales=scales_tensor,
            coord_norm_min=torch.stack(all_norm_min).to(device),  # (B,)
            coord_norm_max=torch.stack(all_norm_max).to(device),  # (B,)
            depth_proc=depth_at_proc,
            conf_proc=conf_at_proc,
            intrinsics_proc=da3_intrinsics_proc,
            proc_res_hw=proc_res_hw,
            umeyama_scales=torch.tensor(umeyama_scale_list, device=device) if umeyama_scale_list is not None else None,
            log_scale_coords=log_scale_coords,
        )

    def scale_render_extrinsics(
        self,
        render_exts: np.ndarray | torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scale render extrinsics by the given scale factor."""
        render_exts[:, :3, 3] = render_exts[:, :3, 3] * scale
        return render_exts
    
    def scale_render_intrinsics(
        self,
        render_ixts: np.ndarray,
        new_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Resize render intrinsics to match new image resolution."""
        new_h, new_w = new_hw
        w, h = render_ixts[0, 0, 2] * 2, render_ixts[0, 1, 2] * 2  # Assuming cx, cy are at center

        render_ixts = np.stack([
            self.input_processor._resize_ixt(ixt, w, h, new_w, new_h)
            for ixt in render_ixts
        ], axis=0)
        return render_ixts

    @staticmethod
    def get_3dgs_renders(
        prediction: Prediction,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        out_image_hw: Optional[tuple[int, int]] = None,
        chunk_size: Optional[int] = 4,
        color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+ED",
    ) -> torch.Tensor:
        """
        Render 3D Gaussians from the prediction at specified camera poses.

        Args:
            prediction: Prediction object containing 3D Gaussians
            extrinsics: Camera extrinsics for rendering (B, V, 4, 4)
            intrinsics: Camera intrinsics for rendering (B, V, 3, 3)
            out_image_hw: Output image height and width (H, W)
            chunk_size: Chunk size for rendering to manage memory usage
        """
        gs_world = prediction.gaussians
        
        tgt_extrs = extrinsics
        tgt_intrs = intrinsics

        if prediction.is_metric:
            scale_factor = prediction.scale_factor
            if scale_factor is not None:
                tgt_extrs[:, :, :3, 3] /= scale_factor

        if out_image_hw is not None:
            H, W = out_image_hw
        else:
            H, W = prediction.depth.shape[-2:]
        
        color, depth, alpha, blur_map = run_renderer_in_chunk_w_trj_mode(
            gaussians=gs_world,
            extrinsics=tgt_extrs,
            intrinsics=tgt_intrs,
            image_shape=(H, W),
            chunk_size=chunk_size,
            trj_mode="original",
            use_sh=True,
            color_mode=color_mode,
            enable_tqdm=False,
        )

        return color, depth, alpha, blur_map

    @staticmethod
    def unproject_depth_to_world_coords(
        depth: torch.Tensor,        # (T, H, W)
        intrinsics: torch.Tensor,   # (T, 3, 3)
        extrinsics: torch.Tensor,   # (T, 4, 4) world-to-camera
        normalize: bool = True,
        norm_min: Optional[torch.Tensor] = None,  # external min (in working space)
        norm_max: Optional[torch.Tensor] = None,  # external max (in working space)
        log_scale_coords: bool = False,  # use signed_log1p compression
    ) -> torch.Tensor:
        """
        Unproject per-pixel depth to world coordinates using camera parameters.
        Produces dense per-pixel coordinate maps with no holes.

        Args:
            depth: Depth maps (T, H, W)
            intrinsics: Camera intrinsic matrices (T, 3, 3)
            extrinsics: World-to-camera extrinsic matrices (T, 4, 4)
            normalize: If True, normalize coords to [0, 1] per scene
            norm_min: Optional external per-axis min for normalization (from input views)
            norm_max: Optional external per-axis max for normalization (from input views)

        Returns:
            coord_maps: (T, 3, H, W) world coordinate maps
        """
        T, H, W = depth.shape
        device = depth.device

        # Pixel grid (shared across views)
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        ones = torch.ones_like(u)
        uv1 = torch.stack([u, v, ones], dim=0).reshape(3, -1)  # (3, H*W)

        # Camera-to-world: c2w = inv(w2c)
        c2w = torch.inverse(extrinsics.float())  # (T, 4, 4)
        if torch.isnan(c2w).any() or torch.isinf(c2w).any():
            print(f"[NaN DEBUG] NaN/Inf in c2w after inverse, det={torch.det(extrinsics.float()).tolist()}")

        # Unproject each view
        K_inv = torch.inverse(intrinsics.float())  # (T, 3, 3)
        if torch.isnan(K_inv).any() or torch.isinf(K_inv).any():
            print(f"[NaN DEBUG] NaN/Inf in K_inv after inverse, det={torch.det(intrinsics.float()).tolist()}")
        cam_dirs = K_inv @ uv1.unsqueeze(0).expand(T, -1, -1)  # (T, 3, H*W)
        cam_pts = cam_dirs * depth.reshape(T, 1, -1)  # (T, 3, H*W)

        # Homogeneous camera coords
        ones_row = torch.ones((T, 1, H * W), device=device)
        cam_pts_h = torch.cat([cam_pts, ones_row], dim=1)  # (T, 4, H*W)

        # Transform to world
        world_pts = c2w @ cam_pts_h  # (T, 4, H*W)
        coord_maps = world_pts[:, :3, :].reshape(T, 3, H, W)  # (T, 3, H, W)

        if normalize:
            if norm_min is not None and norm_max is not None:
                global_min = norm_min.to(device)
                global_max = norm_max.to(device)
            else:
                global_min = coord_maps.min()
                global_max = coord_maps.max()

            scene_size = (global_max - global_min).clamp(min=1e-6)  # scalar
            if log_scale_coords:
                coords_shifted = coord_maps - global_min  # >= 0
                coord_maps = torch.log1p(coords_shifted) / torch.log1p(scene_size) * 2.0 - 1.0
            else:
                coord_maps = (coord_maps - global_min) / scene_size * 2.0 - 1.0
            coord_maps = coord_maps.clamp(-1.0, 1.0)

        return coord_maps

    @staticmethod
    def save_video(frames: list[np.ndarray], output_path: str, fps: int = 30):
        """
        Saves a list of RGB numpy frames to a video file via imageio-ffmpeg.

        Args:
            frames: List of (H, W, 3) numpy arrays in RGB format.
            output_path: Path to save the MP4 file.
            fps: Frames per second.
        """
        if not frames:
            print("Warning: No frames provided to save_video.")
            return

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Saving video with {len(frames)} frames to {output_path}...")

        # imageio expects RGB uint8 — no color conversion needed (unlike OpenCV).
        frames_u8 = [np.asarray(f, dtype=np.uint8) for f in frames]
        imageio.mimwrite(
            output_path,
            frames_u8,
            fps=fps,
            codec="libx264",
            output_params=["-preset", "medium"],
        )
        print(f"Video successfully saved to {output_path}")
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize via Inheritance
    # Using from_pretrained inherited from DepthAnything3 (via PyTorchModelHubMixin)
    # or standard __init__ if you have local weights/configs setup.
    # If loading from a hub path:
    model = GeometryModel.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    model = model.to(device)

    # 2. Data Loading using the new Helper
    colmap_path = "/path/to/DL3DV-10K-Sample/<scene_hash>/colmap/"
    
    # We pass 20 to mimic [:110:20] roughly, though precise slicing logic 
    # might vary slightly depending on file sorting.
    data = GeometryModel.load_colmap_data(colmap_path, image_step=3, cutoff=8)

    # 3. Inference
    # Note: We pass numpy arrays directly. The parent class `inference` method 
    # handles the conversion to Tensor and device movement internally.
    prediction, scale = model.inference(
        image=data['images'],
        
        # Frame Extrinsics/Intrinsics (Input)
        extrinsics=data['extrinsics'],
        intrinsics=data['intrinsics'],

        # Settings from your script
        process_res=504,
        process_res_method="upper_bound_resize",
        infer_gs=True,
        align_to_input_ext_scale=False,
        export_kwargs={
            "gs_video": {
                "trj_mode": "original",
                "chunk_size": 1,
            }
        }
    )

    visualize_cameras_plotly(data["render_exts"], prediction.extrinsics)

    data['render_exts'] = model.scale_render_extrinsics(
        data['render_exts'],
        scale=scale,
    )

    data['render_ixts'] = model.scale_render_intrinsics(
        data['render_ixts'],
        (280, 504),
    )

    visualize_cameras_plotly(data['render_exts'], prediction.extrinsics)

    color, depth, alpha = GeometryModel.get_3dgs_renders(
        prediction,
        extrinsics=torch.from_numpy(data['render_exts']).unsqueeze(0).to(device),
        intrinsics=torch.from_numpy(data['render_ixts']).unsqueeze(0).to(device),
        out_image_hw=(280, 504),
        chunk_size=1,
        color_mode="RGB+ED",
    )
    print(f"Rendered color shape: {color.shape}, depth shape: {depth.shape}")

    # GPU forward-warp rendering (replaces old CPU render_novel_views)
    print("Starting GPU Novel View Synthesis...")
    render_exts_t = torch.from_numpy(data['render_exts']).float().to(device)
    render_ixts_t = torch.from_numpy(data['render_ixts']).float().to(device)
    pred_depth_t = torch.from_numpy(prediction.depth).float().to(device)
    pred_exts_t = torch.from_numpy(
        np.concatenate([prediction.extrinsics, np.tile(np.array([0, 0, 0, 1]), (len(prediction.extrinsics), 1, 1))], axis=1)
    ).float().to(device) if prediction.extrinsics.shape[1] == 3 else torch.from_numpy(prediction.extrinsics).float().to(device)
    pred_ixts_t = torch.from_numpy(prediction.intrinsics).float().to(device)

    # Source images as (T_in, 3, H, W) float [0,1]
    source_imgs_t = torch.from_numpy(prediction.processed_images).float().to(device) / 255.0
    source_imgs_t = source_imgs_t.permute(0, 3, 1, 2)  # (T, 3, H, W)

    T_src = pred_depth_t.shape[0]
    warp_result = GeometryModel.render_views_gpu(
        source_depth=pred_depth_t,
        source_w2c=pred_exts_t,
        source_intrinsics=pred_ixts_t,
        target_w2c=render_exts_t,
        target_intrinsics=render_ixts_t,
        source_images=source_imgs_t,
        foreground_masking=False,
        cameraray_filtering=False,
    )
    # Convert warped images to uint8 numpy frames for video
    warped_imgs = warp_result["images"]  # (T_target, 3, H, W)
    frames = [
        (warped_imgs[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        for i in range(warped_imgs.shape[0])
    ]

    # Save
    output_video_path = os.path.join("./gs_video", "rendered_trajectory.mp4")
    model.save_video(frames, output_video_path, fps=30)
    
    
    vis_depth = "hcat"

    VIDEO_QUALITY_MAP = {
        "low": {"crf": "28", "preset": "veryfast"},
        "medium": {"crf": "23", "preset": "medium"},
        "high": {"crf": "18", "preset": "slow"},
    }

    # save as video
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP["high"]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP["high"]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]  # best compatibility

    os.makedirs(os.path.join("", "gs_video"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        if vis_depth is not None:
            depth_i = vis_depth_map_tensor(depth[0])
            cat_fn = hcat if vis_depth == "hcat" else vcat
            video_i = torch.stack([cat_fn(c, d) for c, d in zip(video_i, depth_i)])
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()

        fps = 24
        output_name = f"{idx:04d}_debug"
        save_path = os.path.join("", f"gs_video/{output_name}.mp4")
        imageio.mimwrite(
            save_path,
            frames,
            fps=fps,
            codec="libx264",
            output_params=ffmpeg_params,
        )
    
    

