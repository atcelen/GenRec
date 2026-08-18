import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.io as io
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import trimesh
import cv2
import einops
import h5py
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PerspectiveCameras
)

from genrec.utils.typing import *
from genrec.utils.cam_ops import get_plucker_rays, get_ray_directions
from genrec.utils.logger import setup_logger
from genrec.models.geo_model import GeometryModel
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.specs import Prediction

logging = setup_logger(__name__)


def _resolve_scene_path(base_dir, scene_file, exts):
    """Resolve a scene-list entry to an on-disk file.

    Scene lists hold bare scene ids (e.g. ``000c3ab189999a83``); the storage
    extension is appended here. Entries that already include an extension, or
    point to an existing file, are used as-is.
    """
    path = os.path.join(str(base_dir), str(scene_file))
    if os.path.exists(path):
        return path
    for ext in exts:
        candidate = path + ext
        if os.path.exists(candidate):
            return candidate
    return path  # let the caller raise with the original name


class RealEstate10KDataset(Dataset):
    def __init__(
        self,
        data_path,
        T_in : int = 3,
        T_out : int = 1,
        is_train: bool = None,
        sample_criterion : Literal["sequential", "random", "equidistant"] = "sequential",
        step_size: int = 1,
    ):
        self.data_path = Path(data_path)
        self.pose_path = Path(data_path) / "train_test_split" / "test"
        self.frame_path = Path(data_path) / "frames"

        self.T_in = T_in
        self.T_out = T_out

        self.is_train = is_train
        if self.is_train is None:
            self.scene_paths = self.get_scene_paths()
        elif self.is_train:
            self.scene_paths = self.read_split_txt(self.data_path / "train_scenes.txt")
        else:
            self.scene_paths = self.read_split_txt(self.data_path / "test_scenes.txt")
        print(f"Found {len(self.scene_paths)} scenes out of {len(os.listdir(self.pose_path))}.")

        self.sample_criterion = sample_criterion
        self.step_size = step_size

    def read_split_txt(self, split_file: str) -> List[str]:
        with open(split_file, "r") as f:
            scene_ids = [line.strip() for line in f.readlines()]
        return scene_ids
    
    def get_scene_paths(self) -> List[str]:
        scene_paths = []
        for scene_txt in tqdm(os.listdir(self.pose_path), desc="Getting scene paths"):
            if not scene_txt.endswith('.txt'):
                continue
            scene_id = scene_txt[:-4]
            video_dir = self.frame_path / scene_id
            if video_dir.exists():
                scene_paths.append(scene_id)

        return scene_paths

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_id = self.scene_paths[idx]
        
        pose_path = self.pose_path / f"{scene_id}.txt"
        frame_dir = self.frame_path / scene_id

        poses = self.read_pose_file(pose_path)
        images_dict = self.read_image_files(frame_dir)

        assert all(poses["timeStamps"] == poses["timeStamps"].sort().values), f"{scene_id} : Timestamps are not sorted: {timestamps}"
        assert len(set(poses["timeStamps"].tolist())) == len(poses["timeStamps"]), f"{scene_id} : Timestamps are not unique: {timestamps}"

        total_frames = poses["extrinsics"].shape[0]
        if total_frames != len(images_dict.keys()):
            logging.warning(f"Number of poses and images do not match for scene {scene_id}.")
        valid_frame_ids = np.arange(total_frames)

        if total_frames < self.T_in + self.T_out:
            logging.warning(f"Scene {scene_id} does not have enough frames ({total_frames}) for T_in + T_out = {self.T_in + self.T_out}. Skipping...")
            return (None, None, None)

        if self.is_train:
            if self.sample_criterion == "sequential":
                start_idx = np.random.randint(0, total_frames - (self.T_in + self.T_out) + 1)
                valid_frame_ids = np.arange(start_idx, start_idx + self.T_in + self.T_out)
            elif self.sample_criterion == "random":
                valid_frame_ids = np.random.choice(total_frames, self.T_in + self.T_out, replace=False)
                valid_frame_ids = np.sort(valid_frame_ids)
            elif self.sample_criterion == "equidistant":
                valid_frame_ids = np.linspace(0, total_frames - 1, self.T_in + self.T_out, dtype=int)
            else:
                raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")
        else:
            valid_frame_ids = np.arange(self.T_in + self.T_out) # Always take the first T_in + T_out frames for evaluation

        extrinsics = poses["extrinsics"][valid_frame_ids]  # (T_in + T_out, 3, 4)
        intrinsics = poses["intrinsics"]  # (4)
        timestamps = poses["timeStamps"][valid_frame_ids]  # (T_in + T_out,)

        # Gather images for the requested timestamps
        missing_ts = [ts.item() for ts in timestamps if ts.item() not in images_dict]
        assert not missing_ts, (
            f"Timestamps not found in scene {scene_id}: {missing_ts} "
            f"(available: {list(images_dict.keys())})."
        )

        # Load images in order
        frame_ids = [ts.item() for ts in timestamps]
        images = torch.stack([images_dict[fid] for fid in frame_ids], dim=0)

        # normalize rgb to [-1, 1]
        images = images.clip(0.0, 1.0) * 2.0 - 1.0

        extrinsics = torch.cat([extrinsics, torch.zeros((self.T_in + self.T_out, 1, 4))], dim=1)
        extrinsics[:, 3, 3] = 1.0  # Make it 4x4

        # Switch from world-to-camera to camera-to-world
        extrinsics = torch.inverse(extrinsics)

        # Update Poses
        relative_poses = torch.inverse(extrinsics[0:1]).repeat(self.T_in + self.T_out, 1, 1) @ extrinsics
        poses = relative_poses

        # Normalize Poses
        poses = self.normalize_poses(poses)

        return (scene_id, images, poses, intrinsics)
    
    def read_pose_file(self, pose_file: str) -> Dict[str, torch.Tensor]:
        """
        Example parser for a simple pose file format:
        Each line: 12 floats = 3x4 camera extrinsic matrix (row-major).
        Returns: (N, 3, 4) tensor.
        Adapt this for your actual RealEstate10K metadata format.
        """
        extrinsics_list = []
        timeStamps_list = []
        with open(pose_file, "r") as f:
            videoPathURL = f.readline().rstrip()
            for l in f.readlines():
                line = l.split(' ')
                timeStamp = int(line[0])  # ns
                intrinsics = [float(i) for i in line[1:7]]
                pose = [float(i) for i in line[7:19]]
                extrinsics_list.append(pose)
                timeStamps_list.append(timeStamp)
        
        extrinsics_tensor = torch.tensor(extrinsics_list, dtype=torch.float32).view(-1, 3, 4)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        timeStamps_tensor = torch.tensor(timeStamps_list, dtype=torch.int64)

        return {
            "url": videoPathURL,
            "extrinsics": extrinsics_tensor,
            "intrinsics": intrinsics_tensor,
            "timeStamps": timeStamps_tensor,
        }


    def read_image_files(self, frame_dir: str, extensions=(".jpg", ".jpeg", ".png")) -> List[str]:
        """
        List and sort all image files in a directory.
        """
        files = [
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.lower().endswith(extensions)
        ]
        files.sort()
        if len(files) == 0:
            raise ValueError(f"No image files found in {frame_dir}")
        
        images_dict = {}
        for p in files:
            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = torch.from_numpy(np.array(im)).float() / 255.0  # [0,1]
                arr = arr.permute(2, 0, 1)  # C,H,W
                images_dict[int(os.path.basename(p).split('.')[0])] = arr

        for k in images_dict:
            images_dict[k] = torch.as_tensor(images_dict[k])
        return images_dict
    
    def calc_scale_mat(
        self, raw_poses: Float[Tensor, "N 4 4"], depth_range: float = None, offset_center: bool = False
    ) -> Tuple[Float[Tensor, "4 4"], float]:
        """
        raw_poses: [N, 4, 4], camera to world poses
        depth_range: maximum depth range of each camera
        """
        c2w_poses = raw_poses
        min_vertices = c2w_poses[:, :3, 3].min(dim=0).values
        max_vertices = c2w_poses[:, :3, 3].max(dim=0).values

        if offset_center:
            center = (min_vertices + max_vertices) / 2.0
        else:
            center = torch.zeros(3, dtype=torch.float32)

        # convert the scene to [-1, 1] unit cube
        # scale = 2. / (torch.max(max_vertices - min_vertices) + 2 * depth_range)
        # TODO: use above scale, but for now, use the depth scale
        positions_scale = torch.max(max_vertices - min_vertices)
        if depth_range is not None:
            # scale = 2.0 / positions_scale if positions_scale > (2 * depth_range) else 2.0 / (2 * depth_range)
            scale = 2.0 / (2 * depth_range)
        else:
            scale = 2.0 / positions_scale

        # we should normalized to unit cube
        scale_mat = torch.eye(4, dtype=torch.float32)
        scale_mat[:3, 3] = -center
        scale_mat[:3] *= scale

        return scale_mat, scale

    def normalize_poses(self, raw_poses: Float[Tensor, "N 4 4"]) -> Float[Tensor, "N 4 4"]:
        """
        raw_poses: [N, 4, 4], camera to world poses
        """
        scale_mat, _ = self.calc_scale_mat(raw_poses)
        for pose_idx in range(raw_poses.shape[0]):
            raw_pose = scale_mat @ raw_poses[pose_idx]
            R_c2w = (raw_pose[:3, :3]).numpy()
            q_c2w = trimesh.transformations.quaternion_from_matrix(R_c2w)
            q_c2w = trimesh.transformations.unit_vector(q_c2w)
            R_c2w = trimesh.transformations.quaternion_matrix(q_c2w)[:3, :3]
            raw_pose[:3, :3] = torch.from_numpy(R_c2w)
            raw_poses[pose_idx] = raw_pose
        return raw_poses
    
    import torch

    def create_coordinate_map(self, depth, K, c2w):
        """
        Generates a World Coordinate Map from Depth and Camera Params.

        Args:
            depth: Tensor of shape (B, 1, H, W). Depth map.
            K: Tensor of shape (B, 3, 3). Intrinsic matrix.
            c2w: Tensor of shape (B, 4, 4). Camera-to-World extrinsic matrix.

        Returns:
            coord_map: Tensor of shape (B, 3, H, W). 
                       The (x, y, z) world coordinates for each pixel.
                       Values are normalized to [-1, 1] if bounds are provided (optional step).
        """
        B, _, H, W = depth.shape
        device = depth.device

        # 1. Create a meshgrid of pixel coordinates (u, v)
        # i corresponds to v (y-axis), j corresponds to u (x-axis)
        i, j = torch.meshgrid(torch.arange(H, device=device), 
                              torch.arange(W, device=device), indexing='ij')

        # Shape: (B, 3, H*W) -> Homogeneous coordinates [u, v, 1]
        # We repeat the grid for the batch size
        ones = torch.ones_like(i)
        uv_grid = torch.stack([j, i, ones], dim=0).float() # (3, H, W)
        uv_grid = uv_grid.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1) # (B, 3, H*W)

        # 2. Unproject to Camera Space
        # P_cam = Z * K_inv * uv

        # Invert K
        K_inv = torch.inverse(K) # (B, 3, 3)

        # Apply K_inv to the pixel grid
        # (B, 3, 3) @ (B, 3, H*W) -> (B, 3, H*W)
        cam_coords_normalized = torch.bmm(K_inv, uv_grid)

        # Flatten depth to multiply: (B, 1, H*W)
        depth_flat = depth.reshape(B, 1, -1)

        # Scale by depth to get actual 3D positions in camera frame
        cam_coords = cam_coords_normalized * depth_flat # (B, 3, H*W)

        # 3. Transform to World Space
        # P_world = R * P_cam + t

        # We need to treat P_cam as homogeneous (x, y, z, 1) to use 4x4 c2w matrix easily
        ones_row = torch.ones((B, 1, H * W), device=device)
        cam_coords_hom = torch.cat([cam_coords, ones_row], dim=1) # (B, 4, H*W)

        # Apply Camera-to-World transformation
        # (B, 4, 4) @ (B, 4, H*W) -> (B, 4, H*W)
        world_coords_hom = torch.bmm(c2w, cam_coords_hom)

        # Extract just x, y, z
        world_coords = world_coords_hom[:, :3, :] # (B, 3, H*W)

        # Reshape back to image format (B, 3, H, W)
        coord_map = world_coords.reshape(B, 3, H, W)

        return coord_map
    
    def normalize_coords(self, coord_map, scene_min, scene_max):
        """
        Normalizes coordinates to [-1, 1] based on global scene bounds.
        scene_min/max should be tensors or lists: [min_x, min_y, min_z]
        """
        # Simple Min-Max normalization
        # Formula: 2 * (x - min) / (max - min) - 1

        # Reshape bounds for broadcasting (1, 3, 1, 1)
        min_v = torch.tensor(scene_min, device=coord_map.device).view(1, 3, 1, 1)
        max_v = torch.tensor(scene_max, device=coord_map.device).view(1, 3, 1, 1)

        # norm_map = 2 * (coord_map - min_v) / (max_v - min_v) - 1
        norm_map = (coord_map - min_v) / (max_v - min_v)

        # Clip values to handle outliers (optional but recommended)
        norm_map = torch.clamp(norm_map, -1, 1)

        return norm_map



class RealEstate10KDatasetWithDA3(RealEstate10KDataset):
    def __init__(
        self,
        data_path: str,
        geometry_model: GeometryModel,
        T_in: int = 3,
        T_out: int = 1,
        is_train: bool = None,
        sample_criterion: Literal["sequential", "random", "equidistant"] = "sequential",
        step_size: int = 1,
    ):
        super().__init__(
            data_path=data_path,
            T_in=T_in,
            T_out=T_out,
            is_train=is_train,
            sample_criterion=sample_criterion,
            step_size=step_size,
        )

        self.geometry_model = geometry_model
    
    def __getitem__(self, idx):
        scene_id = self.scene_paths[idx]
        
        pose_path = self.pose_path / f"{scene_id}.txt"
        frame_dir = self.frame_path / scene_id

        poses = self.read_pose_file(pose_path)

        try:
            assert all(poses["timeStamps"] == poses["timeStamps"].sort().values), f"{scene_id} : Timestamps are not sorted: {poses['timeStamps']}"
            assert len(set(poses["timeStamps"].tolist())) == len(poses["timeStamps"]), f"{scene_id} : Timestamps are not unique: {poses['timeStamps']}"
        except AssertionError as e:
            logging.warning(str(e))
            return None

        total_frames = poses["extrinsics"].shape[0]

        if total_frames < (self.T_in + self.T_out) * self.step_size:
            logging.warning(f"Scene {scene_id} does not have enough frames ({total_frames}) for T_in + T_out = {self.T_in + self.T_out}. Skipping...")
            return None

        if self.sample_criterion == "sequential":
            start_idx = np.random.randint(0, total_frames - (self.T_in + self.T_out) * self.step_size + 1)
            if self.is_train:
                valid_frame_ids = np.arange(start_idx, start_idx + (self.T_in + self.T_out) * self.step_size, self.step_size)
            else:
                valid_frame_ids = np.arange(0, (self.T_in + self.T_out) * self.step_size, self.step_size) # Always take the first T_in + T_out frames for evaluation
        elif self.sample_criterion == "random":
            valid_frame_ids = np.random.choice(total_frames, self.T_in + self.T_out, replace=False)
            valid_frame_ids = np.sort(valid_frame_ids)
        elif self.sample_criterion == "equidistant":
            valid_frame_ids = np.linspace(0, total_frames - 1, self.T_in + self.T_out, dtype=int)
        else:
            raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")

        timestamps = poses["timeStamps"][valid_frame_ids]  # (T_in + T_out)
        image_paths = [os.path.join(frame_dir, f"{ts.item()}.png") for ts in timestamps]

        try:
            assert all([os.path.exists(p) for p in image_paths]), f"Some image paths do not exist for scene {scene_id}: {image_paths}"
        except AssertionError as e:
            logging.warning(str(e))
            return None

        prediction, scale = self.geometry_model.inference(
            image=image_paths,
            process_res=560,
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

        # logging.info(f"Timestamps: {timestamps.tolist()}")

        pred_extrinsics = as_homogeneous(prediction.extrinsics)
        pred_intrinsics = prediction.intrinsics

        ray_directions = torch.stack([
            get_ray_directions(
                H=prediction.processed_images.shape[1],
                W=prediction.processed_images.shape[2],
                focal=(pred_intrinsics[i, 0, 0], pred_intrinsics[i, 1, 1]),
                principal=(pred_intrinsics[i, 0, 2], pred_intrinsics[i, 1, 2]),
            ) for i in range(pred_intrinsics.shape[0])
        ], dim=0)  # (T_in + T_out, H, W, 3)

        c2w = torch.inverse(torch.from_numpy(pred_extrinsics).float())

        plucker_rays = get_plucker_rays(
            directions=ray_directions,
            c2w=c2w,
            keepdim=True,
        )
        plucker_rays = einops.rearrange(plucker_rays, "B H W C -> B C H W")

        # logging.info(f"Scene ID: {scene_id}")
        if scale is None: scale = 1.0

        # Save unscaled extrinsics before in-place scaling (GPU warp needs unscaled source w2c)
        unscaled_extrinsics = pred_extrinsics.copy()

        render_exts = self.geometry_model.scale_render_extrinsics(
            pred_extrinsics,
            scale=scale,
        )
        render_ixts = self.geometry_model.scale_render_intrinsics(
            pred_intrinsics,
            (prediction.processed_images.shape[1], prediction.processed_images.shape[2]),
        )

        # Remove the first target view from the prediction for LOO evaluation
        loo_prediction = Prediction(
            processed_images=prediction.processed_images[: -self.T_out],
            depth=prediction.depth[: -self.T_out],
            sky=prediction.sky[: -self.T_out] if prediction.sky is not None else None,
            conf=prediction.conf[: -self.T_out],
            extrinsics=pred_extrinsics[: -self.T_out],
            intrinsics=pred_intrinsics[: -self.T_out],
            is_metric=prediction.is_metric,
            gaussians=prediction.gaussians,
            aux=prediction.aux,
            scale_factor=prediction.scale_factor,
        )

        gs_renders, gs_depth, gs_alpha, gs_blur_map = GeometryModel.get_3dgs_renders(
            loo_prediction,
            extrinsics=torch.from_numpy(render_exts).unsqueeze(0).to(self.geometry_model.device),
            intrinsics=torch.from_numpy(render_ixts).unsqueeze(0).to(self.geometry_model.device),
            out_image_hw=(prediction.processed_images.shape[1], prediction.processed_images.shape[2]),
            chunk_size=1,
            color_mode="RGB+ED"
        )

        # GPU forward-warp rendering (LOO input views → all views)
        source_imgs = torch.from_numpy(
            prediction.processed_images[:-self.T_out]
        ).float().permute(0, 3, 1, 2) / 255.0  # (T_in, 3, H, W) float [0,1]

        warp_result = GeometryModel.render_views_gpu(
            source_depth=torch.from_numpy(prediction.depth[:-self.T_out]).float(),
            source_w2c=torch.from_numpy(unscaled_extrinsics[:-self.T_out]).float(),
            source_intrinsics=torch.from_numpy(pred_intrinsics[:-self.T_out]).float(),
            target_w2c=torch.from_numpy(unscaled_extrinsics).float(),
            target_intrinsics=torch.from_numpy(pred_intrinsics).float(),
            source_images=source_imgs,
            foreground_masking=False,
            cameraray_filtering=False,
        )
        pc_renders = warp_result["images"]  # (T_in+T_out, 3, H, W), float [0,1]

        coord_maps = self.create_coordinate_map(
            depth=torch.from_numpy(prediction.depth).unsqueeze(1),  # (B, 1, H, W)
            K=torch.from_numpy(pred_intrinsics),  # (B, 3, 3)
            c2w=torch.inverse(torch.from_numpy(unscaled_extrinsics).float()),  # (B, 4, 4)
        )

        scene_min = torch.amin(coord_maps.permute(0, 2, 3, 1), dim=(0, 1, 2)).tolist()
        scene_max = torch.amax(coord_maps.permute(0, 2, 3, 1), dim=(0, 1, 2)).tolist()

        coord_maps = self.normalize_coords(
            coord_map=coord_maps,
            scene_min=scene_min,
            scene_max=scene_max,
        )

        all_images = torch.from_numpy(prediction.processed_images).float().permute(0, 3, 1, 2)  # (T_in + T_out, 3, H, W)
        all_images = all_images.clip(0, 255) / 255.0
        
        # coord_maps = [torch.from_numpy(f) for f in coord_maps]
        # coord_maps = torch.stack(coord_maps, dim=0)  # (T_in + T_out, H, W, 3)
        # coord_maps = coord_maps.permute(0, 3, 1, 2)  # (T_in + T_out, 3, H, W)
        
        gs_renders = gs_renders.clip(0, 1)
        gs_renders = gs_renders.squeeze(0)

        gs_depth = gs_depth.squeeze(0)
        gs_alpha = gs_alpha.squeeze(0)
        gs_alpha = gs_alpha.permute(0, 3, 1, 2)  # (T_in + T_out, 1, H, W)
        gs_blur_map = gs_blur_map.squeeze(0)


        # raise Exception(f"Shapes: {gs_renders.shape}, {gs_depth.shape}, {gs_alpha.shape}")

        target_gs_renders = gs_renders[-self.T_out:]  # (T_out, 3, H, W)
        target_gs_depth = gs_depth[-self.T_out:]  # (T_out, 3, H, W)
        target_gs_alpha = gs_alpha[-self.T_out:]  # (T_out, 3, H, W)
        target_gs_blur_map = gs_blur_map[-self.T_out:]  # (T_out, 3, H, W)
        target_pc_renders = pc_renders[-self.T_out:]  # (T_out, 3, H, W)
        target_images = all_images[-self.T_out:] # (T_out, H, W, 3)
        target_plucker_rays = plucker_rays[-self.T_out:]  # (T_out, C, H, W)
        target_coord_map = coord_maps[-self.T_out:]  # (T_out, 3, H, W)

        input_gs_renders = gs_renders[: self.T_in]  # (T_in, 3, H, W)
        input_gs_depth = gs_depth[: self.T_in]  # (T_in, H, W)
        input_gs_alpha = gs_alpha[: self.T_in]  # (T_in, H, W, 1)
        input_gs_blur_map = gs_blur_map[: self.T_in]  # (T_in, H, W, 1)
        input_pc_renders = pc_renders[: self.T_in]  # (T_in, 3, H, W)
        input_images = all_images[: self.T_in] # (T_in, H, W, 3)
        input_plucker_rays = plucker_rays[: self.T_in]  # (T_in, C, H, W)
        input_coord_map = coord_maps[: self.T_in]  # (T_in, 3, H, W)

        inputs = {
            "images": input_images,
            "gs_renders": input_gs_renders,
            "gs_depth": input_gs_depth,
            "gs_alpha": input_gs_alpha,
            "gs_blur_map": input_gs_blur_map,
            "pc_renders": input_pc_renders,
            "plucker_rays": input_plucker_rays,
            "coord_map": input_coord_map,
        }

        targets = {
            "images": target_images,
            "gs_renders": target_gs_renders,
            "gs_depth": target_gs_depth,
            "gs_alpha": target_gs_alpha,
            "gs_blur_map": target_gs_blur_map,
            "pc_renders": target_pc_renders,
            "plucker_rays": target_plucker_rays,
            "coord_map": target_coord_map,
        }

        return {
            "scene_id": scene_id,
            "inputs": inputs,
            "targets": targets,
        }

    
def collate_fn_ignore_none(batch):
    # 'batch' is a list of tuples: [(scene_id, images, rays), (None, None, None), ...]
    
    # 1. Filter out the Nones
    batch = [item for item in batch if item[0] is not None]
    
    # 2. Handle case where entire batch is None
    if len(batch) == 0:
        return None
    
    # 3. Use default_collate to stack the remaining valid samples
    return default_collate(batch)

def collate_fn_ignore_none_da3(batch):
    # 'batch' is a list of dicts: [{"scene_id": ..., "inputs": ..., "targets": ...}, None, ...]
    
    # 1. Filter out the Nones
    batch = [item for item in batch if item is not None]
    
    # 2. Handle case where entire batch is None
    if len(batch) == 0:
        return None
    
    # 3. Use default_collate to stack the remaining valid samples
    return default_collate(batch)

def visualize_predictions(
    frames: Float[Tensor, "N 3 H W"],
    color: Float[Tensor, "N 3 H W"],
    image_paths: List[str],
    prediction: Prediction,
    ):
    import matplotlib.pyplot as plt
    
    def _to_numpy_img(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 4:
                img = img[0]
            if img.shape[0] in (3, 4):
                img = img.permute(1, 2, 0)
            img = img[..., :3]
            img = img.numpy()
        return img
    
    num_frames = frames.shape[0]
    fig, axes = plt.subplots(num_frames, 5, figsize=(15, 3 * num_frames))
    if num_frames == 1:
        axes = axes[None, :]  # ensure 2D indexing
    for i in range(num_frames):
        axes[i, 0].imshow(_to_numpy_img(color[i]))
        axes[i, 0].set_title("3DGS Color")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(_to_numpy_img(frames[i]))
        axes[i, 1].set_title("Rendered Frame")
        axes[i, 1].axis("off")
        with Image.open(image_paths[i]) as im:
            orig_img = torch.from_numpy(np.array(im.convert("RGB"))).float() / 255.0
        orig_img = orig_img.permute(2, 0, 1)  # (3, H, W)
        axes[i, 2].imshow(_to_numpy_img(orig_img))
        axes[i, 2].set_title("Input Image")
        axes[i, 2].axis("off")
        depth_map_rgb = cv2.applyColorMap(
        (prediction.depth[i] / prediction.depth[i].max() * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
        )
        depth_map_rgb = torch.from_numpy(depth_map_rgb).float() / 255.0
        depth_map_rgb = depth_map_rgb.permute(2, 0, 1)  # (3, H, W)
        axes[i, 3].imshow(_to_numpy_img(depth_map_rgb))
        axes[i, 3].set_title("Depth Map")
        axes[i, 3].axis("off")
        depth_conf_map = cv2.applyColorMap(
        (prediction.conf[i] / prediction.conf[i].max() * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
        )
        depth_conf_map = torch.from_numpy(depth_conf_map).float() / 255.0
        depth_conf_map = depth_conf_map.permute(2, 0, 1)  # (3, H, W)
        axes[i, 4].imshow(_to_numpy_img(depth_conf_map))
        axes[i, 4].set_title("Depth Conf Map")
        axes[i, 4].axis("off")
    plt.tight_layout()
    plt.show()
    plt.close(fig)

class RealEstate10KDatasetWithDA3Euler(RealEstate10KDatasetWithDA3):
    def __init__(
        self,
        data_path: str,
        train_scenes_path: str,
        test_scenes_path: str,
        geometry_model: GeometryModel,
        T_in: int = 3,
        T_out: int = 1,
        is_train: bool = None,
        sample_criterion: Literal["sequential", "sequential-random", "random", "equidistant"] = "sequential",
        step_size: int = 1,
        max_scenes: int = None,
    ):
        self.data_path = Path(data_path)

        self.T_in = T_in
        self.T_out = T_out

        self.is_train = is_train
        if self.is_train is None:
            self.scene_paths = self.read_split_txt(self.data_path / "train_scenes.txt", prefix=train_scenes_path) + self.read_split_txt(self.data_path / "val_scenes.txt", prefix=train_scenes_path)
        elif self.is_train:
            self.scene_paths = self.read_split_txt(self.data_path / "train_scenes.txt", prefix=train_scenes_path)
        else:
            self.scene_paths = self.read_split_txt(self.data_path / "val_scenes.txt", prefix=train_scenes_path)
        
        # Limit number of scenes if specified
        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]
        
        print(f"Found {len(self.scene_paths)} scenes (max_scenes={max_scenes}).")

        self.geometry_model = geometry_model

        self.sample_criterion = sample_criterion
        self.step_size = step_size
    
    def convert_image(self, image):
        """Convert list of bytes images to a numpy array of shape (N, H, W, 3)."""
        # return np.stack([np.array(Image.open(BytesIO(img.numpy().tobytes()))) for img in images])
        decoded = io.decode_image(image, mode="RGB").permute(1, 2, 0).numpy()
        return decoded  # H, W, 3
    
    def read_split_txt(self, split_file: str, prefix: str = None) -> List[str]:
        with open(split_file, "r") as f:
            scene_ids = [os.path.join(prefix, line.strip()) if prefix else line.strip() for line in f.readlines()]
        return scene_ids
    
    def __len__(self):
        return len(self.scene_paths)  # For testing purposes, we just return 1 sample
    
    def __getitem__(self, idx):
        scene_id_file = self.scene_paths[idx]

        data = torch.load(_resolve_scene_path(self.data_path, scene_id_file, (".torch", ".pt")), weights_only=True) # weights_only=True is safe

        images = [
            self.convert_image(torch.as_tensor(d[()]))
            for d in data["images"]
        ]
        
        total_frames = len(data['images'])

        if total_frames < (self.T_in + self.T_out) * self.step_size:
            logging.warning(f"Scene {scene_id_file} does not have enough frames ({total_frames}) for T_in + T_out = {self.T_in + self.T_out}. Skipping...")
            return None

        if self.sample_criterion == "sequential":
            start_idx = np.random.randint(0, total_frames - (self.T_in + self.T_out) * self.step_size + 1)
            valid_frame_ids = np.arange(start_idx, start_idx + (self.T_in + self.T_out) * self.step_size, self.step_size)
        elif self.sample_criterion == "sequential-random":
            sample_step_size = np.random.randint(1, self.step_size + 1)
            start_idx = np.random.randint(0, total_frames - (self.T_in + self.T_out) * sample_step_size + 1)
            valid_frame_ids = np.arange(start_idx, start_idx + (self.T_in + self.T_out) * sample_step_size, sample_step_size)
        elif self.sample_criterion == "random":
            valid_frame_ids = np.random.choice(total_frames, self.T_in + self.T_out, replace=False)
        elif self.sample_criterion == "equidistant":
            valid_frame_ids = np.linspace(0, total_frames - 1, self.T_in + self.T_out, dtype=int)
        else:
            raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")

        images = [images[int(i)] for i in valid_frame_ids]  # (T_in + T_out, H, W, 3)

        prediction, scale = self.geometry_model.inference(
            image=images,
            process_res=616,
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

        # logging.info(f"Timestamps: {timestamps.tolist()}")

        pred_extrinsics = as_homogeneous(prediction.extrinsics)
        pred_intrinsics = prediction.intrinsics

        ray_directions = torch.stack([
            get_ray_directions(
                H=prediction.processed_images.shape[1],
                W=prediction.processed_images.shape[2],
                focal=(pred_intrinsics[i, 0, 0], pred_intrinsics[i, 1, 1]),
                principal=(pred_intrinsics[i, 0, 2], pred_intrinsics[i, 1, 2]),
            ) for i in range(pred_intrinsics.shape[0])
        ], dim=0)  # (T_in + T_out, H, W, 3)

        c2w = torch.inverse(torch.from_numpy(pred_extrinsics).float())

        plucker_rays = get_plucker_rays(
            directions=ray_directions,
            c2w=c2w,
            keepdim=True,
        )
        plucker_rays = einops.rearrange(plucker_rays, "B H W C -> B C H W")

        # logging.info(f"Scene ID: {scene_id}")
        if scale is None: scale = 1.0

        # Save unscaled extrinsics before in-place scaling (GPU warp needs unscaled source w2c)
        unscaled_extrinsics = pred_extrinsics.copy()

        render_exts = self.geometry_model.scale_render_extrinsics(
            pred_extrinsics,
            scale=scale,
        )
        render_ixts = self.geometry_model.scale_render_intrinsics(
            pred_intrinsics,
            (prediction.processed_images.shape[1], prediction.processed_images.shape[2]),
        )

        # Remove the first target view from the prediction for LOO evaluation
        loo_prediction = Prediction(
            processed_images=prediction.processed_images[: -self.T_out],
            depth=prediction.depth[: -self.T_out],
            sky=prediction.sky[: -self.T_out] if prediction.sky is not None else None,
            conf=prediction.conf[: -self.T_out],
            extrinsics=pred_extrinsics[: -self.T_out],
            intrinsics=pred_intrinsics[: -self.T_out],
            is_metric=prediction.is_metric,
            gaussians=prediction.gaussians,
            aux=prediction.aux,
            scale_factor=prediction.scale_factor,
        )

        gs_renders = torch.zeros((1, prediction.processed_images.shape[0] - self.T_out, prediction.processed_images.shape[1], prediction.processed_images.shape[2], 3))
        gs_depth = torch.zeros((1, prediction.processed_images.shape[0] - self.T_out, prediction.processed_images.shape[1], prediction.processed_images.shape[2]))
        gs_alpha = torch.zeros((1, prediction.processed_images.shape[0] - self.T_out, prediction.processed_images.shape[1], prediction.processed_images.shape[2], 1))
        gs_blur_map = torch.zeros((1, prediction.processed_images.shape[0] - self.T_out, prediction.processed_images.shape[1], prediction.processed_images.shape[2]))

        # GPU forward-warp rendering (LOO input views → all views)
        source_imgs = torch.from_numpy(
            prediction.processed_images[:-self.T_out]
        ).float().permute(0, 3, 1, 2) / 255.0  # (T_in, 3, H, W) float [0,1]

        warp_result = GeometryModel.render_views_gpu(
            source_depth=torch.from_numpy(prediction.depth[:-self.T_out]).float(),
            source_w2c=torch.from_numpy(unscaled_extrinsics[:-self.T_out]).float(),
            source_intrinsics=torch.from_numpy(pred_intrinsics[:-self.T_out]).float(),
            target_w2c=torch.from_numpy(unscaled_extrinsics).float(),
            target_intrinsics=torch.from_numpy(pred_intrinsics).float(),
            source_images=source_imgs,
            foreground_masking=False,
            cameraray_filtering=False,
        )
        pc_renders = warp_result["images"]  # (T_in+T_out, 3, H, W), float [0,1]

        all_images = torch.from_numpy(prediction.processed_images).float().permute(0, 3, 1, 2)  # (T_in + T_out, 3, H, W)
        all_images = all_images.clip(0, 255) / 255.0

        gs_renders = gs_renders.clip(0, 1)
        gs_renders = gs_renders.squeeze(0)

        gs_depth = gs_depth.squeeze(0)
        gs_alpha = gs_alpha.squeeze(0)
        gs_alpha = gs_alpha.permute(0, 3, 1, 2)  # (T_in + T_out, 1, H, W)
        gs_blur_map = gs_blur_map.squeeze(0)


        # raise Exception(f"Shapes: {gs_renders.shape}, {gs_depth.shape}, {gs_alpha.shape}")

        target_gs_renders = gs_renders[-self.T_out:]  # (T_out, 3, H, W)
        target_gs_depth = gs_depth[-self.T_out:]  # (T_out, 3, H, W)
        target_gs_alpha = gs_alpha[-self.T_out:]  # (T_out, 3, H, W)
        target_gs_blur_map = gs_blur_map[-self.T_out:]  # (T_out, 3, H, W)
        target_pc_renders = pc_renders[-self.T_out:]  # (T_out, 3, H, W)
        target_images = all_images[-self.T_out:] # (T_out, H, W, 3)
        target_plucker_rays = plucker_rays[-self.T_out:]  # (T_out, C, H, W)

        input_gs_renders = gs_renders[: self.T_in]  # (T_in, 3, H, W)
        input_gs_depth = gs_depth[: self.T_in]  # (T_in, H, W)
        input_gs_alpha = gs_alpha[: self.T_in]  # (T_in, H, W, 1)
        input_gs_blur_map = gs_blur_map[: self.T_in]  # (T_in, H, W, 1)
        input_pc_renders = pc_renders[: self.T_in]  # (T_in, 3, H, W)
        input_images = all_images[: self.T_in] # (T_in, H, W, 3)
        input_plucker_rays = plucker_rays[: self.T_in]  # (T_in, C, H, W)

        inputs = {
            "images": input_images,
            "gs_renders": input_gs_renders,
            "gs_depth": input_gs_depth,
            "gs_alpha": input_gs_alpha,
            "gs_blur_map": input_gs_blur_map,
            "pc_renders": input_pc_renders,
            "plucker_rays": input_plucker_rays,
        }

        targets = {
            "images": target_images,
            "gs_renders": target_gs_renders,
            "gs_depth": target_gs_depth,
            "gs_alpha": target_gs_alpha,
            "gs_blur_map": target_gs_blur_map,
            "pc_renders": target_pc_renders,
            "plucker_rays": target_plucker_rays,
        }

        return {
            "scene_id": scene_id_file.split('.')[0],
            "inputs": inputs,
            "targets": targets,
        }


class RealEstate10KDatasetRaw(RealEstate10KDataset):
    """
    Raw dataset for RealEstate10K that loads PNG files from disk (like RealEstate10KDatasetWithDA3)
    but does NOT perform geometry model inference.
    
    This dataset is designed to be used with models that have integrated
    geometry processing (GeometryModel inside the main model). It only
    loads and returns raw images, deferring all geometry computation to
    the model's forward pass for batched GPU processing.
    
    Data source: PNG files from frames/ directory, using pose files to get timestamps.
    
    Returns:
        Dict containing:
            - scene_id: str
            - raw_images: torch.Tensor of shape (T_in + T_out, H, W, 3) in [0, 255]
            - frame_indices: List[int] of selected frame indices
    """
    
    def __init__(
        self,
        data_path: str,
        T_in: int = 3,
        T_out: int = 1,
        is_train: bool = True,
        sample_criterion: Literal["sequential", "random", "equidistant"] = "sequential",
        step_size: int = 1,
    ):
        # Call parent __init__ to set up pose_path, frame_path, scene_paths, etc.
        super().__init__(
            data_path=data_path,
            T_in=T_in,
            T_out=T_out,
            is_train=is_train,
            sample_criterion=sample_criterion,
            step_size=step_size,
        )
        print(f"RealEstate10KDatasetRaw: Loaded {len(self.scene_paths)} scenes (is_train={is_train})")

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load a single image from path as numpy array (H, W, 3) uint8."""
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            return np.array(im)

    def __getitem__(self, idx):
        scene_id = self.scene_paths[idx]
        
        pose_path = self.pose_path / f"{scene_id}.txt"
        frame_dir = self.frame_path / scene_id
        
        # Read pose file to get timestamps
        try:
            poses = self.read_pose_file(pose_path)
        except Exception as e:
            logging.warning(f"Failed to read pose file for {scene_id}: {e}")
            return None
        
        # Validate timestamps
        try:
            assert all(poses["timeStamps"] == poses["timeStamps"].sort().values), \
                f"{scene_id}: Timestamps are not sorted"
            assert len(set(poses["timeStamps"].tolist())) == len(poses["timeStamps"]), \
                f"{scene_id}: Timestamps are not unique"
        except AssertionError as e:
            logging.warning(str(e))
            return None
        
        total_frames = poses["extrinsics"].shape[0]
        T_total = self.T_in + self.T_out
        
        # Check if scene has enough frames
        if total_frames < T_total * self.step_size:
            logging.warning(
                f"Scene {scene_id} has {total_frames} frames, "
                f"need {T_total * self.step_size} for T={T_total}, step={self.step_size}. Skipping."
            )
            return None
        
        # Select frame indices based on sampling criterion
        if self.sample_criterion == "sequential":
            start_idx = np.random.randint(0, total_frames - T_total * self.step_size + 1)
            if self.is_train:
                valid_frame_ids = np.arange(
                    start_idx,
                    start_idx + T_total * self.step_size,
                    self.step_size
                )
            else:
                # Always take the first frames for evaluation
                valid_frame_ids = np.arange(0, T_total * self.step_size, self.step_size)
        elif self.sample_criterion == "random":
            valid_frame_ids = np.random.choice(total_frames, T_total, replace=False)
            valid_frame_ids = np.sort(valid_frame_ids)
        elif self.sample_criterion == "equidistant":
            valid_frame_ids = np.linspace(0, total_frames - 1, T_total, dtype=int)
        else:
            raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")
        
        # Get timestamps for selected frames
        timestamps = poses["timeStamps"][valid_frame_ids]
        
        # Build image paths
        image_paths = [os.path.join(frame_dir, f"{ts.item()}.png") for ts in timestamps]
        
        # Verify all images exist
        missing_paths = [p for p in image_paths if not os.path.exists(p)]
        if missing_paths:
            logging.warning(f"Missing images for scene {scene_id}: {missing_paths[:3]}...")
            return None
        
        # Load images
        try:
            images = [self._load_image(p) for p in image_paths]
        except Exception as e:
            logging.warning(f"Failed to load images for scene {scene_id}: {e}")
            return None
        
        # Stack into tensor (T, H, W, 3)
        raw_images = np.stack(images, axis=0)  # (T, H, W, 3)
        raw_images = torch.from_numpy(raw_images).to(torch.uint8)
        
        return {
            "scene_id": scene_id,
            "raw_images": raw_images,  # (T, H, W, 3) uint8 in [0, 255]
            "frame_indices": valid_frame_ids.tolist(),
        }


def _pick_heldout_indices(
    valid_frame_ids: np.ndarray,
    T_in_eff: int,
    total_frames: int,
    num_heldout: int,
) -> List[int]:
    """Held-out frame index selection shared across datasets.

    Returns up to `num_heldout` frame indices that fall between consecutive
    target frames in `valid_frame_ids[T_in_eff:]`. Tries midpoints first, then
    backfills with quarter-points if there aren't enough midpoints. Skips any
    candidate that collides with `valid_frame_ids` or is out of `[0, total_frames)`.
    """
    target_ids = valid_frame_ids[T_in_eff:].tolist()
    valid_set = set(int(i) for i in valid_frame_ids.tolist())
    candidates: list = []
    for a, b in zip(target_ids[:-1], target_ids[1:]):
        if a == b:
            continue
        mid = int((a + b) // 2)
        if mid in valid_set or mid < 0 or mid >= total_frames:
            continue
        candidates.append(mid)
    if len(candidates) < num_heldout and len(target_ids) >= 2:
        for a, b in zip(target_ids[:-1], target_ids[1:]):
            if len(candidates) >= num_heldout:
                break
            for frac_num, frac_den in [(1, 4), (3, 4)]:
                q = int(a + (b - a) * frac_num / frac_den)
                if q in valid_set or q < 0 or q >= total_frames:
                    continue
                if q in candidates:
                    continue
                candidates.append(q)
                if len(candidates) >= num_heldout:
                    break
    return candidates[:num_heldout]


class RealEstate10KDatasetWithEulerRaw(Dataset):
    """
    Raw dataset for RealEstate10K that loads from pre-saved .pt files (like RealEstate10KDatasetWithDA3Euler)
    but does NOT perform geometry model inference.
    
    This dataset is designed to be used with models that have integrated
    geometry processing (GeometryModel inside the main model). It only
    loads and returns raw images, deferring all geometry computation to
    the model's forward pass for batched GPU processing.
    
    Data source: Pre-saved .pt files containing encoded image bytes.
    
    Returns:
        Dict containing:
            - scene_id: str
            - raw_images: torch.Tensor of shape (T_in + T_out, H, W, 3) in [0, 255]
            - frame_indices: List[int] of selected frame indices
    """
    
    def __init__(
        self,
        data_path: str,
        train_scenes_path: str = "",
        test_scenes_path: str = "",
        T_in: int = 3,
        T_out: int = 1,
        is_train: bool = True,
        sample_criterion: Literal["sequential", "sequential-random", "random", "equidistant"] = "sequential",
        step_size: int = 1,
        max_scenes: int = None,
        return_camera_poses: bool = False,
        # Dynamic T_in/T_out support
        T_in_range: Optional[Tuple[int, int]] = None,
        num_views: Optional[int] = None,
        dynamic_T: bool = False,
        scene_list_file: str = None,
        sample_from_end: bool = False,
        # Dynamic mask support (SpatialVID)
        dyn_masks_root: Optional[str] = None,
        uuid_group_map: Optional[dict] = None,
        # Image resize
        longest_side: Optional[int] = None,
        # Precomputed depth loading
        return_depths: bool = False,
        # Held-out frames for downstream 3DGS generalization eval. When
        # num_heldout>0, __getitem__ also returns extra GT frames at the
        # midpoints between consecutive target frames (positions T_in..T_total-1
        # in valid_frame_ids), under keys heldout_raw_images / heldout_w2c /
        # heldout_intrinsics / heldout_depths / heldout_frame_indices.
        # Held-out frames inherit return_camera_poses (always extracts poses
        # when num_heldout>0, regardless of return_camera_poses).
        num_heldout: int = 0,
    ):
        self.data_path = Path(data_path)
        self.train_scenes_path = train_scenes_path
        self.test_scenes_path = test_scenes_path

        self.T_in = T_in
        self.T_out = T_out
        self.is_train = is_train
        self.sample_criterion = sample_criterion
        self.step_size = step_size
        self.return_camera_poses = return_camera_poses
        self.sample_from_end = sample_from_end
        self.longest_side = longest_side
        self.return_depths = return_depths
        self.num_heldout = num_heldout

        # Dynamic mask support (SpatialVID)
        self.dyn_masks_root = dyn_masks_root
        self.uuid_group_map = uuid_group_map
        self.use_dyn_masks = dyn_masks_root is not None and uuid_group_map is not None

        # Dynamic T_in/T_out configuration
        # T_out is always derived as num_views - T_in
        self.dynamic_T = dynamic_T
        if dynamic_T:
            self.T_in_range = T_in_range if T_in_range is not None else (T_in, T_in)
            self.num_views = num_views if num_views is not None else (T_in + T_out)
            self.T_total_max = self.num_views
        else:
            self.T_in_range = None
            self.num_views = T_in + T_out
            self.T_total_max = T_in + T_out

        # Load scene paths
        if scene_list_file is not None:
            # Use explicit scene list file (e.g. test_scenes_100.txt)
            scene_list_path = Path(scene_list_file)
            if not scene_list_path.is_absolute():
                # Resolve against data_path first, then fall back to the CWD
                # (repo-relative lists like assets/*_test_scenes_100.txt).
                candidate = self.data_path / scene_list_file
                scene_list_path = candidate if candidate.exists() else scene_list_path
            self.scene_paths = self._read_split_txt(scene_list_path, prefix=train_scenes_path)
        elif self.is_train is None:
            self.scene_paths = (
                self._read_split_txt(self.data_path / "train_scenes.txt", prefix=train_scenes_path) +
                self._read_split_txt(self.data_path / "val_scenes.txt", prefix=train_scenes_path)
            )
        elif self.is_train:
            self.scene_paths = self._read_split_txt(self.data_path / "train_scenes.txt", prefix=train_scenes_path)
        else:
            self.scene_paths = self._read_split_txt(self.data_path / "val_scenes.txt", prefix=train_scenes_path)

        # Limit number of scenes if specified
        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]

        print(f"RealEstate10KDatasetWithEulerRaw: Found {len(self.scene_paths)} scenes (is_train={is_train}, scene_list_file={scene_list_file}, max_scenes={max_scenes})")

    def _read_split_txt(self, split_file: str, prefix: str = None) -> List[str]:
        with open(split_file, "r") as f:
            scene_ids = [
                os.path.join(prefix, line.strip()) if prefix else line.strip()
                for line in f.readlines()
            ]
        return scene_ids

    def _convert_image(self, image_bytes) -> np.ndarray:
        """Convert bytes image to numpy array of shape (H, W, 3)."""
        decoded = io.decode_image(image_bytes, mode="RGB").permute(1, 2, 0).numpy()
        return decoded  # H, W, 3

    def _extract_cameras_from_torch(
        self,
        cameras: torch.Tensor,  # (N, 18) cameras tensor from .torch file
        frame_indices: np.ndarray,  # Selected frame indices
        img_height: int,  # actual raw image height (after padding)
        img_width: int,   # actual raw image width (after padding)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract camera poses from .torch file cameras tensor.

        Args:
            cameras: (N, 18) tensor where each row contains:
                - [0:6]: intrinsics [fx, fy, cx, cy, 0, 0] (normalized)
                - [6:18]: extrinsics as 3x4 w2c matrix (12 values)
            frame_indices: Indices of selected frames
            img_height: Raw image height to denormalize intrinsics to
            img_width: Raw image width to denormalize intrinsics to

        Returns:
            w2c: (T, 4, 4) world-to-camera transformation matrices
            intrinsics_3x3: (T, 3, 3) camera intrinsics in pixel space
        """
        # Extract cameras for selected frames
        selected_cameras = cameras[frame_indices]  # (T, 18)
        T = len(frame_indices)

        # Extract and denormalize intrinsics (first 6 values)
        intrinsics_norm = selected_cameras[:, :6]  # (T, 6)

        # Denormalize to pixel space using actual raw image resolution
        fx = intrinsics_norm[:, 0] * img_width
        fy = intrinsics_norm[:, 1] * img_height
        cx = intrinsics_norm[:, 2] * img_width
        cy = intrinsics_norm[:, 3] * img_height

        # Build 3x3 intrinsics matrices
        intrinsics_3x3 = torch.zeros((T, 3, 3), dtype=torch.float32)
        intrinsics_3x3[:, 0, 0] = fx
        intrinsics_3x3[:, 1, 1] = fy
        intrinsics_3x3[:, 0, 2] = cx
        intrinsics_3x3[:, 1, 2] = cy
        intrinsics_3x3[:, 2, 2] = 1.0

        # Extract extrinsics (last 12 values) and reshape to 3x4
        extrinsics_3x4 = selected_cameras[:, 6:].reshape(T, 3, 4)  # (T, 3, 4)

        # Convert to 4x4 by adding [0, 0, 0, 1] row
        w2c = torch.zeros((T, 4, 4), dtype=torch.float32)
        w2c[:, :3, :] = extrinsics_3x4
        w2c[:, 3, 3] = 1.0

        return w2c, intrinsics_3x3

    def _load_dyn_masks(
        self,
        scene_id_file: str,
        valid_frame_ids: np.ndarray,
        H_target: int,
        W_target: int,
    ) -> Optional[torch.Tensor]:
        """
        Load per-frame dynamic masks from dyn_masks.npz for a given scene.

        Returns:
            (T, 1, H_target, W_target) float32 tensor, 1=dynamic 0=static,
            or None if masks are unavailable.
        """
        try:
            # Extract UUID from scene_id_file (e.g. "train-scenes/abc123.pt" -> "abc123")
            uuid = os.path.basename(scene_id_file).replace('.pt', '')
            if uuid not in self.uuid_group_map:
                return None

            group = self.uuid_group_map[uuid]
            npz_path = os.path.join(self.dyn_masks_root, group, uuid, "dyn_masks.npz")
            if not os.path.exists(npz_path):
                return None

            npz = np.load(npz_path)
            mask_shape = npz["shape"]  # (2,) array: [H, W]
            H_mask = int(mask_shape[0])
            W_mask = int(mask_shape[1])

            masks = []
            for frame_idx in valid_frame_ids:
                data_key = f"f_{frame_idx}_data"
                indices_key = f"f_{frame_idx}_indices"
                indptr_key = f"f_{frame_idx}_indptr"

                if data_key not in npz:
                    # Frame has no mask entry — treat as all-static
                    masks.append(np.zeros((H_mask, W_mask), dtype=np.uint8))
                    continue

                data = npz[data_key]
                indices = npz[indices_key]
                indptr = npz[indptr_key]

                # Reconstruct dense mask from CSR components (vectorized, no scipy)
                row_indices = np.repeat(np.arange(H_mask), np.diff(indptr))
                flat_indices = row_indices * W_mask + indices
                mask = np.zeros(H_mask * W_mask, dtype=np.uint8)
                mask[flat_indices] = 1
                masks.append(mask.reshape(H_mask, W_mask))

            # Stack to (T, H_mask, W_mask), convert to float tensor
            masks_np = np.stack(masks, axis=0)  # (T, H_mask, W_mask)
            masks_tensor = torch.from_numpy(masks_np).float().unsqueeze(1)  # (T, 1, H_mask, W_mask)

            # Resize to target resolution if needed
            if H_mask != H_target or W_mask != W_target:
                masks_tensor = F.interpolate(
                    masks_tensor, size=(H_target, W_target), mode="nearest"
                )

            return masks_tensor  # (T, 1, H_target, W_target)

        except Exception as e:
            logging.warning(f"Failed to load dynamic masks for {scene_id_file}: {e}")
            return None

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_id_file = self.scene_paths[idx]

        try:
            data = torch.load(_resolve_scene_path(self.data_path, scene_id_file, (".torch", ".pt")), weights_only=True)
        except Exception as e:
            logging.warning(f"Failed to load scene {scene_id_file}: {e}")
            return None

        # Initialize camera pose variables
        w2c_matrices = None
        intrinsics_matrices = None
        cameras = None

        # Extract cameras if camera poses are requested OR held-out frames needed
        if self.return_camera_poses or self.num_heldout > 0:
            if 'cameras' not in data:
                logging.warning(f"No cameras field in {scene_id_file}")
                return None
            cameras = data['cameras']  # (N, 18) tensor

        total_frames = len(data["images"])

        # Use T_total_max for frame selection in dynamic mode
        T_total = self.T_total_max if self.dynamic_T else (self.T_in + self.T_out)

        # Check if scene has enough frames
        if total_frames < T_total:
            logging.warning(
                f"Scene {scene_id_file} has {total_frames} frames, "
                f"need at least {T_total}. Skipping."
            )
            return None

        # Select frame indices based on sampling criterion (same as RealEstate10KDatasetWithDA3Euler)
        if self.sample_criterion == "sequential":
            # Use largest step_size that fits, falling back from self.step_size
            effective_step = min(self.step_size, (total_frames - 1) // (T_total - 1))
            if self.is_train:
                start_idx = np.random.randint(0, total_frames - (T_total - 1) * effective_step)
            elif self.sample_from_end:
                start_idx = total_frames - (T_total - 1) * effective_step - 1
            else:
                start_idx = 0
            valid_frame_ids = np.arange(
                start_idx,
                start_idx + T_total * effective_step,
                effective_step
            )
        elif self.sample_criterion == "sequential-random":
            # Clamp max step size to what the scene can support
            max_step = min(self.step_size, (total_frames - 1) // (T_total - 1))
            sample_step_size = np.random.randint(1, max_step + 1)
            start_idx = np.random.randint(0, total_frames - (T_total - 1) * sample_step_size)
            valid_frame_ids = np.arange(
                start_idx,
                start_idx + T_total * sample_step_size,
                sample_step_size
            )
        elif self.sample_criterion == "random":
            valid_frame_ids = np.random.choice(total_frames, T_total, replace=False)
        elif self.sample_criterion == "equidistant":
            valid_frame_ids = np.linspace(0, total_frames - 1, T_total, dtype=int)
        else:
            raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")

        # Reorder frames for interpolation: boundary frames first (inputs), middle frames second (outputs)
        # Only applies in dynamic_T mode with sequential sampling, so the model sees first+last as inputs
        if self.dynamic_T and self.sample_criterion in ["sequential", "sequential-random"]:
            T_in_max = self.T_in_range[1]
            # Pick T_in_max indices spread evenly (e.g. T_in_max=2 -> first and last)
            input_positions = np.round(np.linspace(0, T_total - 1, T_in_max)).astype(int)
            all_positions = np.arange(T_total)
            output_positions = np.array([p for p in all_positions if p not in input_positions])
            # Shuffle output positions during training to avoid positional bias
            # (without this, T_in=1 always places the farthest frame at output position 0)
            if self.is_train:
                np.random.shuffle(output_positions)
            reorder = np.concatenate([input_positions, output_positions])
            valid_frame_ids = valid_frame_ids[reorder]

        # Decode only the selected frames (avoids decoding all frames in the scene)
        selected_images = [
            self._convert_image(torch.as_tensor(data["images"][int(i)][()]))
            for i in valid_frame_ids
        ]

        # Stack into tensor (T, H, W, 3)
        raw_images = np.stack(selected_images, axis=0)  # (T, H, W, 3)
        raw_images = torch.from_numpy(raw_images).to(torch.uint8)

        # Load precomputed depths if requested
        raw_depths = None
        if self.return_depths and "depths" in data:
            raw_depths = torch.stack([data["depths"][int(i)].float() for i in valid_frame_ids])  # (T, H, W) metric depth

        # Ensure dimensions are divisible by 8 (ray encoder patch size)
        H_raw, W_raw = raw_images.shape[1], raw_images.shape[2]
        H_target = round(H_raw / 8) * 8
        W_target = round(W_raw / 8) * 8
        if H_raw != H_target or W_raw != W_target:
            raw_images = F.interpolate(
                raw_images.permute(0, 3, 1, 2).float(),
                size=(H_target, W_target),
                mode="bilinear", align_corners=False,
            ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        # Longest-side resize (preserves aspect ratio, result divisible by 8)
        if self.longest_side is not None:
            H_cur, W_cur = raw_images.shape[1], raw_images.shape[2]
            cur_longest = max(H_cur, W_cur)
            if cur_longest != self.longest_side:
                scale = self.longest_side / cur_longest
                H_new = round(H_cur * scale / 8) * 8
                W_new = round(W_cur * scale / 8) * 8
                raw_images = F.interpolate(
                    raw_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        # Resize depths to match images if needed
        if raw_depths is not None and raw_depths.shape[-2:] != (raw_images.shape[1], raw_images.shape[2]):
            raw_depths = F.interpolate(
                raw_depths.unsqueeze(1),
                size=(raw_images.shape[1], raw_images.shape[2]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        # Extract camera poses if requested
        if self.return_camera_poses:
            try:
                w2c_matrices, intrinsics_matrices = self._extract_cameras_from_torch(
                    cameras=cameras,
                    frame_indices=valid_frame_ids,
                    img_height=raw_images.shape[1],
                    img_width=raw_images.shape[2],
                )
            except Exception as e:
                logging.warning(f"Failed to extract camera poses for {scene_id_file}: {e}")
                # Continue without poses rather than failing entirely
                w2c_matrices = None
                intrinsics_matrices = None

        # Load dynamic masks (SpatialVID)
        dyn_masks = None
        if self.use_dyn_masks:
            dyn_masks = self._load_dyn_masks(
                scene_id_file, valid_frame_ids,
                H_target=raw_images.shape[1], W_target=raw_images.shape[2],
            )

        # Held-out frames at midpoints between consecutive target frames.
        # Targets occupy positions [T_in, T_total) in valid_frame_ids; midpoints
        # land between them, giving up to T_out-1 candidates. Skip any that
        # collide with valid_frame_ids or fall outside the scene range.
        heldout_raw_images = None
        heldout_w2c = None
        heldout_intrinsics = None
        heldout_depths = None
        heldout_frame_ids: list = []
        if self.num_heldout > 0 and cameras is not None:
            T_in_eff = self.T_in_range[1] if self.dynamic_T else self.T_in
            heldout_frame_ids = _pick_heldout_indices(
                valid_frame_ids=valid_frame_ids,
                T_in_eff=T_in_eff,
                total_frames=total_frames,
                num_heldout=self.num_heldout,
            )

        if heldout_frame_ids:
            heldout_arr = np.array(heldout_frame_ids, dtype=np.int64)
            held_imgs_np = np.stack(
                [self._convert_image(torch.as_tensor(data["images"][int(i)][()]))
                 for i in heldout_arr],
                axis=0,
            )  # (Nh, H_raw, W_raw, 3)
            heldout_raw_images = torch.from_numpy(held_imgs_np).to(torch.uint8)

            # Load held-out depths if main path loaded depths
            if raw_depths is not None and "depths" in data:
                heldout_depths = torch.stack(
                    [data["depths"][int(i)].float() for i in heldout_arr]
                )

            # Match the resize chain applied to the main frames so the saved
            # poses use the same resolution as the saved images.
            H_raw_h, W_raw_h = heldout_raw_images.shape[1], heldout_raw_images.shape[2]
            H_target_h = round(H_raw_h / 8) * 8
            W_target_h = round(W_raw_h / 8) * 8
            if H_raw_h != H_target_h or W_raw_h != W_target_h:
                heldout_raw_images = F.interpolate(
                    heldout_raw_images.permute(0, 3, 1, 2).float(),
                    size=(H_target_h, W_target_h),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

            target_h, target_w = raw_images.shape[1], raw_images.shape[2]
            if heldout_raw_images.shape[1] != target_h or heldout_raw_images.shape[2] != target_w:
                heldout_raw_images = F.interpolate(
                    heldout_raw_images.permute(0, 3, 1, 2).float(),
                    size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

            if heldout_depths is not None and heldout_depths.shape[-2:] != (target_h, target_w):
                heldout_depths = F.interpolate(
                    heldout_depths.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)

            try:
                heldout_w2c, heldout_intrinsics = self._extract_cameras_from_torch(
                    cameras=cameras,
                    frame_indices=heldout_arr,
                    img_height=target_h,
                    img_width=target_w,
                )
            except Exception as e:
                logging.warning(f"Failed to extract held-out poses for {scene_id_file}: {e}")
                heldout_raw_images = None
                heldout_w2c = None
                heldout_intrinsics = None
                heldout_depths = None
                heldout_frame_ids = []

        result = {
            "scene_id": scene_id_file.split('.')[0],
            "raw_images": raw_images,  # (T, H, W, 3) uint8 in [0, 255]
            "frame_indices": valid_frame_ids.tolist(),
        }

        if raw_depths is not None:
            result["depths"] = raw_depths  # (T, H, W) float32

        if dyn_masks is not None:
            result["dyn_masks"] = dyn_masks  # (T, 1, H, W) float32, 1=dynamic 0=static

        # Include T range metadata for dynamic mode
        if self.dynamic_T:
            result["T_in_range"] = self.T_in_range
            result["num_views"] = self.num_views
            result["dynamic_T"] = True

        if self.return_camera_poses:
            result["w2c"] = w2c_matrices  # (T, 4, 4)
            result["intrinsics"] = intrinsics_matrices  # (T, 3, 3)

        if heldout_raw_images is not None:
            result["heldout_raw_images"] = heldout_raw_images  # (Nh, H, W, 3) uint8
            result["heldout_w2c"] = heldout_w2c                # (Nh, 4, 4)
            result["heldout_intrinsics"] = heldout_intrinsics  # (Nh, 3, 3)
            result["heldout_frame_indices"] = heldout_frame_ids
            if heldout_depths is not None:
                result["heldout_depths"] = heldout_depths      # (Nh, H, W)

        return result


class DL3DVDataset(Dataset):
    """
    Raw dataset for DL3DV that loads from HDF5 (.h5) files.

    Mirrors the interface and return format of RealEstate10KDatasetWithEulerRaw exactly.
    The only difference is the storage backend (H5 vs .pt).

    Each H5 file contains:
        - 'cameras': (N, 18) float32 — same format as RE10K (normalized intrinsics + w2c)
        - 'images': group with keys '0', '1', ... containing JPEG-encoded bytes

    Returns:
        Dict containing:
            - scene_id: str
            - raw_images: torch.Tensor of shape (T_in + T_out, H, W, 3) in [0, 255]
            - frame_indices: List[int] of selected frame indices
            - w2c: (T, 4, 4) world-to-camera matrices (if return_camera_poses=True)
            - intrinsics: (T, 3, 3) camera intrinsics (if return_camera_poses=True)
    """

    def __init__(
        self,
        data_path: str,
        train_scenes_path: str = "",
        test_scenes_path: str = "",
        T_in: int = 3,
        T_out: int = 1,
        is_train: bool = True,
        sample_criterion: Literal["sequential", "sequential-random", "random", "equidistant"] = "sequential",
        step_size: int = 1,
        max_scenes: int = None,
        return_camera_poses: bool = False,
        return_depths: bool = False,
        # Dynamic T_in/T_out support
        T_in_range: Optional[Tuple[int, int]] = None,
        num_views: Optional[int] = None,
        dynamic_T: bool = False,
        scene_list_file: str = None,
        sample_from_end: bool = False,
        name: str = "dl3dv",
        # Image resize
        longest_side: Optional[int] = None,
        # Held-out frames for downstream 3DGS generalization eval (see
        # RealEstate10KDatasetWithEulerRaw for field semantics).
        num_heldout: int = 0,
    ):
        self.data_path = Path(data_path)
        self.name = name
        self.train_scenes_path = train_scenes_path
        self.test_scenes_path = test_scenes_path

        self.T_in = T_in
        self.T_out = T_out
        self.is_train = is_train
        self.sample_criterion = sample_criterion
        self.step_size = step_size
        self.return_camera_poses = return_camera_poses
        self.return_depths = return_depths
        self.sample_from_end = sample_from_end
        self.longest_side = longest_side
        self.num_heldout = num_heldout

        # Dynamic T_in/T_out configuration
        self.dynamic_T = dynamic_T
        if dynamic_T:
            self.T_in_range = T_in_range if T_in_range is not None else (T_in, T_in)
            self.num_views = num_views if num_views is not None else (T_in + T_out)
            self.T_total_max = self.num_views
        else:
            self.T_in_range = None
            self.num_views = T_in + T_out
            self.T_total_max = T_in + T_out

        # Load scene paths
        if scene_list_file is not None:
            scene_list_path = Path(scene_list_file)
            if not scene_list_path.is_absolute():
                # Resolve against data_path first, then fall back to the CWD
                # (repo-relative lists like assets/*_test_scenes_100.txt).
                candidate = self.data_path / scene_list_file
                scene_list_path = candidate if candidate.exists() else scene_list_path
            self.scene_paths = self._read_scene_list(scene_list_path)
        else:
            scenes_dir = self.data_path / (train_scenes_path if is_train else test_scenes_path)
            self.scene_paths = sorted([
                f.name for f in scenes_dir.iterdir() if f.suffix == '.h5'
            ])

        # Limit number of scenes if specified
        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]

        # Determine which directory scenes live in
        self._scenes_dir = train_scenes_path if is_train else test_scenes_path

        print(f"{self.name}Dataset: Found {len(self.scene_paths)} scenes (is_train={is_train}, max_scenes={max_scenes})")

    def _read_scene_list(self, scene_list_path: Path) -> List[str]:
        with open(scene_list_path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _convert_image(self, image_bytes) -> np.ndarray:
        """Convert JPEG bytes to numpy array of shape (H, W, 3)."""
        decoded = io.decode_image(torch.as_tensor(image_bytes), mode="RGB").permute(1, 2, 0).numpy()
        return decoded  # H, W, 3

    def _extract_cameras(
        self,
        cameras: np.ndarray,  # (N, 18)
        frame_indices: np.ndarray,
        img_height: int,
        img_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract camera poses from (N, 18) cameras array.
        Same format as RE10K: [fx, fy, cx, cy, 0, 0, <3x4 w2c>].
        """
        selected = torch.from_numpy(cameras[frame_indices].astype(np.float32))
        T = len(frame_indices)

        # Denormalize intrinsics
        fx = selected[:, 0] * img_width
        fy = selected[:, 1] * img_height
        cx = selected[:, 2] * img_width
        cy = selected[:, 3] * img_height

        intrinsics_3x3 = torch.zeros((T, 3, 3), dtype=torch.float32)
        intrinsics_3x3[:, 0, 0] = fx
        intrinsics_3x3[:, 1, 1] = fy
        intrinsics_3x3[:, 0, 2] = cx
        intrinsics_3x3[:, 1, 2] = cy
        intrinsics_3x3[:, 2, 2] = 1.0

        extrinsics_3x4 = selected[:, 6:].reshape(T, 3, 4)
        w2c = torch.zeros((T, 4, 4), dtype=torch.float32)
        w2c[:, :3, :] = extrinsics_3x4
        w2c[:, 3, 3] = 1.0

        return w2c, intrinsics_3x3

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_file = self.scene_paths[idx]
        h5_path = _resolve_scene_path(self.data_path / self._scenes_dir, scene_file, (".h5",))

        try:
            with h5py.File(str(h5_path), 'r') as f:
                cameras = f['cameras'][:]  # (N, 18) float32
                total_frames = len(f['images'])

                # Use T_total_max for frame selection in dynamic mode
                T_total = self.T_total_max if self.dynamic_T else (self.T_in + self.T_out)

                # Check if scene has enough frames
                if total_frames < T_total:
                    logging.warning(
                        f"DL3DV scene {scene_file} has {total_frames} frames, "
                        f"need at least {T_total}. Skipping."
                    )
                    return None

                # Select frame indices
                if self.sample_criterion == "sequential":
                    # Use largest step_size that fits, falling back from self.step_size
                    effective_step = min(self.step_size, (total_frames - 1) // (T_total - 1))
                    if self.is_train:
                        start_idx = np.random.randint(0, total_frames - (T_total - 1) * effective_step)
                    elif self.sample_from_end:
                        start_idx = total_frames - (T_total - 1) * effective_step - 1
                    else:
                        start_idx = 0
                    valid_frame_ids = np.arange(
                        start_idx,
                        start_idx + T_total * effective_step,
                        effective_step
                    )
                elif self.sample_criterion == "sequential-random":
                    max_step = min(self.step_size, (total_frames - 1) // (T_total - 1))
                    sample_step_size = np.random.randint(1, max_step + 1)
                    start_idx = np.random.randint(0, total_frames - (T_total - 1) * sample_step_size)
                    valid_frame_ids = np.arange(
                        start_idx,
                        start_idx + T_total * sample_step_size,
                        sample_step_size
                    )
                elif self.sample_criterion == "random":
                    valid_frame_ids = np.random.choice(total_frames, T_total, replace=False)
                elif self.sample_criterion == "equidistant":
                    valid_frame_ids = np.linspace(0, total_frames - 1, T_total, dtype=int)
                else:
                    raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")

                # Reorder frames for dynamic T interpolation
                if self.dynamic_T and self.sample_criterion in ["sequential", "sequential-random"]:
                    T_in_max = self.T_in_range[1]
                    input_positions = np.round(np.linspace(0, T_total - 1, T_in_max)).astype(int)
                    all_positions = np.arange(T_total)
                    output_positions = np.array([p for p in all_positions if p not in input_positions])
                    # Shuffle output positions during training to avoid positional bias
                    # (without this, T_in=1 always places the farthest frame at output position 0)
                    if self.is_train:
                        np.random.shuffle(output_positions)
                    reorder = np.concatenate([input_positions, output_positions])
                    valid_frame_ids = valid_frame_ids[reorder]

                # Decode selected images from H5
                selected_images = [
                    self._convert_image(f['images'][str(int(i))][:])
                    for i in valid_frame_ids
                ]

                # Load precomputed depth maps if requested and available
                raw_depths = None
                if self.return_depths and 'depth' in f:
                    selected_depths = [
                        torch.from_numpy(f['depth'][str(int(i))][()].astype(np.float32))
                        for i in valid_frame_ids
                    ]
                    raw_depths = torch.stack(selected_depths, dim=0)  # (T, H, W) float32

                # Held-out frames at midpoints between consecutive targets.
                # Loaded inside the H5 `with` block since the file closes below.
                heldout_frame_ids: List[int] = []
                heldout_selected_images = None
                heldout_selected_depths = None
                if self.num_heldout > 0:
                    T_in_eff = self.T_in_range[1] if self.dynamic_T else self.T_in
                    heldout_frame_ids = _pick_heldout_indices(
                        valid_frame_ids=valid_frame_ids,
                        T_in_eff=T_in_eff,
                        total_frames=total_frames,
                        num_heldout=self.num_heldout,
                    )
                    if heldout_frame_ids:
                        heldout_selected_images = [
                            self._convert_image(f['images'][str(int(i))][:])
                            for i in heldout_frame_ids
                        ]
                        if self.return_depths and 'depth' in f:
                            heldout_selected_depths = torch.stack([
                                torch.from_numpy(f['depth'][str(int(i))][()].astype(np.float32))
                                for i in heldout_frame_ids
                            ], dim=0)

        except Exception as e:
            logging.warning(f"Failed to load DL3DV scene {scene_file}: {e}")
            return None

        # Stack into tensor (T, H, W, 3)
        raw_images = np.stack(selected_images, axis=0)
        raw_images = torch.from_numpy(raw_images).to(torch.uint8)

        # Ensure dimensions are divisible by 8
        H_raw, W_raw = raw_images.shape[1], raw_images.shape[2]
        H_target = round(H_raw / 8) * 8
        W_target = round(W_raw / 8) * 8
        if H_raw != H_target or W_raw != W_target:
            raw_images = F.interpolate(
                raw_images.permute(0, 3, 1, 2).float(),
                size=(H_target, W_target),
                mode="bilinear", align_corners=False,
            ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        # Longest-side resize (preserves aspect ratio, result divisible by 8)
        if self.longest_side is not None:
            H_cur, W_cur = raw_images.shape[1], raw_images.shape[2]
            cur_longest = max(H_cur, W_cur)
            if cur_longest != self.longest_side:
                scale = self.longest_side / cur_longest
                H_new = round(H_cur * scale / 8) * 8
                W_new = round(W_cur * scale / 8) * 8
                raw_images = F.interpolate(
                    raw_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        # Extract camera poses if requested
        w2c_matrices = None
        intrinsics_matrices = None
        if self.return_camera_poses:
            try:
                w2c_matrices, intrinsics_matrices = self._extract_cameras(
                    cameras=cameras,
                    frame_indices=valid_frame_ids,
                    img_height=raw_images.shape[1],
                    img_width=raw_images.shape[2],
                )
            except Exception as e:
                logging.warning(f"Failed to extract camera poses for DL3DV {scene_file}: {e}")
                w2c_matrices = None
                intrinsics_matrices = None

        result = {
            "scene_id": f"{self.name}/{scene_file.replace('.h5', '')}",
            "raw_images": raw_images,
            "frame_indices": valid_frame_ids.tolist(),
        }

        if self.dynamic_T:
            result["T_in_range"] = self.T_in_range
            result["num_views"] = self.num_views
            result["dynamic_T"] = True

        if self.return_camera_poses:
            result["w2c"] = w2c_matrices
            result["intrinsics"] = intrinsics_matrices

        if self.return_depths and raw_depths is not None:
            # Resize depths to match image resolution if needed
            if raw_depths.shape[1] != raw_images.shape[1] or raw_depths.shape[2] != raw_images.shape[2]:
                raw_depths = F.interpolate(
                    raw_depths.unsqueeze(1),  # (T, 1, H_depth, W_depth)
                    size=(raw_images.shape[1], raw_images.shape[2]),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)  # (T, H, W)
            result["depths"] = raw_depths  # (T, H, W) float32

        # Held-out frames: apply the same resize chain as the main frames so
        # their poses share resolution, then extract poses.
        if heldout_selected_images:
            heldout_raw_images = torch.from_numpy(
                np.stack(heldout_selected_images, axis=0)
            ).to(torch.uint8)
            target_h, target_w = raw_images.shape[1], raw_images.shape[2]
            if heldout_raw_images.shape[1] != target_h or heldout_raw_images.shape[2] != target_w:
                heldout_raw_images = F.interpolate(
                    heldout_raw_images.permute(0, 3, 1, 2).float(),
                    size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

            heldout_depths = None
            if heldout_selected_depths is not None:
                heldout_depths = heldout_selected_depths
                if heldout_depths.shape[-2:] != (target_h, target_w):
                    heldout_depths = F.interpolate(
                        heldout_depths.unsqueeze(1),
                        size=(target_h, target_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)

            try:
                heldout_w2c, heldout_intrinsics = self._extract_cameras(
                    cameras=cameras,
                    frame_indices=np.array(heldout_frame_ids, dtype=np.int64),
                    img_height=target_h,
                    img_width=target_w,
                )
                result["heldout_raw_images"] = heldout_raw_images
                result["heldout_w2c"] = heldout_w2c
                result["heldout_intrinsics"] = heldout_intrinsics
                result["heldout_frame_indices"] = heldout_frame_ids
                if heldout_depths is not None:
                    result["heldout_depths"] = heldout_depths
            except Exception as e:
                logging.warning(f"Failed to extract held-out poses for DL3DV {scene_file}: {e}")

        return result


class MipNeRF360Dataset(Dataset):
    """
    Mip-NeRF 360 dataset loader with COLMAP ground-truth poses.

    Layout expected under `data_path`:
        <scene>/sparse/0/{cameras,images,points3D}.bin
        <scene>/images/      (or images_2/, images_4/, images_8/)

    Return shape matches DL3DVDataset so it plugs into collate_fn_raw unchanged.
    No depth source is provided; `return_depths` is accepted but ignored.
    """

    def __init__(
        self,
        data_path: str,
        train_scenes_path: str = "train-scenes",   # unused (compat with build_dataloader)
        test_scenes_path: str = "test-scenes",     # unused (compat with build_dataloader)
        T_in: int = 3,
        T_out: int = 1,
        is_train: bool = False,
        sample_criterion: Literal["sequential", "sequential-random", "random", "equidistant"] = "sequential",
        step_size: int = 1,
        max_scenes: int = None,
        return_camera_poses: bool = True,
        return_depths: bool = False,
        # Dynamic T_in/T_out support
        T_in_range: Optional[Tuple[int, int]] = None,
        num_views: Optional[int] = None,
        dynamic_T: bool = False,
        scene_list_file: str = None,
        sample_from_end: bool = False,
        name: str = "mipnerf360",
        longest_side: Optional[int] = None,
        factor: int = 4,
        # Held-out frames for downstream 3DGS generalization eval. Mip-NeRF 360
        # has no depth source, so heldout_depths is never emitted.
        num_heldout: int = 0,
    ):
        self.data_path = Path(data_path)
        self.name = name
        self.T_in = T_in
        self.T_out = T_out
        self.is_train = is_train
        self.sample_criterion = sample_criterion
        self.step_size = step_size
        self.return_camera_poses = return_camera_poses
        self.return_depths = return_depths  # accepted, no-op
        self.sample_from_end = sample_from_end
        self.longest_side = longest_side
        self.factor = int(factor)
        self.num_heldout = num_heldout

        self.dynamic_T = dynamic_T
        if dynamic_T:
            self.T_in_range = T_in_range if T_in_range is not None else (T_in, T_in)
            self.num_views = num_views if num_views is not None else (T_in + T_out)
            self.T_total_max = self.num_views
        else:
            self.T_in_range = None
            self.num_views = T_in + T_out
            self.T_total_max = T_in + T_out

        # Scene enumeration: explicit list file, or auto-discover scenes that have COLMAP reconstructions.
        if scene_list_file is not None:
            scene_list_path = Path(scene_list_file)
            if not scene_list_path.is_absolute():
                # Resolve against data_path first, then fall back to the CWD
                # (repo-relative lists like assets/*_test_scenes_100.txt).
                candidate = self.data_path / scene_list_file
                scene_list_path = candidate if candidate.exists() else scene_list_path
            with open(scene_list_path, "r") as f:
                self.scene_paths = [line.strip() for line in f.readlines() if line.strip()]
        else:
            self.scene_paths = sorted([
                p.name for p in self.data_path.iterdir()
                if p.is_dir() and (p / "sparse" / "0" / "cameras.bin").exists()
            ])

        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]

        print(f"{self.name}Dataset: Found {len(self.scene_paths)} scenes (factor={self.factor})")

    @staticmethod
    def _colmap_params(cam) -> Tuple[float, float, float, float, int, int]:
        """Extract (fx, fy, cx, cy, W, H) from a COLMAP Camera namedtuple.

        Distortion coefficients are ignored (Mip-NeRF 360 DSLR images at factor>=4
        have negligible residual distortion at the pinhole approximation).
        """
        params = cam.params
        model = cam.model
        if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx, fy, cx, cy = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                       "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            f = float(params[0])
            fx = fy = f
            cx, cy = float(params[1]), float(params[2])
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {model}")
        return fx, fy, cx, cy, int(cam.width), int(cam.height)

    def _build_cameras_array(self, cameras_dict, sorted_imgs) -> np.ndarray:
        """Build an (N, 18) cameras array matching the RE10K/DL3DV encoding:
        [fx_norm, fy_norm, cx_norm, cy_norm, 0, 0, <w2c 3x4 flat>].
        Intrinsics are normalized by the COLMAP reference resolution (full-res).
        """
        from depth_anything_3.utils.read_write_model import qvec2rotmat

        N = len(sorted_imgs)
        arr = np.zeros((N, 18), dtype=np.float32)
        for i, im in enumerate(sorted_imgs):
            cam = cameras_dict[im.camera_id]
            fx, fy, cx, cy, W, H = self._colmap_params(cam)
            arr[i, 0] = fx / W
            arr[i, 1] = fy / H
            arr[i, 2] = cx / W
            arr[i, 3] = cy / H
            # arr[i, 4:6] = 0

            R = qvec2rotmat(np.asarray(im.qvec, dtype=np.float64)).astype(np.float32)  # (3, 3)
            t = np.asarray(im.tvec, dtype=np.float32).reshape(3, 1)
            w2c_3x4 = np.concatenate([R, t], axis=1)  # (3, 4) world-to-camera, OpenCV
            arr[i, 6:] = w2c_3x4.flatten()
        return arr

    def _extract_cameras(
        self,
        cameras: np.ndarray,  # (N, 18) normalized
        frame_indices: np.ndarray,
        img_height: int,
        img_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same denormalization as DL3DVDataset._extract_cameras."""
        selected = torch.from_numpy(cameras[frame_indices].astype(np.float32))
        T = len(frame_indices)

        fx = selected[:, 0] * img_width
        fy = selected[:, 1] * img_height
        cx = selected[:, 2] * img_width
        cy = selected[:, 3] * img_height

        intrinsics_3x3 = torch.zeros((T, 3, 3), dtype=torch.float32)
        intrinsics_3x3[:, 0, 0] = fx
        intrinsics_3x3[:, 1, 1] = fy
        intrinsics_3x3[:, 0, 2] = cx
        intrinsics_3x3[:, 1, 2] = cy
        intrinsics_3x3[:, 2, 2] = 1.0

        extrinsics_3x4 = selected[:, 6:].reshape(T, 3, 4)
        w2c = torch.zeros((T, 4, 4), dtype=torch.float32)
        w2c[:, :3, :] = extrinsics_3x4
        w2c[:, 3, 3] = 1.0
        return w2c, intrinsics_3x3

    def _images_dir_name(self) -> str:
        return "images" if self.factor == 1 else f"images_{self.factor}"

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        from depth_anything_3.utils.read_write_model import (
            read_cameras_binary, read_images_binary,
        )

        scene_name = self.scene_paths[idx]
        scene_dir = self.data_path / scene_name
        sparse_dir = scene_dir / "sparse" / "0"
        img_dir = scene_dir / self._images_dir_name()

        try:
            cameras_dict = read_cameras_binary(str(sparse_dir / "cameras.bin"))
            images_dict = read_images_binary(str(sparse_dir / "images.bin"))

            # Stable ordering by image filename (_DSC####.JPG → numeric suffix order).
            sorted_imgs = sorted(images_dict.values(), key=lambda im: im.name)
            total_frames = len(sorted_imgs)

            T_total = self.T_total_max if self.dynamic_T else (self.T_in + self.T_out)
            if total_frames < T_total:
                logging.warning(
                    f"MipNeRF360 scene {scene_name} has {total_frames} frames, "
                    f"need at least {T_total}. Skipping."
                )
                return None

            # Frame selection — mirrors DL3DVDataset.__getitem__.
            if self.sample_criterion == "sequential":
                effective_step = min(self.step_size, (total_frames - 1) // (T_total - 1))
                effective_step = max(effective_step, 1)
                if self.is_train:
                    start_idx = np.random.randint(0, total_frames - (T_total - 1) * effective_step)
                elif self.sample_from_end:
                    start_idx = total_frames - (T_total - 1) * effective_step - 1
                else:
                    start_idx = 0
                valid_frame_ids = np.arange(
                    start_idx,
                    start_idx + T_total * effective_step,
                    effective_step,
                )
            elif self.sample_criterion == "sequential-random":
                max_step = min(self.step_size, (total_frames - 1) // (T_total - 1))
                max_step = max(max_step, 1)
                sample_step_size = np.random.randint(1, max_step + 1)
                start_idx = np.random.randint(0, total_frames - (T_total - 1) * sample_step_size)
                valid_frame_ids = np.arange(
                    start_idx,
                    start_idx + T_total * sample_step_size,
                    sample_step_size,
                )
            elif self.sample_criterion == "random":
                valid_frame_ids = np.random.choice(total_frames, T_total, replace=False)
            elif self.sample_criterion == "equidistant":
                valid_frame_ids = np.linspace(0, total_frames - 1, T_total, dtype=int)
            else:
                raise ValueError(f"Unknown sample_criterion: {self.sample_criterion}")

            if self.dynamic_T and self.sample_criterion in ["sequential", "sequential-random"]:
                T_in_max = self.T_in_range[1]
                input_positions = np.round(np.linspace(0, T_total - 1, T_in_max)).astype(int)
                all_positions = np.arange(T_total)
                output_positions = np.array([p for p in all_positions if p not in input_positions])
                if self.is_train:
                    np.random.shuffle(output_positions)
                reorder = np.concatenate([input_positions, output_positions])
                valid_frame_ids = valid_frame_ids[reorder]

            selected = [sorted_imgs[i] for i in valid_frame_ids]

            # Load JPEGs from images_<factor>/. We slice the first 3 channels
            # because some Mip-NeRF 360 scenes contain 4-channel captures.
            loaded = []
            for im in selected:
                img_path = img_dir / im.name
                arr = io.read_image(str(img_path), mode=io.ImageReadMode.RGB).permute(1, 2, 0).numpy()
                loaded.append(arr)

            # Held-out frames at midpoints between consecutive targets.
            heldout_frame_ids: List[int] = []
            heldout_loaded = None
            if self.num_heldout > 0:
                T_in_eff = self.T_in_range[1] if self.dynamic_T else self.T_in
                heldout_frame_ids = _pick_heldout_indices(
                    valid_frame_ids=valid_frame_ids,
                    T_in_eff=T_in_eff,
                    total_frames=total_frames,
                    num_heldout=self.num_heldout,
                )
                if heldout_frame_ids:
                    heldout_loaded = []
                    for i in heldout_frame_ids:
                        im = sorted_imgs[i]
                        img_path = img_dir / im.name
                        arr = io.read_image(str(img_path), mode=io.ImageReadMode.RGB).permute(1, 2, 0).numpy()
                        heldout_loaded.append(arr)

        except Exception as e:
            logging.warning(f"Failed to load MipNeRF360 scene {scene_name}: {e}")
            return None

        # Stack into tensor (T, H, W, 3)
        raw_images = np.stack(loaded, axis=0)
        raw_images = torch.from_numpy(raw_images).to(torch.uint8)

        # Ensure dimensions are divisible by 8
        H_raw, W_raw = raw_images.shape[1], raw_images.shape[2]
        H_target = round(H_raw / 8) * 8
        W_target = round(W_raw / 8) * 8
        if H_raw != H_target or W_raw != W_target:
            raw_images = F.interpolate(
                raw_images.permute(0, 3, 1, 2).float(),
                size=(H_target, W_target),
                mode="bilinear", align_corners=False,
            ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        # Longest-side resize (aspect-ratio preserving, divisible by 8)
        if self.longest_side is not None:
            H_cur, W_cur = raw_images.shape[1], raw_images.shape[2]
            cur_longest = max(H_cur, W_cur)
            if cur_longest != self.longest_side:
                scale = self.longest_side / cur_longest
                H_new = round(H_cur * scale / 8) * 8
                W_new = round(W_cur * scale / 8) * 8
                raw_images = F.interpolate(
                    raw_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        w2c_matrices = None
        intrinsics_matrices = None
        cameras_arr = None
        if self.return_camera_poses or self.num_heldout > 0:
            try:
                cameras_arr = self._build_cameras_array(cameras_dict, sorted_imgs)
                if self.return_camera_poses:
                    w2c_matrices, intrinsics_matrices = self._extract_cameras(
                        cameras=cameras_arr,
                        frame_indices=valid_frame_ids,
                        img_height=raw_images.shape[1],
                        img_width=raw_images.shape[2],
                    )
            except Exception as e:
                logging.warning(f"Failed to extract camera poses for MipNeRF360 {scene_name}: {e}")
                w2c_matrices = None
                intrinsics_matrices = None
                cameras_arr = None

        result = {
            "scene_id": f"{self.name}/{scene_name}",
            "raw_images": raw_images,
            "frame_indices": valid_frame_ids.tolist(),
        }

        if self.dynamic_T:
            result["T_in_range"] = self.T_in_range
            result["num_views"] = self.num_views
            result["dynamic_T"] = True

        if self.return_camera_poses:
            result["w2c"] = w2c_matrices
            result["intrinsics"] = intrinsics_matrices

        # Held-out frames: apply the same resize chain as the main frames.
        if heldout_loaded and cameras_arr is not None:
            heldout_raw_images = torch.from_numpy(
                np.stack(heldout_loaded, axis=0)
            ).to(torch.uint8)
            target_h, target_w = raw_images.shape[1], raw_images.shape[2]
            if heldout_raw_images.shape[1] != target_h or heldout_raw_images.shape[2] != target_w:
                heldout_raw_images = F.interpolate(
                    heldout_raw_images.permute(0, 3, 1, 2).float(),
                    size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            try:
                heldout_w2c, heldout_intrinsics = self._extract_cameras(
                    cameras=cameras_arr,
                    frame_indices=np.array(heldout_frame_ids, dtype=np.int64),
                    img_height=target_h,
                    img_width=target_w,
                )
                result["heldout_raw_images"] = heldout_raw_images
                result["heldout_w2c"] = heldout_w2c
                result["heldout_intrinsics"] = heldout_intrinsics
                result["heldout_frame_indices"] = heldout_frame_ids
            except Exception as e:
                logging.warning(f"Failed to extract held-out poses for MipNeRF360 {scene_name}: {e}")

        return result


class RealEstate10KDatasetWithEulerRawTrajectory(Dataset):
    """
    Dataset for RealEstate10K that returns complete scene trajectories for GEN3C evaluation.

    This dataset is designed for use with GEN3C which requires:
    1. Input frames for 3D cache creation (T_in frames)
    2. Target frames for evaluation (T_out frames)
    3. Full trajectory poses for rendering warped views along the camera path

    Data source: Pre-saved .pt files containing encoded image bytes and camera poses.

    Returns:
        Dict containing:
            - scene_id: str
            - input_images: (T_in, H, W, 3) uint8 [0, 255] for cache creation
            - input_w2c: (T_in, 4, 4) world-to-camera matrices
            - input_intrinsics: (T_in, 3, 3) pixel-space intrinsics
            - input_frame_indices: List[int]
            - target_images: (T_out, H, W, 3) uint8 [0, 255] for evaluation
            - target_w2c: (T_out, 4, 4) world-to-camera matrices
            - target_intrinsics: (T_out, 3, 3) pixel-space intrinsics
            - target_frame_indices: List[int]
            - trajectory_w2c: (N, 4, 4) all scene poses
            - trajectory_intrinsics: (N, 3, 3) all intrinsics
            - total_frames: int
    """

    def __init__(
        self,
        data_path: str,
        train_scenes_path: str = "train-scenes",
        test_scenes_path: str = "test-scenes",
        T_in: int = 2,
        T_out: int = 1,
        is_train: bool = True,
        step_size: int = 10,
        max_scenes: int = None,
        scene_list_file: str = None,
        interpolation: bool = False,
    ):
        self.data_path = Path(data_path)
        self.train_scenes_path = train_scenes_path
        self.test_scenes_path = test_scenes_path

        self.T_in = T_in
        self.T_out = T_out
        self.is_train = is_train
        self.step_size = step_size
        self.interpolation = interpolation

        # Load scene paths
        if scene_list_file is not None:
            # Use explicit scene list file (e.g. test_scenes_100.txt)
            scene_list_path = Path(scene_list_file)
            if not scene_list_path.is_absolute():
                # Resolve against data_path first, then fall back to the CWD
                # (repo-relative lists like assets/*_test_scenes_100.txt).
                candidate = self.data_path / scene_list_file
                scene_list_path = candidate if candidate.exists() else scene_list_path
            self.scene_paths = self._read_split_txt(scene_list_path, prefix=train_scenes_path)
        elif self.is_train is None:
            self.scene_paths = (
                self._read_split_txt(self.data_path / "train_scenes.txt", prefix=train_scenes_path) +
                self._read_split_txt(self.data_path / "val_scenes.txt", prefix=train_scenes_path)
            )
        elif self.is_train:
            self.scene_paths = self._read_split_txt(self.data_path / "train_scenes.txt", prefix=train_scenes_path)
        else:
            self.scene_paths = self._read_split_txt(self.data_path / "val_scenes.txt", prefix=train_scenes_path)

        # Limit number of scenes if specified
        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]

        print(f"RealEstate10KDatasetWithEulerRawTrajectory: Found {len(self.scene_paths)} scenes (is_train={is_train}, scene_list_file={scene_list_file}, max_scenes={max_scenes})")

    def _read_split_txt(self, split_file: str, prefix: str = None) -> List[str]:
        with open(split_file, "r") as f:
            scene_ids = [
                os.path.join(prefix, line.strip()) if prefix else line.strip()
                for line in f.readlines()
            ]
        return scene_ids

    def _convert_image(self, image_bytes) -> np.ndarray:
        """Convert bytes image to numpy array of shape (H, W, 3)."""
        decoded = io.decode_image(image_bytes, mode="RGB").permute(1, 2, 0).numpy()
        return decoded  # H, W, 3

    def _extract_all_cameras(
        self,
        cameras: torch.Tensor,  # (N, 18) cameras tensor from .pt file
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract all camera poses from .pt file cameras tensor.

        Args:
            cameras: (N, 18) tensor where each row contains:
                - [0:6]: intrinsics [fx, fy, cx, cy, 0, 0] (normalized)
                - [6:18]: extrinsics as 3x4 w2c matrix (12 values)

        Returns:
            w2c: (N, 4, 4) world-to-camera transformation matrices
            intrinsics_3x3: (N, 3, 3) camera intrinsics in pixel space
        """
        N = cameras.shape[0]

        # Extract normalized [0, 1] intrinsics (first 6 values)
        intrinsics_norm = cameras[:, :6]  # (N, 6)

        # Build 3x3 normalized intrinsics matrices (resolution-independent)
        intrinsics_3x3 = torch.zeros((N, 3, 3), dtype=torch.float32)
        intrinsics_3x3[:, 0, 0] = intrinsics_norm[:, 0]  # fx_norm
        intrinsics_3x3[:, 1, 1] = intrinsics_norm[:, 1]  # fy_norm
        intrinsics_3x3[:, 0, 2] = intrinsics_norm[:, 2]  # cx_norm
        intrinsics_3x3[:, 1, 2] = intrinsics_norm[:, 3]  # cy_norm
        intrinsics_3x3[:, 2, 2] = 1.0

        # Extract extrinsics (last 12 values) and reshape to 3x4
        extrinsics_3x4 = cameras[:, 6:].reshape(N, 3, 4)  # (N, 3, 4)

        # Convert to 4x4 by adding [0, 0, 0, 1] row
        w2c = torch.zeros((N, 4, 4), dtype=torch.float32)
        w2c[:, :3, :] = extrinsics_3x4
        w2c[:, 3, 3] = 1.0

        return w2c, intrinsics_3x3

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_id_file = self.scene_paths[idx]

        try:
            data = torch.load(_resolve_scene_path(self.data_path, scene_id_file, (".torch", ".pt")), weights_only=True)
        except Exception as e:
            logging.warning(f"Failed to load scene {scene_id_file}: {e}")
            return None

        # Check for cameras
        if 'cameras' not in data:
            logging.warning(f"No cameras field in {scene_id_file}")
            return None

        cameras = data['cameras']  # (N, 18) tensor
        total_frames = len(data['images'])

        # Required frames: T_in + T_out frames with step_size spacing
        required_frames = (self.T_in + self.T_out - 1) * self.step_size + 1

        if total_frames < required_frames:
            logging.warning(
                f"Scene {scene_id_file} has {total_frames} frames, "
                f"need {required_frames} for T_in={self.T_in}, T_out={self.T_out}, step_size={self.step_size}. Skipping."
            )
            return None

        # Compute frame indices
        if self.interpolation:
            # Interpolation: inputs at endpoints, targets in between
            total_views = self.T_in + self.T_out
            all_indices = [i * self.step_size for i in range(total_views)]
            input_frame_indices = [all_indices[0], all_indices[-1]]
            target_frame_indices = all_indices[1:-1]
        else:
            # Extrapolation: sequential layout
            # Input frames: [0, step_size, 2*step_size, ..., (T_in-1)*step_size]
            input_frame_indices = [i * self.step_size for i in range(self.T_in)]
            # Target frames: [T_in*step_size, (T_in+1)*step_size, ..., (T_in+T_out-1)*step_size]
            target_frame_indices = [(self.T_in + i) * self.step_size for i in range(self.T_out)]

        # Verify indices are within bounds
        max_idx = max(max(input_frame_indices), max(target_frame_indices))
        if max_idx >= total_frames:
            logging.warning(
                f"Scene {scene_id_file}: max index {max_idx} >= total_frames {total_frames}. Skipping."
            )
            return None

        # Convert all images
        try:
            all_images = [
                self._convert_image(torch.as_tensor(d[()]))
                for d in data["images"]
            ]
        except Exception as e:
            logging.warning(f"Failed to convert images for scene {scene_id_file}: {e}")
            return None

        # Extract input and target images
        input_images = np.stack([all_images[i] for i in input_frame_indices], axis=0)  # (T_in, H, W, 3)
        target_images = np.stack([all_images[i] for i in target_frame_indices], axis=0)  # (T_out, H, W, 3)

        input_images = torch.from_numpy(input_images).to(torch.uint8)
        target_images = torch.from_numpy(target_images).to(torch.uint8)

        # Extract all camera poses for trajectory
        try:
            trajectory_w2c, trajectory_intrinsics = self._extract_all_cameras(cameras)
        except Exception as e:
            logging.warning(f"Failed to extract camera poses for {scene_id_file}: {e}")
            return None

        # Extract input and target poses
        input_w2c = trajectory_w2c[input_frame_indices]  # (T_in, 4, 4)
        input_intrinsics = trajectory_intrinsics[input_frame_indices]  # (T_in, 3, 3)
        target_w2c = trajectory_w2c[target_frame_indices]  # (T_out, 4, 4)
        target_intrinsics = trajectory_intrinsics[target_frame_indices]  # (T_out, 3, 3)

        return {
            "scene_id": scene_id_file.split('.')[0],

            # Input frames for cache creation
            "input_images": input_images,  # (T_in, H, W, 3) uint8 [0, 255]
            "input_w2c": input_w2c,  # (T_in, 4, 4)
            "input_intrinsics": input_intrinsics,  # (T_in, 3, 3)
            "input_frame_indices": input_frame_indices,

            # Target frames for evaluation
            "target_images": target_images,  # (T_out, H, W, 3) uint8 [0, 255]
            "target_w2c": target_w2c,  # (T_out, 4, 4)
            "target_intrinsics": target_intrinsics,  # (T_out, 3, 3)
            "target_frame_indices": target_frame_indices,

            # Full trajectory for warped view rendering
            "trajectory_w2c": trajectory_w2c,  # (N, 4, 4)
            "trajectory_intrinsics": trajectory_intrinsics,  # (N, 3, 3)
            "total_frames": total_frames,
        }


def collate_fn_trajectory(batch):
    """
    Collate function for RealEstate10KDatasetWithEulerRawTrajectory.
    Filters out None samples, stacks fixed-size tensors, keeps variable-length as lists.
    """
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Stack fixed-size tensors
    result = {
        "scene_ids": [item["scene_id"] for item in batch],
        "input_images": torch.stack([item["input_images"] for item in batch]),  # (B, T_in, H, W, 3)
        "input_w2c": torch.stack([item["input_w2c"] for item in batch]),  # (B, T_in, 4, 4)
        "input_intrinsics": torch.stack([item["input_intrinsics"] for item in batch]),  # (B, T_in, 3, 3)
        "target_images": torch.stack([item["target_images"] for item in batch]),  # (B, T_out, H, W, 3)
        "target_w2c": torch.stack([item["target_w2c"] for item in batch]),  # (B, T_out, 4, 4)
        "target_intrinsics": torch.stack([item["target_intrinsics"] for item in batch]),  # (B, T_out, 3, 3)
        "input_frame_indices": [item["input_frame_indices"] for item in batch],
        "target_frame_indices": [item["target_frame_indices"] for item in batch],
    }

    # Trajectories have variable length - keep as list
    result["trajectory_w2c"] = [item["trajectory_w2c"] for item in batch]
    result["trajectory_intrinsics"] = [item["trajectory_intrinsics"] for item in batch]
    result["total_frames"] = [item["total_frames"] for item in batch]

    return result


class DL3DVDatasetTrajectory(Dataset):
    """
    Dataset for DL3DV that returns complete scene trajectories for GEN3C evaluation.

    Mirrors RealEstate10KDatasetWithEulerRawTrajectory's interface but reads from H5 files.

    Each H5 file contains:
        - 'cameras': (N, 18) float32 — same format as RE10K (normalized intrinsics + w2c)
        - 'images': group with keys '0', '1', ... containing JPEG-encoded bytes

    Returns:
        Dict containing:
            - scene_id: str
            - input_images: (T_in, H, W, 3) uint8 [0, 255] for cache creation
            - input_w2c: (T_in, 4, 4) world-to-camera matrices
            - input_intrinsics: (T_in, 3, 3) normalized intrinsics
            - input_frame_indices: List[int]
            - target_images: (T_out, H, W, 3) uint8 [0, 255] for evaluation
            - target_w2c: (T_out, 4, 4) world-to-camera matrices
            - target_intrinsics: (T_out, 3, 3) normalized intrinsics
            - target_frame_indices: List[int]
            - trajectory_w2c: (N, 4, 4) all scene poses
            - trajectory_intrinsics: (N, 3, 3) all normalized intrinsics
            - total_frames: int
    """

    def __init__(
        self,
        data_path: str,
        train_scenes_path: str = "train-scenes",
        test_scenes_path: str = "test-scenes",
        T_in: int = 2,
        T_out: int = 1,
        is_train: bool = True,
        step_size: int = 10,
        max_scenes: int = None,
        scene_list_file: str = None,
        longest_side: Optional[int] = None,
        interpolation: bool = False,
    ):
        self.data_path = Path(data_path)
        self.T_in = T_in
        self.T_out = T_out
        self.is_train = is_train
        self.step_size = step_size
        self.longest_side = longest_side
        self.interpolation = interpolation
        if self.interpolation:
            assert self.T_in == 2, (
                f"interpolation mode requires T_in=2 (endpoint anchors), got T_in={self.T_in}"
            )
        self._scenes_dir = train_scenes_path if is_train else test_scenes_path

        # Load scene paths
        if scene_list_file is not None:
            scene_list_path = Path(scene_list_file)
            if not scene_list_path.is_absolute():
                # Resolve against data_path first, then fall back to the CWD
                # (repo-relative lists like assets/*_test_scenes_100.txt).
                candidate = self.data_path / scene_list_file
                scene_list_path = candidate if candidate.exists() else scene_list_path
            with open(scene_list_path, "r") as f:
                self.scene_paths = [line.strip() for line in f.readlines() if line.strip()]
        else:
            scenes_dir = self.data_path / self._scenes_dir
            self.scene_paths = sorted([
                f.name for f in scenes_dir.iterdir() if f.suffix == '.h5'
            ])

        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]

        print(f"DL3DVDatasetTrajectory: Found {len(self.scene_paths)} scenes "
              f"(is_train={is_train}, scene_list_file={scene_list_file}, max_scenes={max_scenes})")

    def _convert_image(self, image_bytes) -> np.ndarray:
        """Convert JPEG bytes to numpy array of shape (H, W, 3)."""
        decoded = io.decode_image(torch.as_tensor(image_bytes), mode="RGB").permute(1, 2, 0).numpy()
        return decoded

    def _extract_all_cameras(
        self,
        cameras: np.ndarray,  # (N, 18)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract all camera poses. Returns NORMALIZED intrinsics to match
        RealEstate10KDatasetWithEulerRawTrajectory format (GEN3C expects normalized [0,1]).
        """
        selected = torch.from_numpy(cameras.astype(np.float32))
        N = selected.shape[0]

        intrinsics_3x3 = torch.zeros((N, 3, 3), dtype=torch.float32)
        intrinsics_3x3[:, 0, 0] = selected[:, 0]  # fx_norm
        intrinsics_3x3[:, 1, 1] = selected[:, 1]  # fy_norm
        intrinsics_3x3[:, 0, 2] = selected[:, 2]  # cx_norm
        intrinsics_3x3[:, 1, 2] = selected[:, 3]  # cy_norm
        intrinsics_3x3[:, 2, 2] = 1.0

        extrinsics_3x4 = selected[:, 6:].reshape(N, 3, 4)
        w2c = torch.zeros((N, 4, 4), dtype=torch.float32)
        w2c[:, :3, :] = extrinsics_3x4
        w2c[:, 3, 3] = 1.0

        return w2c, intrinsics_3x3

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        scene_file = self.scene_paths[idx]
        h5_path = _resolve_scene_path(self.data_path / self._scenes_dir, scene_file, (".h5",))

        try:
            with h5py.File(str(h5_path), 'r') as f:
                cameras = f['cameras'][:]  # (N, 18) float32
                total_frames = len(f['images'])

                required_frames = (self.T_in + self.T_out - 1) * self.step_size + 1
                if total_frames < required_frames:
                    logging.warning(
                        f"DL3DV scene {scene_file} has {total_frames} frames, "
                        f"need {required_frames} for T_in={self.T_in}, T_out={self.T_out}, "
                        f"step_size={self.step_size}. Skipping."
                    )
                    return None

                # Frame indices: same logic as RealEstate10KDatasetWithEulerRawTrajectory
                if self.interpolation:
                    # Inputs at temporal endpoints; targets sampled in between.
                    total_views = self.T_in + self.T_out
                    all_indices = [i * self.step_size for i in range(total_views)]
                    input_frame_indices = [all_indices[0], all_indices[-1]]
                    target_frame_indices = all_indices[1:-1]
                else:
                    input_frame_indices = [i * self.step_size for i in range(self.T_in)]
                    target_frame_indices = [(self.T_in + i) * self.step_size for i in range(self.T_out)]

                max_idx = max(max(input_frame_indices), max(target_frame_indices))
                if max_idx >= total_frames:
                    logging.warning(
                        f"DL3DV scene {scene_file}: max index {max_idx} >= total_frames {total_frames}. Skipping."
                    )
                    return None

                # Decode only input + target images (not all frames)
                input_imgs = [
                    self._convert_image(f['images'][str(i)][:])
                    for i in input_frame_indices
                ]
                target_imgs = [
                    self._convert_image(f['images'][str(i)][:])
                    for i in target_frame_indices
                ]

        except Exception as e:
            logging.warning(f"Failed to load DL3DV scene {scene_file}: {e}")
            return None

        input_images = torch.from_numpy(np.stack(input_imgs, axis=0)).to(torch.uint8)
        target_images = torch.from_numpy(np.stack(target_imgs, axis=0)).to(torch.uint8)

        # Longest-side resize (preserves aspect ratio, result divisible by 8)
        if self.longest_side is not None:
            H_cur, W_cur = input_images.shape[1], input_images.shape[2]
            cur_longest = max(H_cur, W_cur)
            if cur_longest != self.longest_side:
                scale = self.longest_side / cur_longest
                H_new = round(H_cur * scale / 8) * 8
                W_new = round(W_cur * scale / 8) * 8
                input_images = F.interpolate(
                    input_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new), mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                target_images = F.interpolate(
                    target_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new), mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        # Extract all camera poses for trajectory (normalized intrinsics)
        try:
            trajectory_w2c, trajectory_intrinsics = self._extract_all_cameras(cameras)
        except Exception as e:
            logging.warning(f"Failed to extract camera poses for DL3DV {scene_file}: {e}")
            return None

        input_w2c = trajectory_w2c[input_frame_indices]
        input_intrinsics = trajectory_intrinsics[input_frame_indices]
        target_w2c = trajectory_w2c[target_frame_indices]
        target_intrinsics = trajectory_intrinsics[target_frame_indices]

        return {
            "scene_id": f"dl3dv/{scene_file.replace('.h5', '')}",

            "input_images": input_images,
            "input_w2c": input_w2c,
            "input_intrinsics": input_intrinsics,
            "input_frame_indices": input_frame_indices,

            "target_images": target_images,
            "target_w2c": target_w2c,
            "target_intrinsics": target_intrinsics,
            "target_frame_indices": target_frame_indices,

            "trajectory_w2c": trajectory_w2c,
            "trajectory_intrinsics": trajectory_intrinsics,
            "total_frames": total_frames,
        }


class MipNeRF360DatasetTrajectory(Dataset):
    """
    Mip-NeRF 360 trajectory dataset for GEN3C-style evaluation.

    Reads COLMAP reconstructions under `<data_path>/<scene>/sparse/0/` and
    the corresponding `images_<factor>/` directory. Returns the same dict
    shape as DL3DVDatasetTrajectory so it plugs into `src/evaluate_gen3c.py`
    without further adapter code.

    Unlike the DL3DV / RE10K trajectory datasets, `scene_list_file` is ignored:
    every scene under `data_path` with a COLMAP reconstruction is evaluated,
    matching the convention used by `scripts/evaluate_ours_mipnerf360.sh`.
    """

    def __init__(
        self,
        data_path: str,
        T_in: int = 1,
        T_out: int = 7,
        step_size: int = 4,
        max_scenes: int = None,
        longest_side: Optional[int] = None,
        factor: int = 4,
        interpolation: bool = False,
    ):
        self.data_path = Path(data_path)
        self.T_in = T_in
        self.T_out = T_out
        self.step_size = step_size
        self.longest_side = longest_side
        self.factor = int(factor)
        self.interpolation = interpolation
        if self.interpolation:
            assert self.T_in == 2, (
                f"interpolation mode requires T_in=2 (endpoint anchors), got T_in={self.T_in}"
            )

        self.scene_paths = sorted([
            p.name for p in self.data_path.iterdir()
            if p.is_dir() and (p / "sparse" / "0" / "cameras.bin").exists()
        ])

        if max_scenes is not None and max_scenes > 0:
            self.scene_paths = self.scene_paths[:max_scenes]

        print(f"MipNeRF360DatasetTrajectory: Found {len(self.scene_paths)} scenes "
              f"(factor={self.factor}, max_scenes={max_scenes})")

    def _images_dir_name(self) -> str:
        return "images" if self.factor == 1 else f"images_{self.factor}"

    @staticmethod
    def _build_cameras_array(cameras_dict, sorted_imgs) -> np.ndarray:
        """Build an (N, 18) cameras array with normalized intrinsics.
        Shared encoding with RE10K / DL3DV: [fx_norm, fy_norm, cx_norm, cy_norm, 0, 0, <w2c 3x4 flat>].
        """
        from depth_anything_3.utils.read_write_model import qvec2rotmat

        N = len(sorted_imgs)
        arr = np.zeros((N, 18), dtype=np.float32)
        for i, im in enumerate(sorted_imgs):
            cam = cameras_dict[im.camera_id]
            fx, fy, cx, cy, W, H = MipNeRF360Dataset._colmap_params(cam)
            arr[i, 0] = fx / W
            arr[i, 1] = fy / H
            arr[i, 2] = cx / W
            arr[i, 3] = cy / H

            R = qvec2rotmat(np.asarray(im.qvec, dtype=np.float64)).astype(np.float32)
            t = np.asarray(im.tvec, dtype=np.float32).reshape(3, 1)
            w2c_3x4 = np.concatenate([R, t], axis=1)
            arr[i, 6:] = w2c_3x4.flatten()
        return arr

    @staticmethod
    def _extract_all_cameras(cameras: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same normalized-intrinsics output as DL3DVDatasetTrajectory._extract_all_cameras."""
        selected = torch.from_numpy(cameras.astype(np.float32))
        N = selected.shape[0]

        intrinsics_3x3 = torch.zeros((N, 3, 3), dtype=torch.float32)
        intrinsics_3x3[:, 0, 0] = selected[:, 0]
        intrinsics_3x3[:, 1, 1] = selected[:, 1]
        intrinsics_3x3[:, 0, 2] = selected[:, 2]
        intrinsics_3x3[:, 1, 2] = selected[:, 3]
        intrinsics_3x3[:, 2, 2] = 1.0

        extrinsics_3x4 = selected[:, 6:].reshape(N, 3, 4)
        w2c = torch.zeros((N, 4, 4), dtype=torch.float32)
        w2c[:, :3, :] = extrinsics_3x4
        w2c[:, 3, 3] = 1.0
        return w2c, intrinsics_3x3

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, idx):
        from depth_anything_3.utils.read_write_model import (
            read_cameras_binary, read_images_binary,
        )

        scene_name = self.scene_paths[idx]
        scene_dir = self.data_path / scene_name
        sparse_dir = scene_dir / "sparse" / "0"
        img_dir = scene_dir / self._images_dir_name()

        try:
            cameras_dict = read_cameras_binary(str(sparse_dir / "cameras.bin"))
            images_dict = read_images_binary(str(sparse_dir / "images.bin"))

            sorted_imgs = sorted(images_dict.values(), key=lambda im: im.name)
            total_frames = len(sorted_imgs)

            required_frames = (self.T_in + self.T_out - 1) * self.step_size + 1
            if total_frames < required_frames:
                logging.warning(
                    f"MipNeRF360 scene {scene_name} has {total_frames} frames, "
                    f"need {required_frames} for T_in={self.T_in}, T_out={self.T_out}, "
                    f"step_size={self.step_size}. Skipping."
                )
                return None

            if self.interpolation:
                total_views = self.T_in + self.T_out
                all_indices = [i * self.step_size for i in range(total_views)]
                input_frame_indices = [all_indices[0], all_indices[-1]]
                target_frame_indices = all_indices[1:-1]
            else:
                input_frame_indices = [i * self.step_size for i in range(self.T_in)]
                target_frame_indices = [(self.T_in + i) * self.step_size for i in range(self.T_out)]

            max_idx = max(max(input_frame_indices), max(target_frame_indices))
            if max_idx >= total_frames:
                logging.warning(
                    f"MipNeRF360 scene {scene_name}: max index {max_idx} >= total_frames {total_frames}. Skipping."
                )
                return None

            def _load(i):
                img_path = img_dir / sorted_imgs[i].name
                return io.read_image(str(img_path), mode=io.ImageReadMode.RGB).permute(1, 2, 0).numpy()

            input_imgs = [_load(i) for i in input_frame_indices]
            target_imgs = [_load(i) for i in target_frame_indices]

            cameras = self._build_cameras_array(cameras_dict, sorted_imgs)

        except Exception as e:
            logging.warning(f"Failed to load MipNeRF360 scene {scene_name}: {e}")
            return None

        input_images = torch.from_numpy(np.stack(input_imgs, axis=0)).to(torch.uint8)
        target_images = torch.from_numpy(np.stack(target_imgs, axis=0)).to(torch.uint8)

        if self.longest_side is not None:
            H_cur, W_cur = input_images.shape[1], input_images.shape[2]
            cur_longest = max(H_cur, W_cur)
            if cur_longest != self.longest_side:
                scale = self.longest_side / cur_longest
                H_new = round(H_cur * scale / 8) * 8
                W_new = round(W_cur * scale / 8) * 8
                input_images = F.interpolate(
                    input_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new), mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                target_images = F.interpolate(
                    target_images.permute(0, 3, 1, 2).float(),
                    size=(H_new, W_new), mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)

        try:
            trajectory_w2c, trajectory_intrinsics = self._extract_all_cameras(cameras)
        except Exception as e:
            logging.warning(f"Failed to extract camera poses for MipNeRF360 {scene_name}: {e}")
            return None

        input_w2c = trajectory_w2c[input_frame_indices]
        input_intrinsics = trajectory_intrinsics[input_frame_indices]
        target_w2c = trajectory_w2c[target_frame_indices]
        target_intrinsics = trajectory_intrinsics[target_frame_indices]

        return {
            "scene_id": f"mipnerf360/{scene_name}",

            "input_images": input_images,
            "input_w2c": input_w2c,
            "input_intrinsics": input_intrinsics,
            "input_frame_indices": input_frame_indices,

            "target_images": target_images,
            "target_w2c": target_w2c,
            "target_intrinsics": target_intrinsics,
            "target_frame_indices": target_frame_indices,

            "trajectory_w2c": trajectory_w2c,
            "trajectory_intrinsics": trajectory_intrinsics,
            "total_frames": total_frames,
        }


class RepeatDataset(Dataset):
    """Wraps a dataset to appear repeat_factor times larger via modular indexing."""

    def __init__(self, dataset: Dataset, repeat_factor: int):
        self.dataset = dataset
        self.repeat_factor = max(1, repeat_factor)

    def __len__(self):
        return len(self.dataset) * self.repeat_factor

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


def collate_fn_raw(batch):
    """
    Collate function for RealEstate10KDatasetRaw and RealEstate10KDatasetWithEulerRaw.
    Filters out None samples and stacks raw images.
    Optionally handles camera poses if present.
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Stack raw images into (B, T, H, W, 3), resizing if resolutions differ
    images_list = [item["raw_images"] for item in batch]  # each (T, H, W, 3)
    target_h = min(img.shape[1] for img in images_list)
    target_w = min(img.shape[2] for img in images_list)
    needs_resize = any(img.shape[1] != target_h or img.shape[2] != target_w for img in images_list)
    if needs_resize:
        import torch.nn.functional as _F
        resized = []
        for img in images_list:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                img = _F.interpolate(
                    img.permute(0, 3, 1, 2).float(),
                    size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                ).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            resized.append(img)
        images_list = resized
    raw_images = torch.stack(images_list, dim=0)
    scene_ids = [item["scene_id"] for item in batch]
    frame_indices = [item["frame_indices"] for item in batch]

    result = {
        "scene_ids": scene_ids,
        "raw_images": raw_images,
        "frame_indices": frame_indices,
    }

    # Pass through dynamic T metadata if present
    if "dynamic_T" in batch[0] and batch[0]["dynamic_T"]:
        result["T_in_range"] = batch[0]["T_in_range"]
        result["num_views"] = batch[0]["num_views"]
        result["dynamic_T"] = True

    # Add dynamic masks if available
    if "dyn_masks" in batch[0]:
        dyn_masks_list = [item["dyn_masks"] for item in batch if item.get("dyn_masks") is not None]
        if len(dyn_masks_list) == len(batch):
            # Resize to common resolution if needed (same target_h/target_w as images)
            if needs_resize:
                import torch.nn.functional as _F
                resized_masks = []
                for m in dyn_masks_list:
                    if m.shape[2] != target_h or m.shape[3] != target_w:
                        m = _F.interpolate(m, size=(target_h, target_w), mode="nearest")
                    resized_masks.append(m)
                dyn_masks_list = resized_masks
            result["dyn_masks"] = torch.stack(dyn_masks_list, dim=0)  # (B, T, 1, H, W)

    # Add camera poses if available
    if "w2c" in batch[0]:
        w2c_batch = [item["w2c"] for item in batch if item["w2c"] is not None]
        if len(w2c_batch) == len(batch):
            result["w2c"] = torch.stack(w2c_batch, dim=0)

    if "intrinsics" in batch[0]:
        intrinsics_batch = [item["intrinsics"] for item in batch if item["intrinsics"] is not None]
        if len(intrinsics_batch) == len(batch):
            result["intrinsics"] = torch.stack(intrinsics_batch, dim=0)

    # Add precomputed depths if available
    if "depths" in batch[0]:
        depths_batch = [item["depths"] for item in batch if item.get("depths") is not None]
        if len(depths_batch) == len(batch):
            if needs_resize:
                import torch.nn.functional as _F
                resized_depths = []
                for d in depths_batch:
                    if d.shape[1] != target_h or d.shape[2] != target_w:
                        d = _F.interpolate(
                            d.unsqueeze(1).float(), size=(target_h, target_w),
                            mode="bilinear", align_corners=False,
                        ).squeeze(1)
                    resized_depths.append(d)
                depths_batch = resized_depths
            result["depths"] = torch.stack(depths_batch, dim=0)  # (B, T, H, W)

    # Held-out frames (per-scene, variable Nh): keep as a list of per-batch
    # entries since batch_size=1 in eval and Nh may vary between scenes.
    if "heldout_raw_images" in batch[0]:
        result["heldout_raw_images"] = [item.get("heldout_raw_images") for item in batch]
        result["heldout_w2c"] = [item.get("heldout_w2c") for item in batch]
        result["heldout_intrinsics"] = [item.get("heldout_intrinsics") for item in batch]
        result["heldout_frame_indices"] = [item.get("heldout_frame_indices", []) for item in batch]
        if "heldout_depths" in batch[0]:
            result["heldout_depths"] = [item.get("heldout_depths") for item in batch]

    return result
