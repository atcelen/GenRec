"""
General novel-view-synthesis inference: input image(s) + camera poses -> novel views.

T_in is inferred from the number of images in --input_dir. Target views come from
one of two pose sources:
  * --poses <poses.json>  : render exactly the camera poses you specify (the same
                            schema this tool emits, so you can generate once, edit
                            poses.json, and re-run). T_out = number of target poses.
  * (no --poses)          : synthesize poses. Single image -> a trajectory around it
                            (--trajectory orbit|dolly|spiral|wiggle). Multi-image ->
                            SLERP interpolation between the input viewpoints.
                            T_out = --total_views - T_in.

DA3 estimates depth + camera poses + intrinsics for the inputs. Poses live in DA3's
coordinate frame (cam0 = identity, median camera distance ~= 1), so user-supplied
target poses should be expressed in that frame (an edited poses.json already is).
The model runs with source_only_da3=True so DA3 only re-runs on the source views.

Outputs (in --output_dir):
  frames/frame_{i:03d}.png   - T_out generated novel views
  inputs/input_{i:03d}.png   - DA3 processed inputs at model resolution
  depths/depth_{i:03d}.npy   - DA3-inferred input depths
  poses.json                 - input_c2w, target_c2w, full_c2w, intrinsics, T_in, T_out

NOTE: input images are resized so longest side = 616, then center-cropped to ~16:9
(W=616, H=336) to match the RE10K/DL3DV training distribution. Portrait photos will
lose top/bottom content.

Usage (run on a GPU compute node):

    # Synthesize a trajectory around a single photo:
    python -m genrec.cli.inference \\
        --input_dir ./my_photos --output_dir ./custom_out \\
        --config configs/dl3dv.yaml --checkpoint genrec \\
        --trajectory orbit --motion_scale 0.3 --total_views 8

    # Render user-specified camera poses (e.g. an edited poses.json from a prior run):
    python -m genrec.cli.inference \\
        --input_dir ./my_photos --output_dir ./custom_out \\
        --config configs/dl3dv.yaml --checkpoint genrec \\
        --poses ./custom_out/poses.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from genrec.utils.config import load_experiment_config
from genrec.utils.traj_utils import (
    _rotmat_to_quat,
    _quat_to_rotmat,
    _slerp_pair,
)
from genrec.cli.evaluate_ours import OursEvaluator


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    input_dir: Path
    output_dir: Path
    checkpoint: str
    config: Path
    total_views: int = 8
    trajectory: str = "orbit"          # orbit | dolly | spiral | wiggle
    motion_scale: float = 0.3
    num_inference_steps: int = 25
    guidance_scale: float = 1.0
    seed: int = 42


# -----------------------------------------------------------------------------
# Image loading
# -----------------------------------------------------------------------------

LONGEST_SIDE = 616          # matches data.longest_side in RE10K/DL3DV configs
ASPECT_W, ASPECT_H = 616, 336   # ~16:9, both divisible by 56 (DA3 patch grid)
PATCH_MULTIPLE = 56

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_resize_images(input_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """Load all images from input_dir, resize longest side to 616, center-crop to 16:9.

    Returns:
        imgs:      (T_in, H, W, 3) uint8 with H=ASPECT_H, W=ASPECT_W.
        filenames: sorted list of filenames (basename).
    """
    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise ValueError(f"No images found in {input_dir} (supported: {sorted(IMG_EXTS)})")

    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        scale = LONGEST_SIDE / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center-crop to ASPECT_W x ASPECT_H. If the resized image is smaller in
        # the short dimension than ASPECT_H/W, pad with edge replication.
        arr = np.asarray(img)  # (H, W, 3)
        H, W = arr.shape[:2]

        if H < ASPECT_H or W < ASPECT_W:
            pad_h = max(0, ASPECT_H - H)
            pad_w = max(0, ASPECT_W - W)
            arr = np.pad(
                arr,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                mode="edge",
            )
            H, W = arr.shape[:2]

        y0 = (H - ASPECT_H) // 2
        x0 = (W - ASPECT_W) // 2
        arr = arr[y0 : y0 + ASPECT_H, x0 : x0 + ASPECT_W]
        assert arr.shape == (ASPECT_H, ASPECT_W, 3), f"unexpected crop shape: {arr.shape}"
        out.append(arr)

    imgs = np.stack(out, axis=0).astype(np.uint8)
    return imgs, [p.name for p in paths]


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def build_model(cfg_path: Path, ckpt: str, device: torch.device,
                overrides: List[str] | None = None):
    """Instantiate OursEvaluator with stubbed kwargs, call build_model(), return .model.

    `ckpt` is a checkpoint spec: a registry name ("genrec", auto-downloaded from
    HuggingFace), an http(s) URL, or a local .pth path (resolved by OursEvaluator).

    `overrides` mirrors genrec/cli/evaluate_ours.py: positional key=value pairs applied to the
    YAML config, e.g. ["model.num_views=16", "model.num_input_views=2"]. These flow into
    `model_opts` so the UNet is built with the matching multi-view attention shape.
    """
    cfg, model_opts, _ = load_experiment_config(cfg_path, overrides or [])

    evaluator = OursEvaluator(
        cfg=cfg,
        model_opts=model_opts,
        device=device,
        checkpoint_path=str(ckpt),
        output_dir="/tmp/_inference_unused",
        scene_list=None,
        step_size=1,
        seed=0,
        num_inference_steps=25,
        rank=0,
        world_size=1,
        save_images=False,
        save_video=False,
        skip_coord_metrics=True,
        use_gt_poses=True,        # we'll supply DA3-derived or user-supplied poses
        source_only_da3=True,     # only run DA3 on source views
    )
    evaluator.build_model()
    return evaluator.model


def load_genrec(config, weights: str = "genrec", device=None, overrides=None):
    """One-line model loader: `model = load_genrec("configs/dl3dv.yaml")`.

    `weights` is a registry name ("genrec"), an http(s) URL, or a local .pth path.
    Returns a ready-to-run model whose `evaluate_with_raw_images(...)` takes images
    + camera poses and returns novel views.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_model(Path(config), weights, device, overrides=overrides)


# -----------------------------------------------------------------------------
# DA3 standalone call
# -----------------------------------------------------------------------------

def run_da3_standalone(model, imgs_uint8: np.ndarray):
    """Run DA3 on the input photos with no GT poses/intrinsics.

    DA3 returns poses in its own normalized frame (cam0 = identity, median
    camera distance ~= 1). We accept that frame and build our trajectory in it.

    Args:
        imgs_uint8: (T_in, H, W, 3) uint8.
    Returns:
        w2c: (T_in, 4, 4) np.float32
        K:   (T_in, 3, 3) np.float32 - intrinsics at DA3's process resolution
        depth_proc: (T_in, H_proc, W_proc) np.float32 - depth at DA3 process resolution
    """
    da3 = model.geometry_model  # GeometryModel is a DepthAnything3 subclass
    img_list = [imgs_uint8[i] for i in range(imgs_uint8.shape[0])]

    # GeometryModel.inference returns (prediction, scale); use the same process_res
    # the model uses internally so depth/intrinsics resolution matches.
    process_res = getattr(model, "geometry_process_res", 504)
    prediction, _scale = da3.inference(
        image=img_list,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        process_res=process_res,
        process_res_method="upper_bound_resize",
    )

    ext = np.asarray(prediction.extrinsics, dtype=np.float32)  # (N, 4, 4) or (N, 3, 4)
    if ext.shape[-2:] == (3, 4):
        # homogenize
        homo = np.zeros((ext.shape[0], 4, 4), dtype=np.float32)
        homo[:, :3, :] = ext
        homo[:, 3, 3] = 1.0
        ext = homo
    K = np.asarray(prediction.intrinsics, dtype=np.float32)    # (N, 3, 3)
    depth = np.asarray(prediction.depth, dtype=np.float32)     # (N, H_proc, W_proc)
    return ext, K, depth


# -----------------------------------------------------------------------------
# Trajectory generators (single-image)
# -----------------------------------------------------------------------------
# All return c2w array (T_out, 4, 4) in DA3 normalized frame. OpenCV convention:
# camera local frame is +X right, +Y down, +Z forward.

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build c2w (4x4) so the camera at `eye` looks at `target`. OpenCV: +Z forward, +Y down.

    `up` is the desired world direction that should align with the camera's -Y axis
    (i.e. the up-in-image direction).
    """
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    # Camera +Z = forward
    # Camera Y = -up (image up is -Y in OpenCV); compute right = forward x (-up_world)
    y_axis = -up / (np.linalg.norm(up) + 1e-12)
    # Re-orthogonalize: x = y x z
    x_axis = np.cross(y_axis, forward)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(forward, x_axis)
    R = np.stack([x_axis, y_axis, forward], axis=1)  # columns = camera axes in world
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.astype(np.float32)
    c2w[:3, 3] = eye.astype(np.float32)
    return c2w


def generate_orbit(c2w_in: np.ndarray, depth_med: float, T_out: int, scale: float) -> np.ndarray:
    """Orbit around a forward look-at point at depth_med along input camera's +Z.

    Sweeps a small angular range so we don't extrapolate too far. Radius =
    scale * depth_med; angular sweep total = ~2*scale radians (capped).
    """
    R_in = c2w_in[:3, :3]
    t_in = c2w_in[:3, 3]
    # World-space basis of input camera
    x_cam = R_in[:, 0]
    y_cam = R_in[:, 1]
    z_cam = R_in[:, 2]
    look_at = t_in + depth_med * z_cam

    radius = scale * depth_med
    half_sweep = min(2.0 * scale, math.pi / 3)  # cap at +/- 60 degrees
    angles = np.linspace(-half_sweep, half_sweep, T_out, dtype=np.float64)

    poses = np.zeros((T_out, 4, 4), dtype=np.float32)
    up_world = -y_cam  # world-up direction (image-up = -Y_cam)
    for i, a in enumerate(angles):
        eye = look_at - radius * (math.cos(a) * z_cam + math.sin(a) * x_cam)
        # Stay at same world height (don't drift along y) - already preserved by basis.
        poses[i] = _look_at(eye, look_at, up_world)
    return poses


def generate_dolly(c2w_in: np.ndarray, depth_med: float, T_out: int, scale: float) -> np.ndarray:
    """Translate along the input camera's forward axis (+Z) by [0, scale * depth_med]."""
    R_in = c2w_in[:3, :3]
    t_in = c2w_in[:3, 3]
    z_cam = R_in[:, 2]
    steps = np.linspace(0.0, scale * depth_med, T_out, dtype=np.float64)
    poses = np.tile(c2w_in[None, :, :], (T_out, 1, 1)).astype(np.float32)
    for i, s in enumerate(steps):
        poses[i, :3, 3] = (t_in + s * z_cam).astype(np.float32)
    return poses


def generate_spiral(c2w_in: np.ndarray, depth_med: float, T_out: int, scale: float) -> np.ndarray:
    """Orbit overlaid with a linear forward dolly."""
    R_in = c2w_in[:3, :3]
    t_in = c2w_in[:3, 3]
    x_cam = R_in[:, 0]
    y_cam = R_in[:, 1]
    z_cam = R_in[:, 2]
    look_at = t_in + depth_med * z_cam
    radius = scale * depth_med
    half_sweep = min(2.0 * scale, math.pi / 3)
    angles = np.linspace(-half_sweep, half_sweep, T_out, dtype=np.float64)
    forward_offsets = np.linspace(0.0, 0.3 * scale * depth_med, T_out, dtype=np.float64)

    poses = np.zeros((T_out, 4, 4), dtype=np.float32)
    up_world = -y_cam
    for i, (a, f) in enumerate(zip(angles, forward_offsets)):
        eye = look_at - radius * (math.cos(a) * z_cam + math.sin(a) * x_cam) + f * z_cam
        poses[i] = _look_at(eye, look_at, up_world)
    return poses


def generate_wiggle(c2w_in: np.ndarray, depth_med: float, T_out: int, scale: float,
                    seed: int = 0) -> np.ndarray:
    """Per-frame small SE(3) jitter around the input pose. Translation sigma =
    0.1 * scale * depth_med; rotation sigma = 3 degrees."""
    rng = np.random.default_rng(seed)
    R_in = c2w_in[:3, :3]
    t_in = c2w_in[:3, 3]
    sigma_t = 0.1 * scale * depth_med
    sigma_r = math.radians(3.0)

    poses = np.zeros((T_out, 4, 4), dtype=np.float32)
    poses[:, 3, 3] = 1.0
    for i in range(T_out):
        dt = rng.normal(scale=sigma_t, size=3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis) + 1e-12
        angle = rng.normal(scale=sigma_r)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        dR = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        poses[i, :3, :3] = (dR @ R_in).astype(np.float32)
        poses[i, :3, 3] = (t_in + dt).astype(np.float32)
    return poses


SINGLE_TRAJECTORIES = {
    "orbit": generate_orbit,
    "dolly": generate_dolly,
    "spiral": generate_spiral,
    "wiggle": generate_wiggle,
}


def dispatch_single_image_trajectory(name: str, c2w_in: np.ndarray, depth_med: float,
                                     T_out: int, scale: float, seed: int) -> np.ndarray:
    if name not in SINGLE_TRAJECTORIES:
        raise ValueError(f"Unknown trajectory '{name}'. Options: {list(SINGLE_TRAJECTORIES)}")
    if name == "wiggle":
        return generate_wiggle(c2w_in, depth_med, T_out, scale, seed=seed)
    return SINGLE_TRAJECTORIES[name](c2w_in, depth_med, T_out, scale)


# -----------------------------------------------------------------------------
# Multi-image SLERP trajectory
# -----------------------------------------------------------------------------

def interpolate_multi_image_trajectory(
    w2c_in: np.ndarray, K_in: np.ndarray, T_out: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Distribute T_out interpolated target c2w poses across the (T_in - 1) input segments
    via SLERP. Returns target_c2w (T_out, 4, 4) and target_K (T_out, 3, 3)."""
    T_in = w2c_in.shape[0]
    if T_in < 2:
        raise ValueError("Multi-image interpolation requires T_in >= 2")

    # w2c -> c2w
    c2w_in = np.linalg.inv(w2c_in).astype(np.float64)

    # Sample target t's uniformly in [0, T_in - 1], EXCLUDING the keyframe indices
    # so we generate strictly new in-between views.
    n_segments = T_in - 1
    # Distribute T_out across segments as evenly as possible.
    per_seg = [T_out // n_segments] * n_segments
    for i in range(T_out % n_segments):
        per_seg[i] += 1

    target_c2w_list = []
    target_K_list = []
    quats_in = _rotmat_to_quat(c2w_in[:, :3, :3])
    trans_in = c2w_in[:, :3, 3]
    K64 = K_in.astype(np.float64)

    for seg in range(n_segments):
        k = per_seg[seg]
        if k == 0:
            continue
        # Place k samples strictly inside the segment (exclusive of endpoints).
        ts = (np.arange(k, dtype=np.float64) + 1.0) / (k + 1.0)
        q_seg = _slerp_pair(quats_in[seg], quats_in[seg + 1], ts)
        R_seg = _quat_to_rotmat(q_seg)
        t_seg = (1.0 - ts)[:, None] * trans_in[seg] + ts[:, None] * trans_in[seg + 1]
        K_seg = (1.0 - ts)[:, None, None] * K64[seg] + ts[:, None, None] * K64[seg + 1]
        for j in range(k):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_seg[j].astype(np.float32)
            pose[:3, 3] = t_seg[j].astype(np.float32)
            target_c2w_list.append(pose)
            target_K_list.append(K_seg[j].astype(np.float32))

    target_c2w = np.stack(target_c2w_list, axis=0)
    target_K = np.stack(target_K_list, axis=0)
    return target_c2w, target_K


# -----------------------------------------------------------------------------
# User-supplied poses (--poses poses.json)
# -----------------------------------------------------------------------------

def load_poses_json(path: Path, T_in: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load target (and optional source) camera poses from a poses.json file.

    Accepts the schema this tool emits. Recognized keys:
      * "target_c2w"  (required): list of T_out 4x4 camera-to-world matrices.
      * "target_K" or "intrinsics" (optional): per-target 3x3 intrinsics, or the
        full (T_in + T_out) intrinsics list (the target slice is taken).
      * "input_c2w"   (optional): list of T_in 4x4 source camera-to-world matrices;
        if present, these override DA3's estimated source poses.

    Returns (target_c2w, target_K_or_None, input_c2w_or_None).
    """
    with open(path, "r") as f:
        data = json.load(f)
    if "target_c2w" not in data:
        raise ValueError(f"{path}: missing required key 'target_c2w'.")
    target_c2w = np.asarray(data["target_c2w"], dtype=np.float32)
    if target_c2w.ndim != 3 or target_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"{path}: 'target_c2w' must be a list of 4x4 matrices.")
    T_out = target_c2w.shape[0]

    target_K = None
    if "target_K" in data:
        target_K = np.asarray(data["target_K"], dtype=np.float32)
    elif "intrinsics" in data:
        K_all = np.asarray(data["intrinsics"], dtype=np.float32)
        if K_all.shape[0] == T_in + T_out:
            target_K = K_all[T_in:]
        elif K_all.shape[0] == T_out:
            target_K = K_all
        # otherwise leave None -> fall back to tiled source K
    if target_K is not None and target_K.shape[0] != T_out:
        raise ValueError(f"{path}: intrinsics count ({target_K.shape[0]}) != target poses ({T_out}).")

    input_c2w = None
    if "input_c2w" in data:
        input_c2w = np.asarray(data["input_c2w"], dtype=np.float32)
        if input_c2w.shape[0] != T_in:
            raise ValueError(
                f"{path}: 'input_c2w' has {input_c2w.shape[0]} poses but {T_in} input image(s) were loaded."
            )
    return target_c2w, target_K, input_c2w


def resolve_target_poses(args, user_target_c2w, user_target_K,
                         c2w_in, K_in, w2c_in, depth_med, T_out):
    """Resolve the target camera trajectory. Returns (target_c2w, target_K, label).

    Priority: user-supplied poses (--poses) > generated trajectory (single image) >
    SLERP interpolation (multi image).
    """
    if user_target_c2w is not None:
        target_K = user_target_K if user_target_K is not None else np.tile(K_in[0:1], (T_out, 1, 1))
        return user_target_c2w, target_K, "user_poses"

    T_in = w2c_in.shape[0]
    if T_in == 1:
        target_c2w = dispatch_single_image_trajectory(
            args.trajectory, c2w_in[0], depth_med, T_out, args.motion_scale, args.seed,
        )
        target_K = np.tile(K_in[0:1], (T_out, 1, 1))
        return target_c2w, target_K, args.trajectory

    target_c2w, target_K = interpolate_multi_image_trajectory(w2c_in, K_in, T_out)
    return target_c2w, target_K, "slerp"


# -----------------------------------------------------------------------------
# Output saving
# -----------------------------------------------------------------------------

def _save_image(path: Path, img01: np.ndarray):
    """img01 is (H, W, 3) in [0, 1] float."""
    arr = np.clip(img01 * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def save_outputs(
    out_dir: Path,
    frames_01: torch.Tensor,           # (T_out, 3, H, W) in [0, 1]
    processed_inputs_m11: torch.Tensor,  # (T_in, 3, H_proc, W_proc) in [-1, 1]
    depths: np.ndarray,                # (T_in, H_proc, W_proc) np.float32
    full_c2w: np.ndarray,              # (T, 4, 4)
    K_out: np.ndarray,                 # (T, 3, 3)
    T_in: int,
    T_out: int,
    trajectory_type: str,
    image_filenames: List[str],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    (out_dir / "inputs").mkdir(exist_ok=True)
    (out_dir / "depths").mkdir(exist_ok=True)

    frames = frames_01.detach().cpu().numpy()
    for i in range(frames.shape[0]):
        _save_image(out_dir / "frames" / f"frame_{i:03d}.png",
                    np.transpose(frames[i], (1, 2, 0)))

    inputs = ((processed_inputs_m11.detach().cpu().numpy() + 1.0) / 2.0)
    for i in range(inputs.shape[0]):
        _save_image(out_dir / "inputs" / f"input_{i:03d}.png",
                    np.transpose(inputs[i], (1, 2, 0)))

    for i in range(depths.shape[0]):
        np.save(out_dir / "depths" / f"depth_{i:03d}.npy", depths[i])

    poses_payload = {
        "trajectory_type": trajectory_type,
        "T_in": T_in,
        "T_out": T_out,
        "total_views": T_in + T_out,
        "image_filenames": image_filenames,
        "intrinsics": K_out.tolist(),
        "input_c2w": full_c2w[:T_in].tolist(),
        "target_c2w": full_c2w[T_in:].tolist(),
        "full_c2w": full_c2w.tolist(),
    }
    with open(out_dir / "poses.json", "w") as f:
        json.dump(poses_payload, f, indent=2)


# -----------------------------------------------------------------------------
# View-count overrides
# -----------------------------------------------------------------------------

def _maybe_inject_view_overrides(overrides: List[str], T_in: int, total_views: int) -> List[str]:
    """Append model.num_views / model.num_input_views overrides if the user didn't set
    them, so the UNet is built for the actual (T_in, T_out) attention shape."""
    keys = {o.split("=", 1)[0] for o in overrides if "=" in o}
    out = list(overrides)
    if "model.num_views" not in keys:
        out.append(f"model.num_views={total_views}")
    if "model.num_input_views" not in keys:
        out.append(f"model.num_input_views={T_in}")
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--checkpoint", type=str, default="genrec",
                   help="Model checkpoint: a registry name ('genrec', auto-downloaded "
                        "from HuggingFace), an http(s) URL, or a local .pth path.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--poses", type=Path, default=None,
                   help="JSON file of target camera poses (poses.json schema: 'target_c2w', "
                        "optional 'intrinsics'/'target_K', optional 'input_c2w'). When given, "
                        "T_out is the number of target poses and --total_views/--trajectory are ignored.")
    p.add_argument("--total_views", type=int, default=8,
                   help="Total views (T_in + T_out) when synthesizing a trajectory (ignored with --poses).")
    p.add_argument("--trajectory", choices=list(SINGLE_TRAJECTORIES), default="orbit",
                   help="Single-image trajectory type (ignored with --poses or multi-image input).")
    p.add_argument("--motion_scale", type=float, default=0.3)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=1.0,
                   help="Classifier-free guidance scale (passed as guidance_scale to the model).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value form, e.g. 'model.num_views=16 model.num_input_views=2' "
             "(same syntax as genrec/cli/evaluate_ours.py). If omitted, num_views/num_input_views are set "
             "automatically from the actual input/target counts.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load + resize images -------------------------------------------------
    imgs_u8, filenames = load_and_resize_images(args.input_dir)
    T_in = imgs_u8.shape[0]

    # ---- Determine target count + load user poses (if any) --------------------
    user_target_c2w = user_target_K = user_input_c2w = None
    if args.poses is not None:
        user_target_c2w, user_target_K, user_input_c2w = load_poses_json(args.poses, T_in)
        T_out = user_target_c2w.shape[0]
        total_views = T_in + T_out
        print(f"[poses] loaded {T_out} target pose(s) from {args.poses}")
    else:
        total_views = args.total_views
        T_out = total_views - T_in

    if T_in < 1 or T_in > total_views:
        raise ValueError(f"Need 1 <= T_in <= total_views; got T_in={T_in}, total={total_views}")
    if T_out < 1:
        raise ValueError(f"T_out must be >= 1; got {T_out}. Increase --total_views or supply more poses.")
    print(f"[load] {T_in} input image(s) -> T_out = {T_out} target view(s)")

    # ---- Build model ---------------------------------------------------------
    overrides = _maybe_inject_view_overrides(args.overrides, T_in, total_views)
    print(f"[model] loading from {args.checkpoint}")
    if overrides:
        print(f"[model] config overrides: {overrides}")
    model = build_model(args.config, args.checkpoint, device, overrides=overrides)

    # ---- DA3 on inputs -------------------------------------------------------
    print("[da3] running on inputs (no GT poses)...")
    w2c_in, K_proc, depth_proc = run_da3_standalone(model, imgs_u8)
    print(f"[da3] poses shape={w2c_in.shape}, K_proc={K_proc.shape}, depth_proc={depth_proc.shape}")

    # Median depth (positive values only) for trajectory scaling
    pos = depth_proc[depth_proc > 0]
    depth_med = float(np.median(pos)) if pos.size > 0 else 1.0
    print(f"[da3] median depth = {depth_med:.4f}")

    # Rescale DA3 intrinsics from process_res to input H/W (ASPECT_H/W)
    H_proc, W_proc = depth_proc.shape[-2:]
    K_input_res = K_proc.copy().astype(np.float32)
    K_input_res[:, 0, :] *= ASPECT_W / W_proc
    K_input_res[:, 1, :] *= ASPECT_H / H_proc

    # ---- Source poses: honor user-supplied input_c2w if present --------------
    if user_input_c2w is not None:
        print("[poses] using user-supplied source poses (input_c2w)")
        w2c_in = np.linalg.inv(user_input_c2w).astype(np.float32)
    c2w_in = np.linalg.inv(w2c_in).astype(np.float32)

    # ---- Build target trajectory ---------------------------------------------
    target_c2w, target_K, trajectory_type = resolve_target_poses(
        args, user_target_c2w, user_target_K, c2w_in, K_input_res, w2c_in, depth_med, T_out,
    )
    print(f"[traj] target pose source: {trajectory_type}")
    target_w2c = np.linalg.inv(target_c2w).astype(np.float32)

    # ---- Assemble model inputs -----------------------------------------------
    full_w2c = np.concatenate([w2c_in, target_w2c], axis=0)    # (T, 4, 4)
    full_K = np.concatenate([K_input_res, target_K], axis=0)   # (T, 3, 3)
    T_total = full_w2c.shape[0]
    assert T_total == T_in + T_out

    raw_images = np.zeros((1, T_total, ASPECT_H, ASPECT_W, 3), dtype=np.uint8)
    raw_images[0, :T_in] = imgs_u8

    raw_images_t = torch.from_numpy(raw_images).to(device)             # (1, T, H, W, 3)
    full_w2c_t = torch.from_numpy(full_w2c).unsqueeze(0).to(device)    # (1, T, 4, 4)
    full_K_t = torch.from_numpy(full_K).unsqueeze(0).to(device)        # (1, T, 3, 3)

    # ---- Inference -----------------------------------------------------------
    print(f"[inference] T_in={T_in} T_out={T_out} steps={args.num_inference_steps}")
    amp_dtype = getattr(model, "mixed_dtype", torch.float32)
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
        eval_results = model.evaluate_with_raw_images(
            raw_images=raw_images_t,
            scene_ids=["custom"],
            T_in=T_in,
            T_out=T_out,
            num_inference_steps=args.num_inference_steps,
            gt_extrinsics=full_w2c_t,
            gt_intrinsics=full_K_t,
            source_only_da3=True,
            guidance_scale=args.guidance,
        )

    frames = eval_results["imgs"][0]                            # (T_out, 3, H, W) in [0, 1]
    processed = eval_results["processed_images"][0, :T_in]      # (T_in, 3, H_proc, W_proc) in [-1, 1]

    # Final intrinsics to save: match prediction resolution
    pred_H, pred_W = frames.shape[-2:]
    K_save = full_K.copy()
    K_save[:, 0, :] *= pred_W / ASPECT_W
    K_save[:, 1, :] *= pred_H / ASPECT_H

    full_c2w_save = np.linalg.inv(full_w2c).astype(np.float32)

    # ---- Save ----------------------------------------------------------------
    print(f"[save] -> {args.output_dir}")
    save_outputs(
        out_dir=args.output_dir,
        frames_01=frames,
        processed_inputs_m11=processed,
        depths=depth_proc,
        full_c2w=full_c2w_save,
        K_out=K_save,
        T_in=T_in,
        T_out=T_out,
        trajectory_type=trajectory_type,
        image_filenames=filenames,
    )
    print(f"[done] wrote {T_out} frame(s) to {args.output_dir / 'frames'}")


if __name__ == "__main__":
    main()
