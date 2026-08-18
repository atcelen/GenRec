#!/usr/bin/env python3
"""Convert DL3DV-10K scenes to GenRec's per-scene .h5 format.

Output format (one file per scene, see README "Dataset file layout"):

    <scene_id>.h5
    ├── cameras   (N, 18) float32:
    │             [0:4] normalized intrinsics [fx, fy, cx, cy], [4:6] zeros,
    │             [6:18] 3x4 world-to-camera matrix, row-major
    └── images/   group of 1-D uint8 datasets "0".."N-1" (encoded frame bytes)

Camera conversion (adapted from depthsplat's convert_dl3dv scripts, MIT License,
(c) 2024 Haofei Xu — https://github.com/cvg/depthsplat): `transforms.json` holds
Blender/OpenGL camera-to-world matrices; right-multiplying by
diag(1, -1, -1, 1) flips the *camera* frame to OpenCV convention. The *world*
frame is deliberately left in Blender's native +Z-up orientation.

Input: standard DL3DV-10K scene folders

    <input_dir>/<scene_hash>/
    ├── transforms.json
    └── images_4/ (or images_8/, images/) frame_00001.png ...

Usage:

    python scripts/preprocess_dl3dv.py \\
        --input_dir  /path/to/DL3DV-10K \\
        --output_dir /path/to/DL3DV-10K_processed \\
        [--img_subdir images_4] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BLENDER2OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def build_cameras(meta: dict) -> tuple[np.ndarray, list]:
    """transforms.json -> ((N, 18) float32 cameras, matching frame basenames).

    Frames are sorted by their numeric index so the stored order is temporal
    (the loaders sample frames with a temporal step size).
    """
    h, w = meta["h"], meta["w"]
    fx, fy, cx, cy = meta["fl_x"], meta["fl_y"], meta["cx"], meta["cy"]
    intr = [fx / w, fy / h, cx / w, cy / h, 0.0, 0.0]  # normalized, zeros padding

    def frame_index(frame):
        return int(Path(frame["file_path"]).stem.split("_")[-1])

    frames = sorted(meta["frames"], key=frame_index)
    rows, names = [], []
    for frame in frames:
        # Blender c2w -> OpenCV-camera c2w (world frame stays Z-up), then invert.
        opencv_c2w = np.array(frame["transform_matrix"]) @ BLENDER2OPENCV
        w2c = np.linalg.inv(opencv_c2w)[:3].flatten().tolist()
        rows.append(intr + w2c)
        names.append(Path(frame["file_path"]).name)
    return np.asarray(rows, dtype=np.float32), names


def find_image(img_dir: Path, name: str) -> Path | None:
    """Locate a frame file, tolerating extension changes across resolutions."""
    p = img_dir / name
    if p.exists():
        return p
    for ext in IMG_EXTS:
        q = p.with_suffix(ext)
        if q.exists():
            return q
    return None


def convert_scene(scene_dir: Path, out_path: Path, img_subdir: str) -> int:
    meta_path = scene_dir / "transforms.json"
    if not meta_path.is_file():
        # Some DL3DV layouts nest one level deeper (e.g. <scene>/<hash>/...).
        nested = list(scene_dir.glob("*/transforms.json"))
        if len(nested) == 1:
            scene_dir = nested[0].parent
            meta_path = nested[0]
        else:
            raise FileNotFoundError(f"no transforms.json under {scene_dir}")

    img_dir = None
    for cand in (img_subdir, "images_4", "images_8", "images"):
        if (scene_dir / cand).is_dir():
            img_dir = scene_dir / cand
            break
    if img_dir is None:
        raise FileNotFoundError(f"no image directory ({img_subdir}/images_4/images_8/images) in {scene_dir}")

    meta = json.loads(meta_path.read_text())
    cameras, names = build_cameras(meta)

    kept_rows, blobs = [], []
    for i, name in enumerate(names):
        p = find_image(img_dir, name)
        if p is None:
            continue  # frame missing at this resolution — drop its camera row too
        kept_rows.append(cameras[i])
        blobs.append(np.fromfile(str(p), dtype=np.uint8))
    if not blobs:
        raise FileNotFoundError(f"no frames from transforms.json found in {img_dir}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("cameras", data=np.stack(kept_rows))
        grp = f.create_group("images")
        for i, blob in enumerate(blobs):
            grp.create_dataset(str(i), data=blob)
    return len(blobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="DL3DV-10K root: one folder per scene")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--img_subdir", default="images_4",
                        help="Preferred image resolution subdirectory (default: images_4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert only the first N scenes (smoke test)")
    args = parser.parse_args()

    scene_dirs = sorted(d for d in args.input_dir.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
    if args.limit:
        scene_dirs = scene_dirs[:args.limit]
    if not scene_dirs:
        sys.exit(f"No scene directories found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for scene_dir in scene_dirs:
        out_path = args.output_dir / f"{scene_dir.name}.h5"
        try:
            n_frames = convert_scene(scene_dir, out_path, args.img_subdir)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"[skip] {scene_dir.name}: {e}")
            continue
        n_ok += 1
        print(f"[ok] {scene_dir.name}: {n_frames} frames")
    print(f"\nConverted {n_ok}/{len(scene_dirs)} scenes -> {args.output_dir}")


if __name__ == "__main__":
    main()
