#!/usr/bin/env python3
"""Convert RealEstate10K to GenRec's per-scene .torch format.

Output format (one file per scene, see README "Dataset file layout"):

    <scene_id>.torch = {
        "images":  list of 1-D uint8 tensors (encoded JPEG/PNG bytes, one per frame),
        "cameras": float32 tensor (N, 18):
                   [0:4] normalized intrinsics [fx, fy, cx, cy], [4:6] zeros,
                   [6:18] 3x4 world-to-camera matrix, row-major,
    }

Two input modes:

1. Official RealEstate10K release (camera .txt files + frames you extracted
   from the videos):

       python scripts/preprocess_re10k.py \\
           --poses_dir  /path/to/RealEstate10K/test \\
           --frames_dir /path/to/frames \\
           --output_dir /path/to/RealEstate10K_processed

   Each `<poses_dir>/<scene_id>.txt` is the official format: a URL line, then
   one line per frame with 19 numbers
   (timestamp_us, fx, fy, cx, cy, 0, 0, <12 w2c values>). Intrinsics are already
   normalized and the pose is already world-to-camera, so the (N, 18) camera row
   is exactly the line minus the timestamp — no convention conversion happens.
   Frames are looked up as `<frames_dir>/<scene_id>/<timestamp>.<ext>` (falling
   back to sorted order if the names are not timestamps) and stored as their
   raw encoded bytes.

2. pixelSplat / depthsplat chunk format (multi-scene ~100-200 MB .torch chunks
   with {key, url, timestamps, cameras, images} entries) — split into per-scene
   files, no re-encoding:

       python scripts/preprocess_re10k.py \\
           --from_chunks /path/to/re10k/test \\
           --output_dir  /path/to/RealEstate10K_processed

Use --limit N to convert only the first N scenes (smoke test).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_raw_bytes(path: Path) -> torch.Tensor:
    """Raw encoded file bytes as a 1-D uint8 tensor (no decode/re-encode)."""
    return torch.tensor(np.fromfile(str(path), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Mode 1: official camera .txt files + extracted frames
# ---------------------------------------------------------------------------

def parse_pose_txt(path: Path):
    """Official RE10K camera file -> (timestamps list, cameras (N, 18) float32)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if lines and not lines[0][0].isdigit():
        lines = lines[1:]  # first line is the video URL
    timestamps, rows = [], []
    for ln in lines:
        vals = ln.split()
        if len(vals) != 19:
            raise ValueError(f"{path}: expected 19 columns per line, got {len(vals)}")
        timestamps.append(int(vals[0]))
        # fx fy cx cy (normalized), two zeros, 3x4 w2c row-major — verbatim.
        rows.append([float(v) for v in vals[1:]])
    return timestamps, torch.tensor(rows, dtype=torch.float32)


def find_frames(scene_frames_dir: Path, timestamps):
    """Match frame files to pose timestamps; fall back to sorted order."""
    files = sorted(p for p in scene_frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    by_stem = {}
    for p in files:
        try:
            by_stem[int(p.stem)] = p
        except ValueError:
            pass
    if all(t in by_stem for t in timestamps):
        return [by_stem[t] for t in timestamps]
    if len(files) == len(timestamps):
        return files  # sorted order matches pose order
    matched = [t for t in timestamps if t in by_stem]
    raise ValueError(
        f"{scene_frames_dir}: cannot match {len(timestamps)} poses to "
        f"{len(files)} frames ({len(matched)} timestamp matches)."
    )


def convert_official(poses_dir: Path, frames_dir: Path, output_dir: Path, limit):
    pose_files = sorted(poses_dir.glob("*.txt"))
    if limit:
        pose_files = pose_files[:limit]
    if not pose_files:
        sys.exit(f"No .txt pose files found in {poses_dir}")
    n_ok = 0
    for pose_file in pose_files:
        scene_id = pose_file.stem
        scene_frames = frames_dir / scene_id
        if not scene_frames.is_dir():
            print(f"[skip] {scene_id}: no frame directory {scene_frames}")
            continue
        try:
            timestamps, cameras = parse_pose_txt(pose_file)
            frame_paths = find_frames(scene_frames, timestamps)
        except ValueError as e:
            print(f"[skip] {scene_id}: {e}")
            continue
        images = [load_raw_bytes(p) for p in frame_paths]
        torch.save({"images": images, "cameras": cameras}, output_dir / f"{scene_id}.torch")
        n_ok += 1
        print(f"[ok] {scene_id}: {len(images)} frames")
    print(f"\nConverted {n_ok}/{len(pose_files)} scenes -> {output_dir}")


# ---------------------------------------------------------------------------
# Mode 2: pixelSplat / depthsplat chunks
# ---------------------------------------------------------------------------

def convert_chunks(chunks_dir: Path, output_dir: Path, limit):
    chunk_files = sorted(chunks_dir.glob("*.torch"))
    if not chunk_files:
        sys.exit(f"No .torch chunk files found in {chunks_dir}")
    n_ok = 0
    for chunk_file in chunk_files:
        examples = torch.load(chunk_file, weights_only=True)
        if isinstance(examples, dict):
            examples = [examples]
        for ex in examples:
            if limit and n_ok >= limit:
                print(f"\nConverted {n_ok} scenes (limit) -> {output_dir}")
                return
            scene_id = ex["key"]
            for prefix in ("re10k_", "dl3dv_"):
                if scene_id.startswith(prefix):
                    scene_id = scene_id[len(prefix):]
            cameras = ex["cameras"].to(torch.float32)
            images = [img if isinstance(img, torch.Tensor) else torch.tensor(img)
                      for img in ex["images"]]
            if len(images) != cameras.shape[0]:
                print(f"[skip] {scene_id}: {len(images)} images vs {cameras.shape[0]} cameras")
                continue
            torch.save({"images": images, "cameras": cameras},
                       output_dir / f"{scene_id}.torch")
            n_ok += 1
        print(f"[ok] {chunk_file.name}: total {n_ok} scenes so far")
    print(f"\nConverted {n_ok} scenes -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses_dir", type=Path,
                        help="Directory of official RE10K camera .txt files")
    parser.add_argument("--frames_dir", type=Path,
                        help="Directory of per-scene frame folders (<frames_dir>/<scene_id>/*.jpg)")
    parser.add_argument("--from_chunks", type=Path,
                        help="Directory of pixelSplat/depthsplat .torch chunks (alternative input)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Convert only the first N scenes (smoke test)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.from_chunks:
        convert_chunks(args.from_chunks, args.output_dir, args.limit)
    elif args.poses_dir and args.frames_dir:
        convert_official(args.poses_dir, args.frames_dir, args.output_dir, args.limit)
    else:
        parser.error("Pass either --from_chunks, or both --poses_dir and --frames_dir.")


if __name__ == "__main__":
    main()
