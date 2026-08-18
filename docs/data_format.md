# Dataset file layout

You normally never need this — `scripts/preprocess_re10k.py` and
`scripts/preprocess_dl3dv.py` produce these files for you (see the
[Data section of the README](../README.md#data)). This page documents the exact
on-disk schema for anyone writing their own converter or loading a new dataset.

Each scene is stored in its own file. RealEstate10K uses one PyTorch
`.torch`/`.pt` file per scene; DL3DV-10K uses one HDF5 `.h5` file per scene. The
two formats carry the **same logical contents** — only the storage backend
differs.

## RealEstate10K — `<scene_id>.torch` (a `torch.load`-able `dict`)

```
<scene_id>.torch                      # dict, one scene
├── "images"   list[bytes-array]      # length N; each entry is a 1-D uint8
│                                     #   array of JPEG/PNG-encoded bytes,
│                                     #   decoded on demand
└── "cameras"  Tensor (N, 18) float32 # one row per frame:
                                      #   [0:4]  normalized intrinsics [fx, fy, cx, cy]
                                      #   [4:6]  unused (zeros)
                                      #   [6:18] 3×4 world-to-camera (w2c), row-major
```

## DL3DV-10K — `<scene_id>.h5`

```
<scene_id>.h5                          # HDF5 file, one scene
├── cameras    dataset (N, 18) float32 # identical layout to RE10K "cameras":
│                                      #   [0:4] norm. intrinsics, [4:6] zeros,
│                                      #   [6:18] 3×4 w2c
└── images/    group                   # JPEG-encoded frames, keyed by frame index
    ├── "0"    dataset (uint8,)        #   1-D byte array, decoded on demand
    ├── "1"    dataset (uint8,)
    └── …      "N-1"
```

## Notes (both formats)

- `N` is the number of frames in the scene; the loader samples `T_in + T_out`
  of them per item.
- Intrinsics are **normalized** by image width/height — multiply `fx, cx` by `W`
  and `fy, cy` by `H` to get pixel-space values.
- Extrinsics are stored as a 3×4 **world-to-camera** matrix; the loader appends
  `[0, 0, 0, 1]` to form a 4×4 `w2c` and inverts it for `c2w`.
- Cameras use the OpenCV convention (+Z forward, −Y up). For DL3DV the world
  frame keeps its native +Z-up orientation.
- Image frames are stored **encoded** (JPEG/PNG bytes), not as raw pixel arrays,
  and decoded lazily for only the sampled frames.
- Scene-list files (e.g. `assets/re10k_test_scenes_100.txt`) contain one scene
  id per line; the loader appends the storage extension (`.torch`/`.pt`/`.h5`)
  automatically.

Mip-NeRF 360 needs no conversion: the loader reads the standard COLMAP layout
(`<scene>/sparse/0` + `images_<factor>/`) directly.
