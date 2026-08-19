# Vendored: Depth Anything 3

This directory is a vendored copy of
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(© 2025 ByteDance Ltd. and/or its affiliates), licensed under the Apache
License 2.0 (see `LICENSE` in this directory).

Vendored from upstream commit
[`ed6989a23cd389e975ed9f7cbd7385396e6d867e`](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/ed6989a23cd389e975ed9f7cbd7385396e6d867e)
(2025-11-21); GenRec uses it as the geometry backbone
(depth / pose / intrinsics estimation) via `depth_anything_3.api.DepthAnything3`.

Local modifications relative to upstream:

- Removed the Gradio demo app (`app/`), the FastAPI backend + gallery server
  (`services/`), and the CLI (`cli.py`) — none are used by GenRec and they pull
  in extra dependencies (gradio, fastapi).
- `api.py`: accepts explicit render extrinsics/intrinsics — target poses are
  rescaled to the predicted scene scale and intrinsics resized to the processed
  resolution before rendering.
- `utils/visualize.py`: added plotly-based camera-frustum visualization helpers.
- Smaller adjustments in `utils/io/input_processor.py`, `utils/export/gs.py`,
  `utils/export/colmap.py`, `model/da3.py`, `model/gs_adapter.py`,
  `model/utils/gs_renderer.py`, and `model/dinov2/vision_transformer.py`;
  `__init__.py` is GenRec-local.
