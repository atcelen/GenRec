# Vendored: Depth Anything 3

This directory is a vendored copy of
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(© 2025 ByteDance Ltd. and/or its affiliates), licensed under the Apache
License 2.0 (see `LICENSE` in this directory).

<!-- TODO(release): pin the exact upstream commit SHA the copy was taken from. -->
Vendored from upstream in mid-2026; GenRec uses it as the geometry backbone
(depth / pose / intrinsics estimation) via `depth_anything_3.api.DepthAnything3`.

Local modifications relative to upstream:

- Removed the Gradio demo app (`app/`), the FastAPI backend + gallery server
  (`services/`), and the CLI (`cli.py`) — none are used by GenRec and they pull
  in extra dependencies (gradio, fastapi).
- No functional changes to the retained model / utility code.
