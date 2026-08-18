#!/bin/bash
# ============================================================================
# GenRec — DL3DV-10K evaluation
# ============================================================================
# Usage:
#   DATA_PATH=/path/to/DL3DV-10K bash scripts/run_eval_dl3dv.sh
#
# CHECKPOINT defaults to the auto-downloaded 'genrec' weights; override with a
# local .pth path or URL. The DL3DV evaluation defaults (longest_side, step_size,
# num_steps, guidance, ...) live in genrec/cli/eval_dl3dv.py and can be overridden by
# forwarding flags, e.g.:  bash scripts/run_eval_dl3dv.sh --step_size 2
# ============================================================================
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_FORCE_DISABLE_TRITON=1

CHECKPOINT="${CHECKPOINT:-genrec}"
DATA_PATH="${DATA_PATH:?set DATA_PATH=/path/to/DL3DV-10K}"
OUTPUT_DIR="${OUTPUT_DIR:-./dl3dv_eval_results}"

python -m genrec.cli.eval_dl3dv \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
