#!/bin/bash
# ============================================================================
# GenRec — RealEstate10K evaluation
# ============================================================================
# Usage:
#   DATA_PATH=/path/to/RealEstate10K bash scripts/run_eval_re10k.sh
#
# CHECKPOINT defaults to the auto-downloaded 'genrec' weights; override with a
# local .pth path or URL. The RE10K evaluation defaults (step_size, num_steps,
# guidance, ...) live in genrec/cli/eval_re10k.py and can be overridden by forwarding
# flags, e.g.:  bash scripts/run_eval_re10k.sh --num_steps 30
# ============================================================================
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_FORCE_DISABLE_TRITON=1

CHECKPOINT="${CHECKPOINT:-genrec}"
DATA_PATH="${DATA_PATH:?set DATA_PATH=/path/to/RealEstate10K}"
OUTPUT_DIR="${OUTPUT_DIR:-./re10k_eval_results}"

python -m genrec.cli.eval_re10k \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
