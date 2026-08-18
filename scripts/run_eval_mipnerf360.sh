#!/bin/bash
# ============================================================================
# GenRec — MipNeRF360 evaluation
# ============================================================================
# Usage:
#   DATA_PATH=/path/to/mipnerf360 bash scripts/run_eval_mipnerf360.sh
#
# CHECKPOINT defaults to the auto-downloaded 'genrec' weights; override with a
# local .pth path or URL. FACTOR selects the image downsampling (images_<factor>/).
# All scenes under DATA_PATH are evaluated. Extra flags are forwarded, e.g.:
#   bash scripts/run_eval_mipnerf360.sh --num_steps 30
# ============================================================================
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_FORCE_DISABLE_TRITON=1

CHECKPOINT="${CHECKPOINT:-genrec}"
DATA_PATH="${DATA_PATH:?set DATA_PATH=/path/to/mipnerf360}"
OUTPUT_DIR="${OUTPUT_DIR:-./mipnerf360_eval_results}"
FACTOR="${FACTOR:-4}"

python -m genrec.cli.eval_mipnerf360 \
    --checkpoint "${CHECKPOINT}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --factor "${FACTOR}" \
    "$@"
