#!/bin/bash
# ============================================================================
# GenRec — general inference: input image(s) + camera poses -> novel views
# ============================================================================
# Drop 1+ images in a folder and synthesize novel views:
#   - default          -> a generated camera trajectory (single image: --trajectory;
#                         2+ images: SLERP interpolation between the input viewpoints)
#   - POSES=poses.json -> render exactly the camera poses in that file (the same
#                         schema this tool emits; T_out = number of target poses)
#
# Usage:
#   INPUT_DIR=./my_photos bash scripts/run_inference.sh
#   INPUT_DIR=./my_photos POSES=./custom_out/poses.json bash scripts/run_inference.sh
#
# CHECKPOINT defaults to the auto-downloaded 'genrec' weights; override with a
# registry name, an http(s) URL, or a local .pth path. Extra args are forwarded.
# ============================================================================
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XFORMERS_FORCE_DISABLE_TRITON=1

CONFIG="${CONFIG:-configs/dl3dv.yaml}"
CHECKPOINT="${CHECKPOINT:-genrec}"
INPUT_DIR="${INPUT_DIR:?set INPUT_DIR=/path/to/your/photos}"
OUTPUT_DIR="${OUTPUT_DIR:-./custom_out}"
TOTAL_VIEWS="${TOTAL_VIEWS:-8}"
TRAJECTORY="${TRAJECTORY:-orbit}"   # orbit | dolly | spiral | wiggle (single-image only)
MOTION_SCALE="${MOTION_SCALE:-0.3}"
NUM_STEPS="${NUM_STEPS:-25}"
GUIDANCE="${GUIDANCE:-1.0}"
SEED="${SEED:-42}"

# Optional: render user-specified poses instead of a generated trajectory.
POSES_ARG=()
if [ -n "${POSES:-}" ]; then
    POSES_ARG=(--poses "${POSES}")
fi

python -m genrec.cli.inference \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --total_views "${TOTAL_VIEWS}" \
    --trajectory "${TRAJECTORY}" \
    --motion_scale "${MOTION_SCALE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --guidance "${GUIDANCE}" \
    --seed "${SEED}" \
    "${POSES_ARG[@]}" \
    "$@"
