#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the UHDM directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a Uformer-B checkpoint}"
output_dir="${OUTPUT_DIR:-./outputs/uhdm/eval}"
nproc="${NPROC_PER_NODE:-1}"
"${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nproc_per_node="$nproc" test/test_uhdm_rect.py --data_root "$DATA_ROOT" --checkpoint "$CHECKPOINT" --output_dir "$output_dir" --modes full --num_images 500 "${@:1}"
