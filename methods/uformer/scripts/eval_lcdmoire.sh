#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the LCDMoire directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a Uformer-B checkpoint}"
nproc="${NPROC_PER_NODE:-1}"
"${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nproc_per_node="$nproc" test/test_lcdmoire.py --data_root "$DATA_ROOT" --checkpoint "$CHECKPOINT" "${@:1}"
