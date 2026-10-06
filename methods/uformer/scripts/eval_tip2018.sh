#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the TIP2018 directory}"
: "${CHECKPOINT:?Set CHECKPOINT to a Uformer-B checkpoint}"
"${PYTHON_BIN:-python}" test/test_tip2018.py --data_root "$DATA_ROOT" --checkpoint "$CHECKPOINT" "${@:1}"
