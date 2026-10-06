#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the UHDM directory}"
python_bin="${PYTHON_BIN:-python}"
nproc="${NPROC_PER_NODE:-1}"
output_dir="${OUTPUT_DIR:-./outputs/uhdm/uformer-b}"
args=(--data_root "$DATA_ROOT" --output_dir "$output_dir" --epochs 250 --crop_size 768 --batch_size 1 --grad_accum_steps 2 --num_workers 4 --lr 2e-4 --min_lr 1e-6 --weight_decay 0.02 --eval_every 10 --checkpoint_every 10 --tile_size 768 --tile_overlap 128 --seed 123)
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$nproc" train/train_demoire_uhdm.py "${args[@]}"
