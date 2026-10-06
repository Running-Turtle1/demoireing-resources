#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATA_ROOT:?Set DATA_ROOT to the LCDMoire directory}"
python_bin="${PYTHON_BIN:-python}"
nproc="${NPROC_PER_NODE:-1}"
output_dir="${OUTPUT_DIR:-./outputs/lcdmoire/uformer-b}"
epochs="${EPOCHS:-150}"
lr="${LR:-2e-4}"
args=(--data_root "$DATA_ROOT" --output_dir "$output_dir" --epochs "$epochs" --crop_size 512 --batch_size 1 --grad_accum_steps 2 --num_workers 4 --lr "$lr" --min_lr 1e-6 --weight_decay 0.02 --eval_every 10 --checkpoint_every 10 --tile_size 1024 --tile_overlap 0 --seed 123)
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ "${RESET_SCHEDULER_ON_RESUME:-0}" == "1" ]]; then args+=(--reset_scheduler_on_resume); fi
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$nproc" train/train_demoire_lcdmoire.py "${args[@]}"
