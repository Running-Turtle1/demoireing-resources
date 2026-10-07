#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_ROOT:?Set TRAIN_ROOT to the UHDM training directory}"
: "${VAL_ROOT:?Set VAL_ROOT to a held-out validation directory}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mbcnn_uhdm_6gpu}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m torch.distributed.run \
  --standalone --nproc_per_node=6 train.py \
  --config configs/uhdm_ddp_6gpu.yaml \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
