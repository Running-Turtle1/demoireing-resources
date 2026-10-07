#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to an MBCNN checkpoint path}"
: "${DATA_ROOT:?Set DATA_ROOT to the UHDM test directory}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/mbcnn_uhdm_full}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m torch.distributed.run \
  --standalone --nproc_per_node=6 evaluate.py \
  --config configs/eval_uhdm_full.yaml \
  --checkpoint "${CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
