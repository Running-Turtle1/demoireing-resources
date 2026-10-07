# MBCNN for Image Demoiréing

This resource provides a standalone PyTorch extraction of MBCNN and a
six-GPU reproduction on UHDM. It contains only the MBCNN model and paired-image
training/evaluation pipeline; it does not use the UniDemoire model or synthetic
moire generation.

## UHDM results

| Checkpoint | Selection | Images | PSNR | SSIM |
|---|---|---:|---:|---:|
| Epoch 110 best | Highest 100-center-crop validation PSNR | 500 | **20.223574** | **0.771880** |
| Epoch 150 last | Fixed final training epoch | 500 | 20.002245 | 0.768795 |

Both rows use direct FP32 inference on all 500 full-resolution UHDM test
images. The epoch-110 checkpoint was selected using center crops from the
official test set during training, so it must be labeled **test-selected** and
should not be presented as a clean held-out estimate. The epoch-150 checkpoint
is included as the fixed-schedule result. LPIPS was not computed.

Machine-readable summaries and all per-image PSNR/SSIM values are under
[`results/`](results/).

## Evaluation protocol

- Input: paired UHDM `*_moire.jpg` and `*_gt.jpg` files, RGB `[0, 1]`.
- Geometry: complete 3840x2160 image, batch size 1, no tiling or crop.
- Padding: symmetric to a multiple of 32 using RGB values
  `(0.3827, 0.4141, 0.3912)`, then removed before metrics.
- PSNR: full RGB floating-point MSE after clamping prediction and target.
- SSIM: 11x11 Gaussian window, sigma 1.5, valid convolution, full RGB.
- Aggregation: unweighted arithmetic mean of per-image metrics.

This matches the UHDM PSNR/SSIM path and padding convention in the UniDemoire
ESDNet evaluator. Lossless PNG outputs were inspected locally but are not
committed because of their size.

## Installation

From this directory:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Download either published checkpoint from the repository root:

```bash
python tools/download_weights.py mbcnn-uhdm-best-epoch110 --output_dir checkpoints
python tools/download_weights.py mbcnn-uhdm-last-epoch150 --output_dir checkpoints
```

## Full-resolution evaluation

From `methods/mbcnn`, evaluate on six GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=6 evaluate.py \
  --config configs/eval_uhdm_full.yaml \
  --checkpoint ../../checkpoints/uhdm/best_epoch110/model.safetensors \
  --data-root /path/to/UHDM/test \
  --output-dir outputs/eval_full_best
```

Replace `best_epoch110` with `last_epoch150` for the final-epoch checkpoint.
Run with `--max-samples 1` on one GPU before a complete evaluation. The
evaluator writes `summary.json` and `metrics.csv`. Set `save_images: true` in
the evaluation config only when restored PNGs are needed.

## Six-GPU training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=6 train.py \
  --config configs/uhdm_ddp_6gpu.yaml \
  --train-root /path/to/UHDM/train \
  --val-root /path/to/validation \
  --output-dir outputs/mbcnn_uhdm_6gpu
```

The reproduction used 150 epochs, Adam at `1e-4`, AMP, random 384x384 crops,
batch size 1 per GPU, no scheduler, and validation every 10 epochs. Use a
held-out validation directory for future clean checkpoint selection.

## Provenance

The PyTorch implementation was extracted from
[4DVLab/UniDemoire](https://github.com/4DVLab/UniDemoire) commit
`4221a0e98f0078c3f72ef1da8128642e7f02662a`. The architecture originates from
[Image Demoireing with Learnable Bandpass Filters (CVPR 2020)](https://arxiv.org/abs/2004.00406).
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSE`](LICENSE).
