# Uformer-B for Image Demoiréing

This resource reproduces Uformer-B on TIP2018 and extends the same architecture
to UHDM, FHDMi, and LCDMoire. Uformer-B uses base width 32, encoder depths
`[1, 2, 8, 8]`, mirrored decoder depths, 8×8 attention windows, and about
50.88M parameters.

## Results

| Dataset | Selection | PSNR | SSIM | Protocol |
|---|---:|---:|---:|---|
| TIP2018 | Epoch 250 | 31.009843 | 0.897576 | UHDM-compatible central crop, 256×256 RGB |
| UHDM | Epoch 250 | 20.764296 | 0.773419 | 500 full-resolution images, rectangular full-image RGB |
| FHDMi | Epoch 150 | 23.486293 | 0.809517 | Full-resolution overlap-add RGB |
| LCDMoire | Epoch 220 | 41.902345 | 0.986314 | 100 paired val images, full 1024×1024 RGB uint8 |

The Uformer supplementary material reports `29.28 / 0.917` on TIP18 using a
central `[0.15, 0.85]` crop. The released TIP2018 run uses a different,
explicitly documented preprocessing protocol and is therefore not presented as
a strict numerical reproduction of that row.

Paper: [Uformer (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.html) ·
[supplementary material](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Wang_Uformer_A_General_CVPR_2022_supplemental.pdf)

## Quick start

```bash
python -m pip install -r requirements.txt
DATA_ROOT=/path/to/TIP2018 bash scripts/train_tip2018.sh
python test/test_tip2018.py \
  --data_root /path/to/TIP2018 \
  --checkpoint /path/to/model.pth
```

See `configs/` for all training protocols and `scripts/` for portable launchers.
Set `PYTHON_BIN`, `NPROC_PER_NODE`, `OUTPUT_DIR`, `RESUME`, and
`CUDA_VISIBLE_DEVICES` as needed.

Validate the architecture and an optional downloaded checkpoint with:

```bash
python tests/smoke_model.py --checkpoint /path/to/model.safetensors
```

## Attribution

This method retains the upstream MIT license. See `THIRD_PARTY_NOTICES.md` and
cite the original CVPR 2022 Uformer paper.
