# 使用 UHDM 权重进行跨数据集测试

本文说明如何直接使用在 UHDM 上训练的 Uformer-B 权重，在 TIP2018、
FHDMi、LCDMoire 或自定义数据上进行测试。这里不进行微调，属于
**跨数据集零样本评测**（zero-shot cross-dataset evaluation）。

## 1. 准备环境和权重

以下命令均从仓库根目录执行：

```bash
python -m pip install -r methods/uformer/requirements.txt
python -m pip install PyYAML

python tools/download_weights.py \
  uformer-b-uhdm-epoch250 \
  --output_dir checkpoints

REPO_ROOT="$PWD"
UHDM_CKPT="$REPO_ROOT/checkpoints/uhdm/model.safetensors"
```

下载工具会自动校验 SHA256。UHDM 权重仍然使用标准 Uformer-B 架构，
因此可以直接传给其他数据集的评测程序。

## 2. 数据目录

准备好目标数据集，并保持仓库所要求的配对目录结构：

- [TIP2018](../../datasets/tip2018/README.md)
- [FHDMi](../../datasets/fhdmi/README.md)
- [LCDMoire](../../datasets/lcdmoire/README.md)

现有评测脚本需要 moiré 输入和对应的无 moiré GT。只有输入图像而没有
GT 时，需要另写 inference-only 数据加载器；此时只能保存恢复结果，不能
计算 PSNR 和 SSIM。

## 3. 运行测试

### UHDM → TIP2018

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_ROOT=/path/to/TIP2018 \
CHECKPOINT="$UHDM_CKPT" \
bash methods/uformer/scripts/eval_tip2018.sh \
  --batch_size 1 \
  --save_dir "$REPO_ROOT/outputs/uhdm_to_tip2018"
```

TIP2018 会使用本仓库记录的中央 `1/6..5/6` 裁剪，并缩放至
256×256。这个协议与部分论文采用的预处理不同。

### UHDM → FHDMi

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
DATA_ROOT=/path/to/FHDMi \
CHECKPOINT="$UHDM_CKPT" \
bash methods/uformer/scripts/eval_fhdmi.sh \
  --tile_size 512 \
  --tile_overlap 128 \
  --save_dir "$REPO_ROOT/outputs/uhdm_to_fhdmi"
```

FHDMi 使用重叠分块恢复完整分辨率图像。公开测试集应包含 2,019 对图像。

### UHDM → LCDMoire

```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
DATA_ROOT=/path/to/LCDMoire \
CHECKPOINT="$UHDM_CKPT" \
bash methods/uformer/scripts/eval_lcdmoire.sh \
  --tile_size 1024 \
  --tile_overlap 0 \
  --save_dir "$REPO_ROOT/outputs/uhdm_to_lcdmoire"
```

上述命令对每张 1024×1024 图像执行一次完整前向传播。如果显存不足，
可以改用 `--tile_size 512 --tile_overlap 128`，但必须注明评测协议已经变化。
LCDMoire 使用包含 GT 的 100 对验证图像，而不是无公开 GT 的 challenge test。

程序结束时会输出类似结果：

```text
checkpoint=... images=... PSNR=... SSIM=...
```

## 4. 测试自定义配对数据

对于新的配对数据集，可以复制最接近的数据加载器和测试程序：

- 256×256 或需要统一缩放的数据：参考 `dataset/dataset_demoire.py` 和
  `test/test_tip2018.py`。
- 高分辨率数据：参考 `dataset/dataset_fhdmi.py` 和
  `test/test_fhdmi.py`。

自定义 `Dataset` 每次应返回：

```python
(target_tensor, source_tensor, sample_name)
```

其中张量应为 RGB、`float32`、`CHW` 排列，数值范围为 `[0, 1]`。
高分辨率图像建议继续使用测试程序中的 overlap-add 分块推理。

## 5. 结果记录建议

建议将实验明确标记为：

```text
Uformer-B, trained on UHDM, zero-shot evaluated on <target dataset>
```

同时记录目标数据集、裁剪或缩放方式、tile 大小、overlap、颜色空间、
测试图像数量以及 checkpoint SHA256。跨数据集结果不应直接冒充在目标
数据集上训练所得的结果，也不应在协议不一致时直接宣称优于论文 SOTA。

## 6. 常见问题

- **显存不足**：减小 `--tile_size`；高分辨率数据建议保留一定 overlap。
- **只想快速检查流程**：TIP2018 使用 `--max_batches 1`，其他测试程序使用
  `--max_images 1`。
- **关闭混合精度**：添加 `--no_amp`，但推理会更慢、显存占用通常更高。
- **保存结果图**：为 `--save_dir` 指定绝对路径，避免脚本切换工作目录后
  输出到意外位置。
