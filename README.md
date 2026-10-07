# Demoireing Resources

An extensible collection of reproducible image demoiréing baselines, dataset
protocols, pretrained weights, and benchmark results.

## Available methods

| Method | Architecture | Datasets | Code | Weights |
|---|---|---|---|---|
| Uformer-B | U-shaped Transformer, 50.88M parameters | TIP2018, UHDM, FHDMi, LCDMoire | [methods/uformer](methods/uformer) | [Hugging Face](https://huggingface.co/running-Turtle/uformer-b-demoireing) |
| MBCNN | Multi-scale bandpass CNN, 14.19M parameters | UHDM | [methods/mbcnn](methods/mbcnn) | [Hugging Face](https://huggingface.co/running-Turtle/mbcnn-uhdm-demoireing) |

Each method is self-contained and carries its own dependencies, upstream
license, configurations, tests, and detailed results. Machine-readable entries
are stored in [`registry/`](registry/).

## Reproduction results

The released checkpoints produce the following results under their documented
method-specific protocols:

| Method | Dataset | Checkpoint | PSNR | SSIM |
|---|---|---:|---:|---:|
| Uformer-B | TIP2018 | Epoch 250 | 31.009843 | 0.897576 |
| Uformer-B | UHDM | Epoch 250 | 20.764296 | 0.773419 |
| Uformer-B | FHDMi | Epoch 150 | 23.486293 | 0.809517 |
| Uformer-B | LCDMoire | Epoch 220 | 41.902345 | 0.986314 |
| MBCNN | UHDM | Epoch 110 best | 20.223574 | 0.771880 |
| MBCNN | UHDM | Epoch 150 last | 20.002245 | 0.768795 |

For reference, the Uformer supplementary material reports `29.28 / 0.917` on
TIP18. Our TIP2018 result uses a different documented crop and preprocessing
protocol, so it should not be treated as a strict like-for-like comparison.

The MBCNN rows use direct FP32 inference on all 500 full-resolution UHDM test
images with the UniDemoire ESDNet padding and PSNR/SSIM protocol. Epoch 110 is
explicitly test-selected using 100 center crops; epoch 150 is the fixed final
epoch. See the detailed [Uformer](methods/uformer#results) and
[MBCNN](methods/mbcnn#uhdm-results) protocol notes before comparing methods.

## Repository layout

```text
datasets/    Dataset cards and expected directory layouts
methods/     Self-contained method reproductions
registry/    Method, dataset, and checkpoint metadata
tools/       Lightweight repository maintenance utilities
docs/        Contribution and evaluation conventions
```

## Adding a resource

See [docs/adding_a_method.md](docs/adding_a_method.md). New methods must declare
their upstream source and license, avoid machine-specific paths, provide a
smoke test, and report the complete evaluation protocol with every metric.

## Data and weights

Datasets are not redistributed. Follow the dataset cards and obtain data from
the corresponding owners. Large model weights are hosted in method-specific
Hugging Face repositories and indexed by `registry/checkpoints.yaml` with
SHA256 hashes.

For a worked example using the UHDM-trained Uformer-B checkpoint on another
dataset, see the [cross-dataset evaluation guide](methods/uformer/CROSS_DATASET_EVALUATION.zh-CN.md).

## License

The repository-level utilities and metadata are MIT licensed. Vendored or
adapted methods retain their own licenses under their method directories.
