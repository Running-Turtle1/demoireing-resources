# Demoireing Resources

An extensible collection of reproducible image demoiréing baselines, dataset
protocols, pretrained weights, and benchmark results.

## Available methods

| Method | Architecture | Datasets | Code | Weights |
|---|---|---|---|---|
| Uformer-B | U-shaped Transformer, 50.88M parameters | TIP2018, UHDM, FHDMi, LCDMoire | [methods/uformer](methods/uformer) | [Hugging Face](https://huggingface.co/running-Turtle/uformer-b-demoireing) |

Each method is self-contained and carries its own dependencies, upstream
license, configurations, tests, and detailed results. Machine-readable entries
are stored in [`registry/`](registry/).

## Reproduction results

The released Uformer-B checkpoints produce the following results under the
protocols documented in [`methods/uformer`](methods/uformer):

| Dataset | Checkpoint | PSNR | SSIM |
|---|---:|---:|---:|
| TIP2018 | Epoch 250 | 31.009843 | 0.897576 |
| UHDM | Epoch 250 | 20.764296 | 0.773419 |
| FHDMi | Epoch 150 | 23.486293 | 0.809517 |
| LCDMoire | Epoch 220 | 41.902345 | 0.986314 |

For reference, the Uformer supplementary material reports `29.28 / 0.917` on
TIP18. Our TIP2018 result uses a different documented crop and preprocessing
protocol, so it should not be treated as a strict like-for-like comparison.
See the [full results and protocol notes](methods/uformer#results).

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
the corresponding owners. Large model weights are hosted separately and
indexed by `registry/checkpoints.yaml` with SHA256 hashes.

## License

The repository-level utilities and metadata are MIT licensed. Vendored or
adapted methods retain their own licenses under their method directories.
