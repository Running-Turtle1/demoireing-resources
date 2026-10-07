from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from mbcnn import MBCNN, load_model_checkpoint
from mbcnn.data import PairedImageDataset
from mbcnn.distributed import DistributedContext, DistributedEvalSampler, gather_objects
from mbcnn.evaluation import pad_esdnet_style, remove_padding, save_rgb_tensor
from mbcnn.metrics import batch_uhdm_psnr, batch_uhdm_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MBCNN on full-resolution paired images"
    )
    parser.add_argument("--config", default="configs/eval_uhdm_full.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = DistributedContext.initialize(args.device)
    try:
        run(args, context)
    finally:
        context.close()


def run(args: argparse.Namespace, context: DistributedContext) -> None:
    config = load_config(args.config)
    apply_overrides(config, args)
    validate_config(config, context.device)

    output_dir = Path(config["output_dir"]).expanduser().resolve()
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as output:
            yaml.safe_dump(config, output, sort_keys=False)
    context.barrier()

    dataset = PairedImageDataset(
        config["data"]["root"],
        input_suffix=str(config["data"]["input_suffix"]),
        target_suffix=str(config["data"]["target_suffix"]),
        crop_size=None,
        random_crop=False,
        augment=False,
    )
    sampler = DistributedEvalSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        max_samples=optional_int(config["evaluation"]["max_samples"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(
            config["data"]["pin_memory"] and context.device.type == "cuda"
        ),
        persistent_workers=False,
    )

    checkpoint_path = Path(config["checkpoint"]).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    model = MBCNN().to(context.device).eval()
    load_model_checkpoint(model, checkpoint)
    lpips_model = make_lpips_model(
        enabled=bool(config["evaluation"]["compute_lpips"]),
        device=context.device,
    )

    if context.is_main:
        print(
            f"device={context.device} world_size={context.world_size} "
            f"pairs={sampler.sample_count} precision={config['evaluation']['precision']} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

    local_records: list[dict[str, Any]] = []
    progress = tqdm(
        loader,
        desc=f"full-resolution rank {context.rank}",
        disable=not context.is_main,
        dynamic_ncols=True,
    )
    for batch in progress:
        record = evaluate_batch(
            batch,
            model=model,
            lpips_model=lpips_model,
            device=context.device,
            precision=str(config["evaluation"]["precision"]),
            padding_multiple=int(config["evaluation"]["padding_multiple"]),
            padding_values=tuple(
                float(value) for value in config["evaluation"]["padding_values"]
            ),
            image_dir=output_dir / "images"
            if config["evaluation"]["save_images"]
            else None,
        )
        local_records.append(record)
        if context.is_main:
            progress.set_postfix(
                psnr=f"{record['psnr']:.3f}",
                memory=f"{record['peak_memory_mb']:.0f}MB",
            )

    gathered_records = gather_objects(local_records, context.world_size)
    if context.is_main:
        records = sorted(
            (record for rank_records in gathered_records for record in rank_records),
            key=lambda record: record["id"],
        )
        summary = make_summary(
            records,
            config=config,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            world_size=context.world_size,
        )
        write_results(output_dir, records, summary)
        print(
            f"PSNR={summary['metrics']['psnr']:.4f} "
            f"SSIM={summary['metrics']['ssim']:.6f} "
            f"images={summary['images']} output={output_dir}",
            flush=True,
        )
    context.barrier()


def evaluate_batch(
    batch: dict[str, Any],
    *,
    model: MBCNN,
    lpips_model: torch.nn.Module | None,
    device: torch.device,
    precision: str,
    padding_multiple: int,
    padding_values: tuple[float, float, float],
    image_dir: Path | None,
) -> dict[str, Any]:
    inputs = _batch_tensor(batch, "input").to(device, non_blocking=True)
    target_cpu = _batch_tensor(batch, "target")
    sample_id = str(batch["id"][0])
    height, width = inputs.shape[-2:]
    padded, padding = pad_esdnet_style(
        inputs,
        multiple=padding_multiple,
        channel_values=padding_values,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision == "fp16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        outputs = model(padded)
        restored = remove_padding(outputs[-1], padding)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start
    del outputs, padded, inputs

    restored = restored.float().clamp(0.0, 1.0)
    target = target_cpu.to(device, non_blocking=True).float().clamp(0.0, 1.0)
    psnr = float(batch_uhdm_psnr(restored, target).item())
    ssim = float(batch_uhdm_ssim(restored, target).item())
    lpips_value = None
    if lpips_model is not None:
        with torch.inference_mode():
            lpips_value = float(
                lpips_model(restored, target, normalize=True).flatten()[0].item()
            )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    if image_dir is not None:
        save_rgb_tensor(restored, image_dir / f"{sample_id}.png")

    return {
        "id": sample_id,
        "height": int(height),
        "width": int(width),
        "pad_left": padding.left,
        "pad_right": padding.right,
        "pad_top": padding.top,
        "pad_bottom": padding.bottom,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "inference_seconds": inference_seconds,
        "peak_memory_mb": peak_memory_mb,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise TypeError("Evaluation config must be a mapping")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        (None, "checkpoint"): args.checkpoint,
        (None, "output_dir"): args.output_dir,
        ("data", "root"): args.data_root,
        ("data", "num_workers"): args.num_workers,
        ("evaluation", "precision"): args.precision,
        ("evaluation", "max_samples"): args.max_samples,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            if section is None:
                config[key] = value
            else:
                config[section][key] = value
    if args.lpips:
        config["evaluation"]["compute_lpips"] = True
    if args.no_save_images:
        config["evaluation"]["save_images"] = False


def validate_config(config: dict[str, Any], device: torch.device) -> None:
    for key in ("checkpoint", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Set {key} in the config or pass --{key.replace('_', '-')}")
    if not Path(config["checkpoint"]).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {config['checkpoint']}")
    if not config["data"].get("root"):
        raise ValueError("Set data.root in the config or pass --data-root")
    if int(config["data"]["num_workers"]) < 0:
        raise ValueError("data.num_workers must be non-negative")
    if int(config["evaluation"]["padding_multiple"]) <= 0:
        raise ValueError("evaluation.padding_multiple must be positive")
    if len(config["evaluation"]["padding_values"]) != 3:
        raise ValueError("evaluation.padding_values must have three entries")
    if config["evaluation"]["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("FP16 evaluation requires CUDA")
    max_samples = optional_int(config["evaluation"]["max_samples"])
    if max_samples is not None and max_samples <= 0:
        raise ValueError("evaluation.max_samples must be positive or null")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Loading safetensors requires: python -m pip install safetensors"
            ) from error
        return dict(load_file(path, device="cpu"))
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a mapping")
    return checkpoint


def make_lpips_model(
    *, enabled: bool, device: torch.device
) -> torch.nn.Module | None:
    if not enabled:
        return None
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS evaluation requires: python -m pip install -e '.[lpips]'"
        ) from error
    return lpips.LPIPS(net="alex").to(device).eval()


def make_summary(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    world_size: int,
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("No evaluation records were produced")
    lpips_values = [record["lpips"] for record in records if record["lpips"] is not None]
    epoch = checkpoint.get("epoch")
    return {
        "protocol": "UniDemoire ESDNet UHDM full-resolution",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(epoch) + 1 if epoch is not None else None,
        "checkpoint_validation_best_psnr": checkpoint.get("best_psnr"),
        "images": len(records),
        "world_size": world_size,
        "precision": config["evaluation"]["precision"],
        "metrics": {
            "psnr": fmean(record["psnr"] for record in records),
            "ssim": fmean(record["ssim"] for record in records),
            "lpips": fmean(lpips_values) if lpips_values else None,
        },
        "performance": {
            "mean_inference_seconds": fmean(
                record["inference_seconds"] for record in records
            ),
            "max_peak_memory_mb": max(record["peak_memory_mb"] for record in records),
        },
        "padding": {
            "multiple": config["evaluation"]["padding_multiple"],
            "rgb_values": config["evaluation"]["padding_values"],
        },
    }


def write_results(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")


def optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _batch_tensor(batch: dict[str, Any], key: str) -> Tensor:
    value = batch[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"Batch field {key!r} must be a tensor")
    return value


if __name__ == "__main__":
    main()
