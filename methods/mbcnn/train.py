from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mbcnn import MBCNN, MBCNNLoss
from mbcnn.data import PairedImageDataset
from mbcnn.distributed import (
    DistributedContext,
    DistributedEvalSampler,
    gather_objects,
)
from mbcnn.training import (
    TrainingState,
    append_jsonl,
    capture_rng_state,
    restore_training_checkpoint,
    save_training_checkpoint,
    train_one_epoch,
    unwrap_model,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MBCNN on paired images")
    parser.add_argument("--config", default="configs/uhdm.yaml")
    parser.add_argument("--train-root")
    parser.add_argument("--val-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Checkpoint path, or 'auto' for last.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--val-crop-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--validate-every-epochs", type=int)
    parser.add_argument("--no-amp", action="store_true")
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
    validate_config(config)

    seed = int(config["seed"])
    seed_everything(seed + context.rank)
    device = context.device
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    output_dir = Path(config["logging"]["output_dir"]).expanduser().resolve()
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as output:
            yaml.safe_dump(config, output, sort_keys=False)
    context.barrier()

    train_dataset = PairedImageDataset(
        config["data"]["train_root"],
        input_suffix=config["data"]["input_suffix"],
        target_suffix=config["data"]["target_suffix"],
        crop_size=int(config["data"]["crop_size"]),
        random_crop=True,
        augment=bool(config["data"]["augment"]),
    )
    val_dataset = PairedImageDataset(
        config["data"]["val_root"],
        input_suffix=config["data"]["input_suffix"],
        target_suffix=config["data"]["target_suffix"],
        crop_size=int(config["data"]["val_crop_size"]),
        random_crop=False,
        augment=False,
    )
    train_generator = torch.Generator().manual_seed(seed + context.rank * 1000)
    val_generator = torch.Generator().manual_seed(seed + context.rank * 1000 + 1)
    train_sampler = None
    val_sampler = None
    if context.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        val_sampler = DistributedEvalSampler(
            val_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            max_samples=_optional_int(config["validation"]["max_samples"]),
        )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"] and device.type == "cuda"),
        persistent_workers=bool(config["data"]["persistent_workers"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=train_generator,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=int(config["data"]["val_batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=bool(config["data"]["pin_memory"] and device.type == "cuda"),
        persistent_workers=bool(config["data"]["persistent_workers"]),
        shuffle=False,
        sampler=val_sampler,
        generator=val_generator,
    )

    model = MBCNN().to(device)
    if context.enabled:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if device.type == "cuda" else None,
            output_device=context.local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    criterion = MBCNNLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=tuple(float(value) for value in config["training"]["betas"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, config["training"]["scheduler"])
    scaler = make_grad_scaler(amp)
    state = TrainingState()

    resume_path = resolve_resume_path(args.resume, output_dir)
    if resume_path is not None:
        state = restore_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            train_generator=train_generator,
            val_generator=val_generator,
            device=device,
            rank=context.rank,
        )
        if context.is_main:
            print(f"Resumed from {resume_path} at epoch {state.epoch}")

    if context.is_main:
        global_batch_size = (
            int(config["data"]["batch_size"])
            * context.world_size
            * int(config["training"]["accumulation_steps"])
        )
        print(
            f"device={device} world_size={context.world_size} amp={amp} "
            f"train_pairs={len(train_dataset)} val_pairs={len(val_dataset)} "
            f"global_batch={global_batch_size} output={output_dir}",
            flush=True,
        )
    total_epochs = int(config["training"]["epochs"])
    if state.epoch >= total_epochs:
        raise ValueError(
            f"Checkpoint is already at epoch {state.epoch}; configured epochs={total_epochs}"
        )

    for epoch in range(state.epoch, total_epochs):
        state.epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            state,
            accumulation_steps=int(config["training"]["accumulation_steps"]),
            amp=amp,
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
            log_every_batches=int(config["logging"]["log_every_batches"]),
            max_batches=_optional_int(config["training"]["max_train_batches"]),
            is_main=context.is_main,
        )

        val_metrics = None
        should_validate = (epoch + 1) % int(config["validation"]["every_epochs"]) == 0
        if should_validate:
            preview_path = None
            if context.is_main and config["validation"]["save_preview"]:
                preview_path = output_dir / "previews" / f"epoch_{epoch + 1:04d}.png"
            val_metrics = validate(
                unwrap_model(model),
                criterion,
                val_loader,
                device,
                amp=amp,
                max_samples=None
                if context.enabled
                else _optional_int(config["validation"]["max_samples"]),
                preview_path=preview_path,
                is_main=context.is_main,
            )
            if scheduler is not None:
                scheduler.step(val_metrics["psnr"])

        is_best = val_metrics is not None and val_metrics["psnr"] > state.best_psnr
        if is_best:
            state.best_psnr = float(val_metrics["psnr"])

        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": state.global_step,
            "train": train_metrics,
        }
        if val_metrics is not None:
            record["validation"] = val_metrics
        if context.is_main:
            append_jsonl(output_dir / "metrics.jsonl", record)

        local_rng_state = capture_rng_state(train_generator, val_generator)
        rng_states_by_rank = gather_objects(local_rng_state, context.world_size)
        checkpoint_args = {
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "scheduler": scheduler,
            "state": state,
            "config": config,
            "train_generator": train_generator,
            "val_generator": val_generator,
            "rng_states_by_rank": rng_states_by_rank,
        }
        if context.is_main:
            save_training_checkpoint(output_dir / "last.pth", **checkpoint_args)
            if is_best:
                save_training_checkpoint(
                    output_dir / "best_psnr.pth", **checkpoint_args
                )
        context.barrier()

        summary = (
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"lr={train_metrics['learning_rate']:.3e}"
        )
        if val_metrics is not None:
            summary += (
                f" val_psnr={val_metrics['psnr']:.4f} "
                f"val_ssim={val_metrics['ssim']:.6f} best={state.best_psnr:.4f}"
            )
        if context.is_main:
            print(summary, flush=True)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise TypeError("Training config must be a mapping")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    overrides = {
        ("data", "train_root"): args.train_root,
        ("data", "val_root"): args.val_root,
        ("logging", "output_dir"): args.output_dir,
        ("training", "epochs"): args.epochs,
        ("data", "batch_size"): args.batch_size,
        ("data", "num_workers"): args.num_workers,
        ("data", "crop_size"): args.crop_size,
        ("data", "val_crop_size"): args.val_crop_size,
        ("training", "accumulation_steps"): args.accumulation_steps,
        ("training", "max_train_batches"): args.max_train_batches,
        ("validation", "max_samples"): args.max_val_samples,
        ("validation", "every_epochs"): args.validate_every_epochs,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    if args.no_amp:
        config["training"]["amp"] = False


def validate_config(config: dict[str, Any]) -> None:
    for key in ("train_root", "val_root"):
        if not config["data"].get(key):
            raise ValueError(
                f"Set data.{key} in the config or pass --{key.replace('_', '-')}"
            )
    for key in ("crop_size", "val_crop_size"):
        value = int(config["data"][key])
        if value <= 0 or value % 8:
            raise ValueError(f"data.{key} must be positive and divisible by 8")
    for section, key in (
        ("data", "batch_size"),
        ("data", "val_batch_size"),
        ("training", "epochs"),
        ("training", "accumulation_steps"),
        ("validation", "every_epochs"),
    ):
        if int(config[section][key]) <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    scheduler_name = str(config["training"]["scheduler"]["name"]).lower()
    if scheduler_name not in {"none", "plateau"}:
        raise ValueError("training.scheduler.name must be 'none' or 'plateau'")


def make_loader(
    dataset: PairedImageDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    shuffle: bool,
    sampler: torch.utils.data.Sampler[int] | None,
    generator: torch.Generator,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> Any:
    if str(config["name"]).lower() == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["factor"]),
        patience=int(config["patience"]),
        threshold=float(config["threshold"]),
        min_lr=float(config["min_lr"]),
    )


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def resolve_resume_path(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    path = output_dir / "last.pth" if value == "auto" else Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path.resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


if __name__ == "__main__":
    main()

