from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .distributed import reduce_max, reduce_sums
from .metrics import batch_psnr, batch_ssim


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    best_psnr: float = float("-inf")


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    state: TrainingState,
    *,
    accumulation_steps: int,
    amp: bool,
    grad_clip_norm: float,
    log_every_batches: int,
    max_batches: int | None,
    is_main: bool = True,
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    if total_batches <= 0:
        raise ValueError("Training loader produced no batches")

    loss_sum = 0.0
    optimizer_steps = 0
    start_time = time.perf_counter()
    progress = tqdm(
        loader,
        total=total_batches,
        desc=f"train {state.epoch:03d}",
        disable=not is_main,
    )

    for batch_index, batch in enumerate(progress):
        if batch_index >= total_batches:
            break
        inputs, targets = _move_batch(batch, device)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        should_step = (batch_index + 1) % accumulation_steps == 0
        should_step = should_step or batch_index + 1 == total_batches
        synchronization = nullcontext()
        if isinstance(model, DistributedDataParallel) and not should_step:
            synchronization = model.no_sync()

        with synchronization:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                outputs = model(inputs)
            with torch.autocast(device_type=device.type, enabled=False):
                loss = criterion(
                    *(output.float() for output in outputs), targets.float()
                )
            scaler.scale(loss / group_size).backward()
        if should_step:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            optimizer_steps += 1

        loss_value = float(loss.detach())
        loss_sum += loss_value
        average_loss = loss_sum / (batch_index + 1)
        progress.set_postfix(loss=f"{average_loss:.3f}", step=state.global_step)
        if (
            is_main
            and log_every_batches > 0
            and (batch_index + 1) % log_every_batches == 0
        ):
            print(
                f"epoch={state.epoch} batch={batch_index + 1}/{total_batches} "
                f"step={state.global_step} loss={average_loss:.6f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    loss_sum, global_batches = reduce_sums(
        [loss_sum, float(total_batches)],
        device,
    )
    elapsed = reduce_max(elapsed, device)
    return {
        "loss": loss_sum / global_batches,
        "batches": int(global_batches),
        "optimizer_steps": optimizer_steps,
        "seconds": elapsed,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@torch.inference_mode()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool,
    max_samples: int | None,
    preview_path: Path | None,
    is_main: bool = True,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    sample_count = 0
    batch_count = 0
    start_time = time.perf_counter()

    progress = tqdm(loader, desc="validate", leave=False, disable=not is_main)
    for batch in progress:
        inputs, targets = _move_batch(batch, device)
        if max_samples is not None:
            remaining = max_samples - sample_count
            if remaining <= 0:
                break
            inputs = inputs[:remaining]
            targets = targets[:remaining]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            outputs = model(inputs)
        with torch.autocast(device_type=device.type, enabled=False):
            float_outputs = tuple(output.float() for output in outputs)
            float_targets = targets.float()
            loss = criterion(*float_outputs, float_targets)
            prediction = float_outputs[-1].clamp(0.0, 1.0)
            psnr_values = batch_psnr(prediction, float_targets)
            ssim_values = batch_ssim(prediction, float_targets)

        batch_size = inputs.shape[0]
        loss_sum += float(loss) * batch_size
        psnr_sum += float(psnr_values.sum())
        ssim_sum += float(ssim_values.sum())
        sample_count += batch_size
        batch_count += 1

        if is_main and preview_path is not None and sample_count == batch_size:
            save_preview(inputs[0], prediction[0], targets[0], preview_path)
        progress.set_postfix(psnr=f"{psnr_sum / sample_count:.3f}")

    elapsed = time.perf_counter() - start_time
    loss_sum, psnr_sum, ssim_sum, global_samples, global_batches = reduce_sums(
        [loss_sum, psnr_sum, ssim_sum, float(sample_count), float(batch_count)],
        device,
    )
    elapsed = reduce_max(elapsed, device)
    if global_samples == 0:
        raise ValueError("Validation loader produced no samples")
    return {
        "loss": loss_sum / global_samples,
        "psnr": psnr_sum / global_samples,
        "ssim": ssim_sum / global_samples,
        "samples": int(global_samples),
        "batches": int(global_batches),
        "seconds": elapsed,
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Any,
    state: TrainingState,
    config: dict[str, Any],
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    rng_states_by_rank: list[dict[str, Any]] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rng_states_by_rank is None:
        rng_states_by_rank = [capture_rng_state(train_generator, val_generator)]
    payload = {
        "format_version": 2,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_psnr": state.best_psnr,
        "config": config,
        "rng_states_by_rank": rng_states_by_rank,
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def restore_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Any,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    device: torch.device,
    rank: int = 0,
) -> TrainingState:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload.get("scaler", {}))
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if "rng_states_by_rank" in payload:
        rng_states = payload["rng_states_by_rank"]
        rng_state = rng_states[rank] if rank < len(rng_states) else rng_states[0]
        restore_rng_state(rng_state, train_generator, val_generator, device)
    else:
        _restore_legacy_rng_state(payload, train_generator, val_generator, device)
    return TrainingState(
        epoch=int(payload["epoch"]) + 1,
        global_step=int(payload["global_step"]),
        best_psnr=float(payload["best_psnr"]),
    )


def capture_rng_state(
    train_generator: torch.Generator,
    val_generator: torch.Generator,
) -> dict[str, Any]:
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device is not None else None,
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }


def restore_rng_state(
    rng_state: dict[str, Any],
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    device: torch.device,
) -> None:
    torch.set_rng_state(rng_state["torch"].cpu())
    if device.type == "cuda" and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state(rng_state["cuda"].cpu(), device)
    train_generator.set_state(rng_state["train_generator"].cpu())
    val_generator.set_state(rng_state["val_generator"].cpu())


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def save_preview(
    input_image: Tensor,
    prediction: Tensor,
    target: Tensor,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [_tensor_to_image(image) for image in (input_image, prediction, target)]
    canvas = np.concatenate(panels, axis=1)
    Image.fromarray(canvas).save(path)


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return inputs, targets


def _tensor_to_image(image: Tensor) -> np.ndarray:
    image = image.detach().float().clamp(0.0, 1.0)
    image = image.mul(255.0).round().byte().permute(1, 2, 0).cpu()
    return image.numpy()


def _restore_legacy_rng_state(
    payload: dict[str, Any],
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    device: torch.device,
) -> None:
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
        cuda_rng_states = [state.cpu() for state in payload["cuda_rng_state_all"]]
        torch.cuda.set_rng_state_all(cuda_rng_states)
    train_generator.set_state(payload["train_generator_state"].cpu())
    val_generator.set_state(payload["val_generator_state"].cpu())

