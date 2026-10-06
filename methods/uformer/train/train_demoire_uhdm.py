"""Train Uformer-B on full-resolution demoireing datasets with DDP."""

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_uhdm import (  # noqa: E402
    get_uhdm_training_data,
    get_uhdm_validation_data,
)
from dataset.dataset_fhdmi import (  # noqa: E402
    get_fhdmi_training_data,
    get_fhdmi_validation_data,
)
from dataset.dataset_lcdmoire import (  # noqa: E402
    get_lcdmoire_training_data,
    get_lcdmoire_validation_data,
)
from losses import CharbonnierLoss  # noqa: E402
from model import Uformer  # noqa: E402
from train.train_demoire import (  # noqa: E402
    DistributedEvalSampler,
    append_log,
    batch_psnr,
    batch_ssim,
    gaussian_window,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    unwrap_model,
)


def parse_args(dataset_name="UHDM"):
    dataset_name = dataset_name.upper()
    if dataset_name not in {"UHDM", "FHDMI", "LCDMOIRE"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    is_fhdmi = dataset_name == "FHDMI"
    is_lcdmoire = dataset_name == "LCDMOIRE"
    parser = argparse.ArgumentParser(
        description=(
            f"Train Uformer-B on {dataset_name} with official aligned-crop "
            "preprocessing"
        )
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="dataset root; see the corresponding dataset card",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "./logs/demoireing/FHDMi/Uformer_B_official_preprocess"
            if is_fhdmi
            else (
                "./logs/demoireing/LCDMoire/Uformer_B_official_preprocess"
                if is_lcdmoire
                else "./logs/demoireing/UHDM/Uformer_B_uhdm_preprocess"
            )
        ),
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--crop_size", type=int, default=512 if is_fhdmi or is_lcdmoire else 768
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="micro-batch size per GPU"
    )
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4, help="workers per rank")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024 if is_lcdmoire else (512 if is_fhdmi else 768),
        help=(
            "validation tile size; LCDMoire defaults to its full 1024x1024 "
            "image size"
        ),
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=0 if is_lcdmoire else 128,
        help="validation tile overlap; LCDMoire full-image evaluation uses 0",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--reset_scheduler_on_resume",
        action="store_true",
        help=(
            "load model/optimizer/scaler from --resume but start with the new "
            "scheduler configuration"
        ),
    )
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="recompute Transformer activations during backward to save memory",
    )
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument(
        "--max_epochs_this_run",
        type=int,
        default=0,
        help="limit epochs in this invocation for a smoke test; 0 means all",
    )
    parser.add_argument(
        "--max_val_images",
        type=int,
        default=0,
        help="maximum images per rank for a smoke test; 0 means all",
    )
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()
    args.dataset_name = (
        "FHDMi" if is_fhdmi else ("LCDMoire" if is_lcdmoire else "UHDM")
    )

    positive_names = (
        "epochs",
        "crop_size",
        "batch_size",
        "grad_accum_steps",
        "eval_every",
        "tile_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("max_train_batches", "max_epochs_this_run", "max_val_images"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative")
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        parser.error("--tile_overlap must satisfy 0 <= overlap < tile_size")
    if args.reset_scheduler_on_resume and not args.resume:
        parser.error("--reset_scheduler_on_resume requires --resume")
    if not is_lcdmoire and args.tile_size != args.crop_size:
        parser.error("--tile_size must equal --crop_size for UHDM and FHDMi")
    if is_lcdmoire and args.tile_size not in (args.crop_size, 1024):
        parser.error("LCDMoire --tile_size must equal --crop_size or 1024")
    return args


def build_uformer_b(image_size=768, activation_checkpointing=False):
    return Uformer(
        img_size=image_size,
        in_chans=3,
        dd_in=3,
        embed_dim=32,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        win_size=8,
        token_projection="linear",
        token_mlp="leff",
        modulator=True,
        use_checkpoint=activation_checkpointing,
    )


def tile_starts(length, tile_size, overlap):
    if length < tile_size:
        raise ValueError(f"Image side {length} is smaller than tile size {tile_size}")
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def tile_weight(tile_size):
    one_dimensional = torch.hann_window(tile_size, periodic=False, dtype=torch.float32)
    one_dimensional = one_dimensional.clamp_min(1e-3)
    return torch.outer(one_dimensional, one_dimensional).unsqueeze(0)


@torch.no_grad()
def tiled_restore(model, source, device, tile_size, overlap, amp_enabled):
    """Restore one CHW full-resolution image using overlap-add on CPU."""
    if source.ndim != 3:
        raise ValueError(f"Expected CHW source tensor, got shape {tuple(source.shape)}")
    _, height, width = source.shape
    evaluation_model = unwrap_model(model)

    # LCDMoire validation images are exactly 1024x1024. Avoid overlap-add when
    # one tile covers the complete image: this is a true single full-image
    # forward and matches the official ESDNet inference geometry.
    if height == tile_size and width == tile_size:
        source_batch = source.unsqueeze(0).to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            restored = evaluation_model(source_batch)
        return restored[0].float().cpu()

    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    weight = tile_weight(tile_size)
    restored_sum = torch.zeros_like(source, dtype=torch.float32, device="cpu")
    weight_sum = torch.zeros((1, height, width), dtype=torch.float32)
    for top in y_starts:
        for left in x_starts:
            source_tile = source[
                :, top : top + tile_size, left : left + tile_size
            ].unsqueeze(0)
            source_tile = source_tile.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                restored_tile = evaluation_model(source_tile)
            restored_tile = restored_tile[0].float().cpu()
            restored_sum[
                :, top : top + tile_size, left : left + tile_size
            ] += restored_tile * weight
            weight_sum[
                :, top : top + tile_size, left : left + tile_size
            ] += weight

    return restored_sum / weight_sum.clamp_min(1e-8)


def fhdmi_psnr_ssim(prediction, target):
    """Compute the uint8 scikit-image metrics used by the FHDMi benchmark."""
    try:
        from skimage.metrics import peak_signal_noise_ratio
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError(
            "FHDMi evaluation requires scikit-image; install scikit-image==0.21.0"
        ) from error

    prediction_array = (
        prediction.clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    target_array = (
        target.clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    psnr = peak_signal_noise_ratio(target_array, prediction_array, data_range=255)
    ssim = structural_similarity(
        target_array,
        prediction_array,
        channel_axis=2,
        data_range=255,
    )
    return float(psnr), float(ssim)


def lcdmoire_psnr_ssim(prediction, target):
    """Compute the quantized float RGB metrics used by the AIM benchmark."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError(
            "LCDMoire evaluation requires scikit-image; install scikit-image"
        ) from error

    prediction_array = (
        prediction.clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
        / 255.0
    )
    target_array = (
        target.clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
        / 255.0
    )
    mean_squared_error = np.mean((target_array - prediction_array) ** 2)
    psnr = math.inf if mean_squared_error == 0 else 10 * math.log10(
        1.0 / mean_squared_error
    )
    ssim = structural_similarity(
        target_array,
        prediction_array,
        channel_axis=2,
        data_range=1.0,
    )
    return float(psnr), float(ssim)


@torch.no_grad()
def evaluate_full_resolution(
    model,
    loader,
    device,
    amp_enabled,
    distributed,
    tile_size,
    tile_overlap,
    metric_mode="uhdm",
    max_images=0,
):
    evaluation_model = unwrap_model(model)
    evaluation_model.eval()
    window = gaussian_window(3, device)
    totals = torch.zeros(3, dtype=torch.float64, device=device)

    for image_index, (target, source, _) in enumerate(loader):
        if max_images and image_index >= max_images:
            break
        target = target[0]
        source = source[0]
        restored = tiled_restore(
            evaluation_model,
            source,
            device,
            tile_size,
            tile_overlap,
            amp_enabled,
        ).clamp(0, 1)
        if metric_mode == "fhdmi":
            image_psnr, image_ssim = fhdmi_psnr_ssim(restored, target)
            totals[0] += image_psnr
            totals[1] += image_ssim
        elif metric_mode == "lcdmoire":
            image_psnr, image_ssim = lcdmoire_psnr_ssim(restored, target)
            totals[0] += image_psnr
            totals[1] += image_ssim
        else:
            restored_device = restored.unsqueeze(0).to(device)
            target_device = target.unsqueeze(0).to(device)
            totals[0] += batch_psnr(restored_device, target_device).double().sum()
            totals[1] += batch_ssim(restored_device, target_device, window).double().sum()
            del restored_device, target_device
        totals[2] += 1
        del restored

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = int(totals[2].item())
    if count == 0:
        raise RuntimeError("Full-resolution validation processed no images")
    return totals[0].item() / count, totals[1].item() / count, count


def main(dataset_name="UHDM"):
    args = parse_args(dataset_name)
    distributed, rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
    distributed_backend = dist.get_backend() if distributed else "none"
    amp_enabled = not args.no_amp and device.type == "cuda"
    seed_everything(args.seed, rank)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir).resolve()
    model_dir = output_dir / "models"
    if is_main:
        model_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log_path = output_dir / "train.jsonl"

    if args.dataset_name == "FHDMi":
        train_dataset = get_fhdmi_training_data(args.data_root, args.crop_size)
        validation_dataset = get_fhdmi_validation_data(args.data_root)
        expected_train, expected_validation = 9982, 2019
        metric_mode = "fhdmi"
    elif args.dataset_name == "LCDMoire":
        train_dataset = get_lcdmoire_training_data(args.data_root, args.crop_size)
        validation_dataset = get_lcdmoire_validation_data(args.data_root)
        expected_train, expected_validation = 10000, 100
        metric_mode = "lcdmoire"
    else:
        train_dataset = get_uhdm_training_data(args.data_root, args.crop_size)
        validation_dataset = get_uhdm_validation_data(args.data_root)
        expected_train, expected_validation = 4500, 500
        metric_mode = "uhdm"
    if (
        len(train_dataset) != expected_train
        or len(validation_dataset) != expected_validation
    ):
        raise RuntimeError(
            f"Unexpected {args.dataset_name} split sizes: "
            f"train={len(train_dataset)} (expected {expected_train}), "
            f"validation={len(validation_dataset)} (expected {expected_validation})"
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    validation_sampler = (
        DistributedEvalSampler(validation_dataset, rank, world_size)
        if distributed
        else None
    )
    common_loader_options = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **common_loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        **common_loader_options,
    )

    model = build_uformer_b(
        image_size=args.crop_size,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    criterion = CharbonnierLoss(eps=1e-3).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_psnr = -math.inf
    best_ssim = -math.inf
    if args.resume:
        start_epoch, best_psnr, best_ssim = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            load_scheduler=not args.reset_scheduler_on_resume,
            optimizer_lr=args.lr if args.reset_scheduler_on_resume else None,
        )

    if is_main:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Uformer-B {args.dataset_name} | "
            f"parameters={parameter_count / 1e6:.2f}M | "
            f"world_size={world_size} | micro_batch={args.batch_size} | "
            f"grad_accum={args.grad_accum_steps} | effective_batch="
            f"{world_size * args.batch_size * args.grad_accum_steps} | "
            f"backend={distributed_backend} | "
            f"activation_checkpointing={args.activation_checkpointing}"
        )
        print(
            f"train_pairs={len(train_dataset)} | val_pairs={len(validation_dataset)} | "
            f"crop={args.crop_size} | eval_every={args.eval_every}"
        )
        append_log(
            log_path,
            {
                "event": "start",
                "args": vars(args),
                "world_size": world_size,
                "distributed_backend": distributed_backend,
                "parameters": parameter_count,
            },
        )

    try:
        end_epoch = args.epochs
        if args.max_epochs_this_run:
            end_epoch = min(
                args.epochs, start_epoch + args.max_epochs_this_run - 1
            )
        for epoch in range(start_epoch, end_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.time()
            loss_totals = torch.zeros(2, dtype=torch.float64, device=device)
            total_batches = len(train_loader)
            if args.max_train_batches:
                total_batches = min(total_batches, args.max_train_batches)
            progress = tqdm(
                train_loader,
                total=total_batches,
                disable=not is_main,
                desc=f"Epoch {epoch}/{args.epochs}",
                dynamic_ncols=True,
            )
            optimizer.zero_grad(set_to_none=True)

            for batch_index, (target, source, _) in enumerate(progress):
                if batch_index >= total_batches:
                    break
                group_start = (
                    batch_index // args.grad_accum_steps
                ) * args.grad_accum_steps
                group_size = min(
                    args.grad_accum_steps, total_batches - group_start
                )
                group_position = batch_index - group_start + 1
                should_step = group_position == group_size
                synchronization_context = (
                    model.no_sync()
                    if distributed and not should_step
                    else contextlib.nullcontext()
                )

                target = target.to(device, non_blocking=True)
                source = source.to(device, non_blocking=True)
                with synchronization_context:
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        restored = model(source)
                        loss = criterion(restored, target)
                    scaler.scale(loss / group_size).backward()

                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                sample_count = target.shape[0]
                loss_totals[0] += loss.detach().double() * sample_count
                loss_totals[1] += sample_count
                if is_main and (batch_index + 1) % args.log_interval == 0:
                    progress.set_postfix(
                        loss=f"{loss.item():.5f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

            if distributed:
                dist.all_reduce(loss_totals, op=dist.ReduceOp.SUM)
            train_loss = loss_totals[0].item() / loss_totals[1].item()
            memory_peak = torch.zeros(2, dtype=torch.float64, device=device)
            if device.type == "cuda":
                memory_peak[0] = torch.cuda.max_memory_allocated(device)
                memory_peak[1] = torch.cuda.max_memory_reserved(device)
            if distributed:
                dist.all_reduce(memory_peak, op=dist.ReduceOp.MAX)
            peak_allocated_gib = memory_peak[0].item() / (1024**3)
            peak_reserved_gib = memory_peak[1].item() / (1024**3)
            scheduler.step()

            should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
            validation_psnr = None
            validation_ssim = None
            validation_count = None
            if should_evaluate:
                (
                    validation_psnr,
                    validation_ssim,
                    validation_count,
                ) = evaluate_full_resolution(
                    model,
                    validation_loader,
                    device,
                    amp_enabled,
                    distributed,
                    args.tile_size,
                    args.tile_overlap,
                    metric_mode,
                    args.max_val_images,
                )

            if is_main:
                elapsed = time.time() - epoch_start
                record = {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "peak_allocated_gib": peak_allocated_gib,
                    "peak_reserved_gib": peak_reserved_gib,
                    "seconds": elapsed,
                }
                if should_evaluate:
                    record.update(
                        {
                            "val_psnr": validation_psnr,
                            "val_ssim": validation_ssim,
                            "val_count": validation_count,
                        }
                    )
                append_log(log_path, record)
                metrics_text = ""
                if should_evaluate:
                    metrics_text = (
                        f" | PSNR={validation_psnr:.4f} | "
                        f"SSIM={validation_ssim:.4f} | images={validation_count}"
                    )
                print(
                    f"Epoch {epoch}: loss={train_loss:.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}{metrics_text} | "
                    f"train_peak={peak_allocated_gib:.2f}/{peak_reserved_gib:.2f}GiB "
                    f"(allocated/reserved) | time={elapsed:.1f}s"
                )

                if should_evaluate and validation_psnr > best_psnr:
                    best_psnr = validation_psnr
                    best_ssim = validation_ssim
                    if not args.no_save:
                        save_checkpoint(
                            model_dir / "model_best.pth",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            best_psnr,
                            best_ssim,
                            args,
                        )
                if not args.no_save:
                    save_checkpoint(
                        model_dir / "model_latest.pth",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_psnr,
                        best_ssim,
                        args,
                    )
                    if (
                        args.checkpoint_every > 0
                        and epoch % args.checkpoint_every == 0
                    ):
                        save_checkpoint(
                            model_dir / f"model_epoch_{epoch}.pth",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            best_psnr,
                            best_ssim,
                            args,
                        )
            if distributed:
                dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
