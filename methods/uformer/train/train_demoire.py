"""Train Uformer-B on TIP2018 with multi-GPU DistributedDataParallel."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_demoire import (  # noqa: E402
    get_tip2018_training_data,
    get_tip2018_validation_data,
)
from losses import CharbonnierLoss  # noqa: E402
from model import Uformer  # noqa: E402


class DistributedEvalSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.world_size + 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Uformer-B for TIP2018 image demoireing"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="TIP2018 root containing trainData/ and testData/",
    )
    parser.add_argument(
        "--output_dir",
        default="./logs/demoireing/TIP2018/Uformer_B_uhdm_preprocess",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument(
        "--batch_size", type=int, default=4, help="training batch size per GPU"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="evaluation batch size per GPU"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="workers per rank")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", default="", help="checkpoint path to resume")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="disable paired flips and 90-degree rotations",
    )
    parser.add_argument("--no_amp", action="store_true", help="disable FP16 AMP")
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="limit batches for a smoke test; 0 means all",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=0,
        help="limit validation batches for a smoke test; 0 means all",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="skip checkpoints for a smoke test",
    )
    args = parser.parse_args()

    for name in ("epochs", "batch_size", "eval_batch_size", "eval_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed GPU training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl").lower()
        if backend not in {"nccl", "gloo"}:
            raise RuntimeError(
                "TORCH_DISTRIBUTED_BACKEND must be either 'nccl' or 'gloo'"
            )
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    return distributed, rank, world_size, local_rank, device


def seed_everything(seed, rank):
    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def build_uformer_b():
    return Uformer(
        img_size=256,
        in_chans=3,
        dd_in=3,
        embed_dim=32,
        depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
        win_size=8,
        token_projection="linear",
        token_mlp="leff",
        modulator=True,
    )


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def gaussian_window(channels, device, dtype=torch.float32):
    coordinates = torch.arange(11, device=device, dtype=dtype) - 5
    gaussian = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    return window.expand(channels, 1, 11, 11).contiguous()


def batch_psnr(prediction, target):
    mse = (prediction - target).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def batch_ssim(prediction, target, window):
    channels = prediction.shape[1]
    mu_prediction = F.conv2d(prediction, window, groups=channels)
    mu_target = F.conv2d(target, window, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    sigma_prediction = (
        F.conv2d(prediction.square(), window, groups=channels) - mu_prediction_sq
    )
    sigma_target = F.conv2d(target.square(), window, groups=channels) - mu_target_sq
    sigma_product = (
        F.conv2d(prediction * target, window, groups=channels) - mu_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_product + c1) * (2 * sigma_product + c2)) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (sigma_prediction + sigma_target + c2)
    )
    return ssim_map.flatten(1).mean(1)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, distributed, max_batches=0):
    evaluation_model = unwrap_model(model)
    evaluation_model.eval()
    window = gaussian_window(3, device)
    metric_totals = torch.zeros(3, dtype=torch.float64, device=device)

    for batch_index, (target, source, _) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        target = target.to(device, non_blocking=True)
        source = source.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            restored = evaluation_model(source)
        restored = restored.float().clamp(0, 1)
        target = target.float().clamp(0, 1)
        metric_totals[0] += batch_psnr(restored, target).double().sum()
        metric_totals[1] += batch_ssim(restored, target, window).double().sum()
        metric_totals[2] += target.shape[0]

    if distributed:
        dist.all_reduce(metric_totals, op=dist.ReduceOp.SUM)
    count = int(metric_totals[2].item())
    if count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return metric_totals[0].item() / count, metric_totals[1].item() / count, count


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, epoch, best_psnr, best_ssim, args
):
    state = {
        "epoch": epoch,
        "state_dict": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "args": vars(args),
    }
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    load_scheduler=True,
    optimizer_lr=None,
):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"]
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if optimizer_lr is not None:
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = optimizer_lr
            parameter_group["initial_lr"] = optimizer_lr
    if load_scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_psnr", -math.inf)),
        float(checkpoint.get("best_ssim", -math.inf)),
    )


def append_log(log_path, record):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed()
    is_main = rank == 0
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

    train_dataset = get_tip2018_training_data(
        args.data_root, augment=not args.no_augment
    )
    validation_dataset = get_tip2018_validation_data(args.data_root)
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
    loader_options = {
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
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        **loader_options,
    )

    model = build_uformer_b().to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
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
            args.resume, model, optimizer, scheduler, scaler, device
        )

    if is_main:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Uformer-B | parameters={parameter_count / 1e6:.2f}M | "
            f"world_size={world_size} | per_gpu_batch={args.batch_size} | "
            f"global_batch={args.batch_size * world_size}"
        )
        print(
            f"train_pairs={len(train_dataset)} | val_pairs={len(validation_dataset)} | "
            f"eval_every={args.eval_every}"
        )
        append_log(
            log_path,
            {
                "event": "start",
                "args": vars(args),
                "world_size": world_size,
                "parameters": parameter_count,
            },
        )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_start = time.time()
            loss_total = torch.zeros(2, dtype=torch.float64, device=device)
            progress = tqdm(
                train_loader,
                disable=not is_main,
                desc=f"Epoch {epoch}/{args.epochs}",
                dynamic_ncols=True,
            )

            for batch_index, (target, source, _) in enumerate(progress):
                if args.max_train_batches and batch_index >= args.max_train_batches:
                    break
                target = target.to(device, non_blocking=True)
                source = source.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    restored = model(source)
                    loss = criterion(restored, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_count = target.shape[0]
                loss_total[0] += loss.detach().double() * batch_count
                loss_total[1] += batch_count
                if is_main and (batch_index + 1) % args.log_interval == 0:
                    progress.set_postfix(
                        loss=f"{loss.item():.5f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

            if distributed:
                dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
            train_loss = loss_total[0].item() / loss_total[1].item()
            scheduler.step()

            should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
            validation_psnr = None
            validation_ssim = None
            validation_count = None
            if should_evaluate:
                validation_psnr, validation_ssim, validation_count = evaluate(
                    model,
                    validation_loader,
                    device,
                    amp_enabled,
                    distributed,
                    args.max_val_batches,
                )

            if is_main:
                elapsed = time.time() - epoch_start
                record = {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
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
                        f"SSIM={validation_ssim:.4f}"
                    )
                print(
                    f"Epoch {epoch}: loss={train_loss:.6f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}{metrics_text} | "
                    f"time={elapsed:.1f}s"
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
