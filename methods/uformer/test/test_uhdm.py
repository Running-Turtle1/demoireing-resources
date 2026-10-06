"""Distributed full-resolution demoireing evaluation for Uformer-B."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_uhdm import get_uhdm_validation_data  # noqa: E402
from checkpoint_utils import load_checkpoint, normalized_state_dict  # noqa: E402
from dataset.dataset_fhdmi import get_fhdmi_validation_data  # noqa: E402
from dataset.dataset_lcdmoire import get_lcdmoire_validation_data  # noqa: E402
from train.train_demoire import (  # noqa: E402
    DistributedEvalSampler,
    batch_psnr,
    batch_ssim,
    gaussian_window,
    setup_distributed,
)
from train.train_demoire_uhdm import (  # noqa: E402
    build_uformer_b,
    fhdmi_psnr_ssim,
    lcdmoire_psnr_ssim,
    tiled_restore,
)


def parse_args(dataset_name="UHDM"):
    dataset_name = dataset_name.upper()
    if dataset_name not in {"UHDM", "FHDMI", "LCDMOIRE"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    is_fhdmi = dataset_name == "FHDMI"
    is_lcdmoire = dataset_name == "LCDMOIRE"
    parser = argparse.ArgumentParser(
        description=f"Evaluate Uformer-B on {dataset_name}"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data_root",
        required=True,
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024 if is_lcdmoire else (512 if is_fhdmi else 768),
        help=(
            "inference tile size; LCDMoire defaults to one full 1024x1024 "
            "forward"
        ),
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=0 if is_lcdmoire else 128,
        help="tile overlap; LCDMoire full-image evaluation defaults to 0",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", default="")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()
    if args.tile_size <= 0:
        parser.error("--tile_size must be positive")
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        parser.error("--tile_overlap must satisfy 0 <= overlap < tile_size")
    return args


def load_model(checkpoint_path, device, image_size):
    model = build_uformer_b(image_size=image_size).to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = normalized_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return model, epoch


def save_image(image, path):
    array = (
        image.clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(np.asarray(array)).save(path)


@torch.no_grad()
def main(dataset_name="UHDM"):
    args = parse_args(dataset_name)
    is_fhdmi = dataset_name.upper() == "FHDMI"
    is_lcdmoire = dataset_name.upper() == "LCDMOIRE"
    distributed, rank, world_size, _, device = setup_distributed()
    is_main = rank == 0
    amp_enabled = not args.no_amp and device.type == "cuda"
    if is_fhdmi:
        dataset = get_fhdmi_validation_data(args.data_root)
        expected_count = 2019
    elif is_lcdmoire:
        dataset = get_lcdmoire_validation_data(args.data_root)
        expected_count = 100
    else:
        dataset = get_uhdm_validation_data(args.data_root)
        expected_count = 500
    if len(dataset) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} {dataset_name} test pairs, "
            f"found {len(dataset)}"
        )
    sampler = (
        DistributedEvalSampler(dataset, rank, world_size) if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model, epoch = load_model(args.checkpoint, device, args.tile_size)
    window = gaussian_window(3, device)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(
        loader,
        disable=not is_main,
        desc=f"{dataset_name} full-resolution evaluation",
        dynamic_ncols=True,
    )
    for image_index, (target, source, names) in enumerate(progress):
        if args.max_images and image_index >= args.max_images:
            break
        target = target[0]
        source = source[0]
        restored = tiled_restore(
            model,
            source,
            device,
            args.tile_size,
            args.tile_overlap,
            amp_enabled,
        ).clamp(0, 1)
        if is_fhdmi:
            image_psnr, image_ssim = fhdmi_psnr_ssim(restored, target)
            totals[0] += image_psnr
            totals[1] += image_ssim
        elif is_lcdmoire:
            image_psnr, image_ssim = lcdmoire_psnr_ssim(restored, target)
            totals[0] += image_psnr
            totals[1] += image_ssim
        else:
            restored_device = restored.unsqueeze(0).to(device)
            target_device = target.unsqueeze(0).to(device)
            totals[0] += batch_psnr(restored_device, target_device).double().sum()
            totals[1] += batch_ssim(restored_device, target_device, window).double().sum()
        totals[2] += 1
        if save_dir is not None:
            save_image(restored, save_dir / f"{names[0]}.png")

    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = int(totals[2].item())
    if is_main:
        print(
            f"checkpoint={args.checkpoint} epoch={epoch} images={count} "
            f"PSNR={totals[0].item() / count:.4f} "
            f"SSIM={totals[1].item() / count:.4f}"
        )
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
