"""Evaluate a Uformer-B checkpoint on the complete TIP2018 test split."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_demoire import get_tip2018_validation_data  # noqa: E402
from checkpoint_utils import load_checkpoint, normalized_state_dict  # noqa: E402
from train.train_demoire import (  # noqa: E402
    batch_psnr,
    batch_ssim,
    build_uformer_b,
    gaussian_window,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Uformer-B on TIP2018")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--save_dir", default="", help="optional directory for restored PNG images"
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = build_uformer_b().to(device)
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = normalized_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


def save_batch(images, names, save_dir):
    for image, name in zip(images, names):
        array = (
            image.detach()
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        Image.fromarray(np.asarray(array)).save(save_dir / f"{name}.png")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = not args.no_amp and device.type == "cuda"
    dataset = get_tip2018_validation_data(args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model, epoch = load_model(args.checkpoint, device)
    window = gaussian_window(3, device)
    psnr_sum = 0.0
    ssim_sum = 0.0
    sample_count = 0

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, (target, source, names) in enumerate(
        tqdm(loader, desc="TIP2018 evaluation", dynamic_ncols=True)
    ):
        if args.max_batches and batch_index >= args.max_batches:
            break
        target = target.to(device, non_blocking=True)
        source = source.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            restored = model(source)
        restored = restored.float().clamp(0, 1)
        target = target.float().clamp(0, 1)
        psnr_sum += batch_psnr(restored, target).sum().item()
        ssim_sum += batch_ssim(restored, target, window).sum().item()
        sample_count += target.shape[0]
        if save_dir is not None:
            save_batch(restored, names, save_dir)

    if sample_count == 0:
        raise RuntimeError("Evaluation processed no samples")
    print(
        f"checkpoint={args.checkpoint} epoch={epoch} samples={sample_count} "
        f"PSNR={psnr_sum / sample_count:.4f} "
        f"SSIM={ssim_sum / sample_count:.4f}"
    )


if __name__ == "__main__":
    main()
