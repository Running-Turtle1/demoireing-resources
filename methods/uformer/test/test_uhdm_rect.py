"""Compare tiled and rectangular full-image inference on UHDM."""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_uhdm import get_uhdm_validation_data  # noqa: E402
from checkpoint_utils import load_checkpoint, normalized_state_dict  # noqa: E402
from train.train_demoire import (  # noqa: E402
    DistributedEvalSampler,
    batch_psnr,
    batch_ssim,
    gaussian_window,
    setup_distributed,
)
from train.train_demoire_uhdm import (  # noqa: E402
    build_uformer_b,
    tiled_restore,
)


ESDNET_PADDING_RGB = (0.3827, 0.4141, 0.3912)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Uformer-B tiled and rectangular UHDM inference"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data_root",
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("tiled", "full"),
        default=("tiled", "full"),
    )
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--tile_size", type=int, default=768)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--pad_multiple", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num_images must be positive")
    if args.tile_size <= 0:
        parser.error("--tile_size must be positive")
    if args.tile_overlap < 0 or args.tile_overlap >= args.tile_size:
        parser.error("--tile_overlap must satisfy 0 <= overlap < tile_size")
    if args.pad_multiple <= 0:
        parser.error("--pad_multiple must be positive")
    args.modes = tuple(dict.fromkeys(args.modes))
    return args


def load_model(checkpoint_path, device, image_size):
    checkpoint = load_checkpoint(checkpoint_path)
    state_dict = normalized_state_dict(checkpoint)
    model = build_uformer_b(image_size=image_size)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {load_result}")
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    del checkpoint, state_dict
    return model.to(device).eval(), epoch


def pad_with_esdnet_constants(source, multiple):
    if source.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(source.shape)}")
    _, height, width = source.shape
    padded_height = math.ceil(height / multiple) * multiple
    padded_width = math.ceil(width / multiple) * multiple
    vertical = padded_height - height
    horizontal = padded_width - width
    top = vertical // 2
    bottom = vertical - top
    left = horizontal // 2
    right = horizontal - left
    padded = torch.cat(
        [
            F.pad(
                source[channel : channel + 1],
                (left, right, top, bottom),
                value=ESDNET_PADDING_RGB[channel],
            )
            for channel in range(3)
        ],
        dim=0,
    )
    return padded, (left, right, top, bottom)


@torch.inference_mode()
def full_restore(model, source, device, amp_enabled, pad_multiple):
    _, original_height, original_width = source.shape
    padded, padding = pad_with_esdnet_constants(source, pad_multiple)
    left, _, top, _ = padding
    padded = padded.unsqueeze(0).to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        restored = model(padded)
    restored = restored[0].float().cpu()
    return restored[
        :, top : top + original_height, left : left + original_width
    ], padding


def image_metrics(restored, target, device, window):
    restored_device = restored.clamp(0, 1).unsqueeze(0).to(device)
    target_device = target.clamp(0, 1).unsqueeze(0).to(device)
    psnr = batch_psnr(restored_device, target_device).item()
    ssim = batch_ssim(restored_device, target_device, window).item()
    del restored_device, target_device
    return psnr, ssim


def run_mode(args, mode, model, source, device, amp_enabled):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if mode == "full":
        restored, padding = full_restore(
            model, source, device, amp_enabled, args.pad_multiple
        )
    else:
        restored = tiled_restore(
            model,
            source,
            device,
            args.tile_size,
            args.tile_overlap,
            amp_enabled,
        )
        padding = (0, 0, 0, 0)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    allocated_peak = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved_peak = torch.cuda.max_memory_reserved(device) / 1024**3
    return restored, padding, seconds, allocated_peak, reserved_peak


def aggregate(records, modes):
    summaries = {}
    for mode in modes:
        selected = [record for record in records if record["mode"] == mode]
        summaries[mode] = {
            "images": len(selected),
            "psnr": sum(record["psnr"] for record in selected) / len(selected),
            "ssim": sum(record["ssim"] for record in selected) / len(selected),
            "mean_seconds": sum(record["seconds"] for record in selected)
            / len(selected),
            "max_allocated_peak_gib": max(
                record["allocated_peak_gib"] for record in selected
            ),
            "max_reserved_peak_gib": max(
                record["reserved_peak_gib"] for record in selected
            ),
        }
    return summaries


def write_results(args, epoch, world_size, records):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda row: (row["name"], row["mode"]))
    fieldnames = [
        "name",
        "mode",
        "height",
        "width",
        "pad_left",
        "pad_right",
        "pad_top",
        "pad_bottom",
        "psnr",
        "ssim",
        "seconds",
        "allocated_peak_gib",
        "reserved_peak_gib",
        "rank",
    ]
    with (output_dir / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summaries = aggregate(records, args.modes)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": epoch,
        "world_size": world_size,
        "precision": "FP32" if args.no_amp else "AMP FP16",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "sample_selection": f"first {args.num_images} sorted UHDM test pairs",
        "modes": list(args.modes),
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "pad_multiple": args.pad_multiple,
        "padding_rgb_constants": list(ESDNET_PADDING_RGB),
        "summary": summaries,
    }
    if "full" in summaries and "tiled" in summaries:
        result["full_minus_tiled"] = {
            "psnr": summaries["full"]["psnr"] - summaries["tiled"]["psnr"],
            "ssim": summaries["full"]["ssim"] - summaries["tiled"]["ssim"],
        }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main():
    args = parse_args()
    distributed, rank, world_size, _, device = setup_distributed()
    is_main = rank == 0
    if device.type != "cuda":
        raise RuntimeError("This memory-sensitive evaluation requires CUDA")
    amp_enabled = not args.no_amp
    # cuDNN autotuning for the first 4K rectangular forward takes roughly
    # 90 seconds and requests workspace close to the 32 GiB device limit.
    # The deterministic shape does not need that risky one-time search.
    torch.backends.cudnn.benchmark = False

    full_dataset = get_uhdm_validation_data(args.data_root)
    if len(full_dataset) != 500:
        raise RuntimeError(f"Expected 500 UHDM test pairs, found {len(full_dataset)}")
    if args.num_images > len(full_dataset):
        raise RuntimeError(
            f"Requested {args.num_images} images from a {len(full_dataset)}-image set"
        )
    dataset = Subset(full_dataset, range(args.num_images))
    sampler = (
        DistributedEvalSampler(dataset, rank, world_size) if distributed else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model, epoch = load_model(args.checkpoint, device, args.tile_size)
    window = gaussian_window(3, device)
    local_records = []

    progress = tqdm(
        loader,
        disable=not is_main,
        desc="UHDM full/tiled comparison",
        dynamic_ncols=True,
    )
    for target_batch, source_batch, names in progress:
        target = target_batch[0]
        source = source_batch[0]
        _, height, width = source.shape
        for mode in args.modes:
            restored, padding, seconds, allocated, reserved = run_mode(
                args, mode, model, source, device, amp_enabled
            )
            psnr, ssim = image_metrics(restored, target, device, window)
            left, right, top, bottom = padding
            local_records.append(
                {
                    "name": names[0],
                    "mode": mode,
                    "height": height,
                    "width": width,
                    "pad_left": left,
                    "pad_right": right,
                    "pad_top": top,
                    "pad_bottom": bottom,
                    "psnr": psnr,
                    "ssim": ssim,
                    "seconds": seconds,
                    "allocated_peak_gib": allocated,
                    "reserved_peak_gib": reserved,
                    "rank": rank,
                }
            )
            del restored

    if distributed:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_records)
        records = [record for rank_records in gathered for record in rank_records]
    else:
        records = local_records

    if is_main:
        expected_records = args.num_images * len(args.modes)
        if len(records) != expected_records:
            raise RuntimeError(
                f"Expected {expected_records} records, collected {len(records)}"
            )
        result = write_results(args, epoch, world_size, records)
        for mode in args.modes:
            summary = result["summary"][mode]
            print(
                f"RESULT mode={mode} epoch={epoch} images={summary['images']} "
                f"PSNR={summary['psnr']:.6f} SSIM={summary['ssim']:.8f} "
                f"mean_time={summary['mean_seconds']:.3f}s "
                f"peak={summary['max_allocated_peak_gib']:.3f}/"
                f"{summary['max_reserved_peak_gib']:.3f}GiB"
            )
        if "full_minus_tiled" in result:
            difference = result["full_minus_tiled"]
            print(
                f"DELTA full-minus-tiled PSNR={difference['psnr']:+.6f} "
                f"SSIM={difference['ssim']:+.8f}"
            )
        print(f"results={Path(args.output_dir).resolve()}")

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
