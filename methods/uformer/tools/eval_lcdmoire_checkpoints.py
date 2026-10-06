"""Evaluate every numeric LCDMoire checkpoint with full-image inference.

The sweep keeps one Uformer-B instance and one distributed validation loader
alive while checkpoint weights are loaded in epoch order. Results are appended
after every checkpoint so an interrupted run can continue with ``--resume``.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_lcdmoire import get_lcdmoire_validation_data  # noqa: E402
from train.train_demoire import DistributedEvalSampler, setup_distributed  # noqa: E402
from train.train_demoire_uhdm import (  # noqa: E402
    build_uformer_b,
    evaluate_full_resolution,
)


CHECKPOINT_PATTERN = re.compile(r"model_epoch_(\d+)\.pth")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep LCDMoire checkpoints using 1024x1024 full-image evaluation"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=(
            "./logs/demoireing/LCDMoire/"
            "Uformer_B_6gpu_restart/models"
        ),
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--output_dir",
        default=(
            "./logs/demoireing/LCDMoire/"
            "Uformer_B_6gpu_restart/eval_full_image_sweep"
        ),
    )
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--end_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="images per rank for a smoke test; 0 evaluates all 100 images",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip checkpoints already recorded in results.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing results.jsonl instead of resuming it",
    )
    parser.add_argument(
        "--list_only", action="store_true", help="list checkpoints and exit"
    )
    args = parser.parse_args()
    if args.start_epoch <= 0:
        parser.error("--start_epoch must be positive")
    if args.end_epoch < 0:
        parser.error("--end_epoch cannot be negative")
    if args.end_epoch and args.end_epoch < args.start_epoch:
        parser.error("--end_epoch must be 0 or >= --start_epoch")
    if args.num_workers < 0 or args.max_images < 0:
        parser.error("--num_workers and --max_images cannot be negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def discover_checkpoints(checkpoint_dir, start_epoch, end_epoch):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoints = []
    for path in checkpoint_dir.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch < start_epoch or (end_epoch and epoch > end_epoch):
            continue
        checkpoints.append((epoch, path))
    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise RuntimeError(
            f"No model_epoch_<N>.pth checkpoints found in {checkpoint_dir}"
        )
    return checkpoints


def load_state_dict(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return state_dict, checkpoint_epoch


def read_completed_results(path):
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path} at line {line_number}; "
                    "use --overwrite to restart the sweep"
                ) from error
    return records


def append_result(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_summary(path, args, results, world_size):
    ordered = sorted(results, key=lambda record: record["epoch"])
    best_psnr = max(ordered, key=lambda record: record["psnr"])
    best_ssim = max(ordered, key=lambda record: record["ssim"])
    summary = {
        "protocol": "LCDMoire 1024x1024 full-image RGB uint8 PSNR/SSIM",
        "data_root": str(Path(args.data_root).resolve()),
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "world_size": world_size,
        "amp": not args.no_amp,
        "images_per_checkpoint": ordered[0]["images"],
        "evaluated_checkpoints": len(ordered),
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "results": ordered,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(path)
    return summary


@torch.no_grad()
def main():
    args = parse_args()
    checkpoints = discover_checkpoints(
        args.checkpoint_dir, args.start_epoch, args.end_epoch
    )
    if args.list_only:
        for epoch, path in checkpoints:
            print(f"epoch={epoch} checkpoint={path}")
        print(f"checkpoints={len(checkpoints)}")
        return

    distributed, rank, world_size, _, device = setup_distributed()
    is_main = rank == 0
    amp_enabled = not args.no_amp and device.type == "cuda"
    output_dir = Path(args.output_dir).resolve()
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if results_path.exists() and not args.resume and not args.overwrite:
            raise FileExistsError(
                f"Results already exist: {results_path}; "
                "pass --resume or --overwrite"
            )
        if args.overwrite:
            results_path.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
    if distributed:
        dist.barrier()

    existing_results = read_completed_results(results_path) if args.resume else []
    completed_epochs = {int(record["epoch"]) for record in existing_results}

    dataset = get_lcdmoire_validation_data(args.data_root)
    if len(dataset) != 100:
        raise RuntimeError(f"Expected 100 LCDMoire validation pairs, found {len(dataset)}")
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
    model = build_uformer_b(image_size=1024).to(device).eval()

    if is_main:
        print(
            f"LCDMoire full-image sweep | checkpoints={len(checkpoints)} | "
            f"completed={len(completed_epochs)} | world_size={world_size} | "
            f"amp={amp_enabled}"
        )

    new_results = []
    for position, (filename_epoch, checkpoint_path) in enumerate(checkpoints, start=1):
        if filename_epoch in completed_epochs:
            if is_main:
                print(
                    f"[{position}/{len(checkpoints)}] skip epoch {filename_epoch} "
                    "(already recorded)"
                )
            continue
        if is_main:
            print(
                f"[{position}/{len(checkpoints)}] evaluating epoch "
                f"{filename_epoch}: {checkpoint_path.name}",
                flush=True,
            )
        state_dict, checkpoint_epoch = load_state_dict(checkpoint_path)
        if checkpoint_epoch is not None and int(checkpoint_epoch) != filename_epoch:
            raise RuntimeError(
                f"Epoch mismatch for {checkpoint_path}: filename={filename_epoch}, "
                f"checkpoint={checkpoint_epoch}"
            )
        model.load_state_dict(state_dict)
        del state_dict
        if distributed:
            dist.barrier()

        start_time = time.time()
        psnr, ssim, image_count = evaluate_full_resolution(
            model,
            loader,
            device,
            amp_enabled,
            distributed,
            tile_size=1024,
            tile_overlap=0,
            metric_mode="lcdmoire",
            max_images=args.max_images,
        )
        elapsed = time.time() - start_time
        record = {
            "epoch": filename_epoch,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint": str(checkpoint_path),
            "images": image_count,
            "psnr": psnr,
            "ssim": ssim,
            "seconds": elapsed,
        }
        if is_main:
            append_result(results_path, record)
            new_results.append(record)
            print(
                f"epoch={filename_epoch} images={image_count} "
                f"PSNR={psnr:.6f} SSIM={ssim:.8f} time={elapsed:.1f}s",
                flush=True,
            )
        if distributed:
            dist.barrier()

    if is_main:
        all_results = existing_results + new_results
        if not all_results:
            raise RuntimeError("No checkpoint results are available")
        summary = write_summary(summary_path, args, all_results, world_size)
        best_psnr = summary["best_psnr"]
        best_ssim = summary["best_ssim"]
        print(
            "BEST_PSNR "
            f"epoch={best_psnr['epoch']} PSNR={best_psnr['psnr']:.6f} "
            f"SSIM={best_psnr['ssim']:.8f}"
        )
        print(
            "BEST_SSIM "
            f"epoch={best_ssim['epoch']} PSNR={best_ssim['psnr']:.6f} "
            f"SSIM={best_ssim['ssim']:.8f}"
        )
        print(f"results={results_path}")
        print(f"summary={summary_path}")

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
