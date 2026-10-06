#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from checkpoint_utils import load_checkpoint, normalized_state_dict  # noqa: E402
from train.train_demoire_uhdm import build_uformer_b  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    model = build_uformer_b(image_size=128).eval()
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        model.load_state_dict(normalized_state_dict(checkpoint), strict=True)
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 128, 256))
    if output.shape != (1, 3, 128, 256) or not torch.isfinite(output).all():
        raise RuntimeError("Uformer-B rectangular smoke test failed")
    print(f"ok: output_shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
