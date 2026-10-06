#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Export a training checkpoint as inference-only safetensors")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    tensors = {}
    for name, tensor in state_dict.items():
        clean_name = name[7:] if name.startswith("module.") else name
        tensors[clean_name] = tensor.detach().cpu().contiguous()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata={"architecture": "Uformer-B", "dataset": args.dataset})
    metadata = {
        "architecture": "Uformer-B",
        "dataset": args.dataset,
        "selection": args.selection,
        "epoch": checkpoint.get("epoch"),
        "source_checkpoint": source.name,
        "source_sha256": sha256(source),
        "sha256": sha256(output),
        "parameters": sum(tensor.numel() for tensor in tensors.values()),
        "format": "safetensors",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
