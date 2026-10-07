#!/usr/bin/env python3
import argparse
import hashlib
import shutil
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_id")
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()
    registry = yaml.safe_load((ROOT / "registry/checkpoints.yaml").read_text())
    entries = {entry["id"]: entry for entry in registry["checkpoints"]}
    if args.checkpoint_id not in entries:
        raise KeyError(f"unknown checkpoint: {args.checkpoint_id}")
    entry = entries[args.checkpoint_id]
    repository = entry.get("repository", registry.get("repository"))
    if not repository:
        raise RuntimeError("checkpoint repository is not published yet")
    source = hf_hub_download(repo_id=repository, filename=entry["path"])
    output = Path(args.output_dir) / entry["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    actual_hash = sha256(output)
    if actual_hash != entry["sha256"]:
        output.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {args.checkpoint_id}: {actual_hash}"
        )
    print(output)


if __name__ == "__main__":
    main()
