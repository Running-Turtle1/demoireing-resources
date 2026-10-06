#!/usr/bin/env python3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT / "registry" / name).open() as handle:
        return yaml.safe_load(handle)


def require_unique(entries, label):
    identifiers = [entry["id"] for entry in entries]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"duplicate {label} IDs")


def main():
    methods = load("methods.yaml")["methods"]
    datasets = load("datasets.yaml")["datasets"]
    checkpoints = load("checkpoints.yaml")["checkpoints"]
    require_unique(methods, "method")
    require_unique(datasets, "dataset")
    require_unique(checkpoints, "checkpoint")
    method_ids = {entry["id"] for entry in methods}
    dataset_ids = {entry["id"] for entry in datasets}
    for method in methods:
        if not (ROOT / method["path"]).is_dir():
            raise RuntimeError(f"missing method path: {method['path']}")
    for dataset in datasets:
        if not (ROOT / dataset["card"]).is_file():
            raise RuntimeError(f"missing dataset card: {dataset['card']}")
    for checkpoint in checkpoints:
        if checkpoint["method"] not in method_ids:
            raise RuntimeError(f"unknown method: {checkpoint['method']}")
        if checkpoint["dataset"] not in dataset_ids:
            raise RuntimeError(f"unknown dataset: {checkpoint['dataset']}")
    print(f"validated {len(methods)} methods, {len(datasets)} datasets, {len(checkpoints)} checkpoints")


if __name__ == "__main__":
    main()
