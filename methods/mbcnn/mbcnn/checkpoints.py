from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

Checkpoint = str | PathLike[str] | Mapping[str, Any]


def load_model_checkpoint(
    model: nn.Module,
    checkpoint: Checkpoint,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.modules.module._IncompatibleKeys:
    """Load a plain, DataParallel, or UniDemoire Lightning checkpoint."""

    payload = _load_payload(checkpoint, map_location)
    raw_state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(raw_state, Mapping):
        raise TypeError("Checkpoint must contain a state_dict mapping")

    state = {
        str(key): value for key, value in raw_state.items() if isinstance(value, Tensor)
    }
    target_keys = set(model.state_dict())
    prefixes = ("", "module.", "model.", "module.model.")

    candidates: list[dict[str, Tensor]] = []
    for prefix in prefixes:
        candidate = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if not prefix or key.startswith(prefix)
        }
        candidates.append(candidate)

    best = max(
        candidates, key=lambda candidate: len(target_keys.intersection(candidate))
    )
    model_state = {key: value for key, value in best.items() if key in target_keys}
    if not model_state:
        raise ValueError("No MBCNN parameters were found in the checkpoint")
    return model.load_state_dict(model_state, strict=strict)


def _load_payload(
    checkpoint: Checkpoint,
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        return checkpoint
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "Loading safetensors requires: python -m pip install safetensors"
            ) from error
        return load_file(checkpoint_path, device=str(map_location))
    try:
        payload = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint file must contain a mapping")
    return payload
