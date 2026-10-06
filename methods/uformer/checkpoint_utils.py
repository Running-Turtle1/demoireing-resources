from pathlib import Path

import torch
from safetensors.torch import load_file


def load_checkpoint(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        return {"state_dict": load_file(str(path), device="cpu"), "epoch": None}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalized_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint)
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict
