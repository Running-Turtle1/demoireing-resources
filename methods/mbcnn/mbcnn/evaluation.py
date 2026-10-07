from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ImagePadding:
    left: int
    right: int
    top: int
    bottom: int


def pad_esdnet_style(
    image: Tensor,
    *,
    multiple: int = 32,
    channel_values: tuple[float, float, float] = (0.3827, 0.4141, 0.3912),
) -> tuple[Tensor, ImagePadding]:
    """Pad an RGB NCHW batch exactly as UniDemoire's ESDNet evaluator."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected an NCHW RGB image batch")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if len(channel_values) != 3:
        raise ValueError("channel_values must contain three RGB values")

    height, width = image.shape[-2:]
    width_delta = (-width) % multiple
    height_delta = (-height) % multiple
    padding = ImagePadding(
        left=width_delta // 2,
        right=width_delta - width_delta // 2,
        top=height_delta // 2,
        bottom=height_delta - height_delta // 2,
    )
    torch_padding = (
        padding.left,
        padding.right,
        padding.top,
        padding.bottom,
    )
    channels = [
        F.pad(image[:, index : index + 1], torch_padding, value=float(value))
        for index, value in enumerate(channel_values)
    ]
    return torch.cat(channels, dim=1), padding


def remove_padding(image: Tensor, padding: ImagePadding) -> Tensor:
    height_end = image.shape[-2] - padding.bottom
    width_end = image.shape[-1] - padding.right
    return image[
        ...,
        padding.top:height_end,
        padding.left:width_end,
    ]


def save_rgb_tensor(image: Tensor, path: str | Path) -> None:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Only single-image batches can be saved")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a CHW RGB image")
    array = (
        image.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array), mode="RGB").save(output_path)

