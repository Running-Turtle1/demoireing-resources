from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImagePair:
    sample_id: str
    input_path: Path
    target_path: Path


def discover_image_pairs(
    root: str | Path,
    *,
    input_suffix: str = "_moire.jpg",
    target_suffix: str = "_gt.jpg",
) -> list[ImagePair]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not input_suffix or not target_suffix:
        raise ValueError("Input and target suffixes must be non-empty")

    targets = sorted(root.rglob(f"*{target_suffix}"))
    if not targets:
        raise FileNotFoundError(
            f"No target images ending with {target_suffix!r} were found under {root}"
        )

    pairs: list[ImagePair] = []
    missing: list[Path] = []
    for target_path in targets:
        stem = target_path.name[: -len(target_suffix)]
        input_path = target_path.with_name(f"{stem}{input_suffix}")
        if not input_path.is_file():
            missing.append(input_path)
            continue
        relative_id = target_path.relative_to(root).as_posix()
        sample_id = relative_id[: -len(target_suffix)]
        pairs.append(ImagePair(sample_id, input_path, target_path))

    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} paired input image(s). Examples:\n{examples}"
        )
    return pairs


class PairedImageDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        root: str | Path,
        *,
        input_suffix: str = "_moire.jpg",
        target_suffix: str = "_gt.jpg",
        crop_size: int | None = None,
        random_crop: bool = False,
        augment: bool = False,
    ) -> None:
        if crop_size is not None:
            if crop_size <= 0:
                raise ValueError("crop_size must be positive or None")
            if crop_size % 8:
                raise ValueError("crop_size must be divisible by 8")
        if augment and not random_crop:
            raise ValueError("augment=True is only supported for random training crops")

        self.root = Path(root).expanduser().resolve()
        self.pairs = discover_image_pairs(
            self.root,
            input_suffix=input_suffix,
            target_suffix=target_suffix,
        )
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        pair = self.pairs[index]
        input_image = _load_rgb(pair.input_path)
        target_image = _load_rgb(pair.target_path)
        if input_image.size != target_image.size:
            raise ValueError(
                f"Pair {pair.sample_id!r} has mismatched sizes: "
                f"input={input_image.size}, target={target_image.size}"
            )

        if self.crop_size is not None:
            input_image, target_image = self._paired_crop(input_image, target_image)

        input_tensor = _image_to_tensor(input_image)
        target_tensor = _image_to_tensor(target_image)
        if self.augment:
            input_tensor, target_tensor = _paired_augment(input_tensor, target_tensor)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "id": pair.sample_id,
            "input_path": str(pair.input_path),
            "target_path": str(pair.target_path),
        }

    def _paired_crop(
        self,
        input_image: Image.Image,
        target_image: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        assert self.crop_size is not None
        width, height = input_image.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"Image size {width}x{height} is smaller than crop size {self.crop_size}"
            )

        if self.random_crop:
            left = int(torch.randint(width - self.crop_size + 1, ()).item())
            top = int(torch.randint(height - self.crop_size + 1, ()).item())
        else:
            left = (width - self.crop_size) // 2
            top = (height - self.crop_size) // 2
        box = (left, top, left + self.crop_size, top + self.crop_size)
        return input_image.crop(box), target_image.crop(box)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _image_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _paired_augment(input_image: Tensor, target_image: Tensor) -> tuple[Tensor, Tensor]:
    if torch.rand(()) < 0.5:
        input_image = torch.flip(input_image, dims=(2,))
        target_image = torch.flip(target_image, dims=(2,))
    if torch.rand(()) < 0.5:
        input_image = torch.flip(input_image, dims=(1,))
        target_image = torch.flip(target_image, dims=(1,))
    rotations = int(torch.randint(4, ()).item())
    if rotations:
        input_image = torch.rot90(input_image, rotations, dims=(1, 2))
        target_image = torch.rot90(target_image, rotations, dims=(1, 2))
    return input_image.contiguous(), target_image.contiguous()

