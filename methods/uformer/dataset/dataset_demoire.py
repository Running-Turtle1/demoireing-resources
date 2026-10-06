"""TIP2018 paired dataset with the preprocessing used by the UHDM repository."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9.1
    _BILINEAR = Image.BILINEAR


def _pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class TIP2018Dataset(Dataset):
    """Load aligned ``source``/``target`` PNG pairs from one TIP2018 split.

    The geometric preprocessing intentionally follows
    ``UHDM/dataset/load_data.py::tip_data_loader``: crop the central region
    bounded by 1/6 and 5/6 of the image, add a paired +/-6 pixel translation
    during training, and bilinearly resize to 256x256.
    """

    def __init__(self, split_dir, train=False, augment=False, output_size=256):
        self.split_dir = Path(split_dir)
        self.source_dir = self.split_dir / "source"
        self.target_dir = self.split_dir / "target"
        self.train = train
        self.augment = augment
        self.output_size = output_size

        if not self.source_dir.is_dir() or not self.target_dir.is_dir():
            raise FileNotFoundError(
                f"Expected source/ and target/ under {self.split_dir}"
            )

        source_paths = sorted(self.source_dir.glob("*_source.png"))
        if not source_paths:
            raise RuntimeError(f"No *_source.png images found in {self.source_dir}")

        target_paths = sorted(self.target_dir.glob("*_target.png"))
        target_by_name = {path.name: path for path in target_paths}
        pairs = []
        missing_targets = []
        expected_target_names = set()
        for source_path in source_paths:
            target_name = source_path.name.replace("_source.png", "_target.png")
            expected_target_names.add(target_name)
            target_path = target_by_name.get(target_name)
            if target_path is None:
                missing_targets.append(target_name)
            else:
                pairs.append((source_path, target_path))

        extra_targets = sorted(set(target_by_name) - expected_target_names)
        if missing_targets or extra_targets:
            details = []
            if missing_targets:
                details.append(
                    f"missing targets ({len(missing_targets)}): {missing_targets[:3]}"
                )
            if extra_targets:
                details.append(
                    f"unpaired targets ({len(extra_targets)}): {extra_targets[:3]}"
                )
            raise RuntimeError("Invalid TIP2018 pairing: " + "; ".join(details))

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _crop_box(self, width, height):
        offset_x = random.randint(-6, 6) if self.train else 0
        offset_y = random.randint(-6, 6) if self.train else 0
        return (
            int(width / 6) + offset_x,
            int(height / 6) + offset_y,
            int(width * 5 / 6) + offset_x,
            int(height * 5 / 6) + offset_y,
        )

    @staticmethod
    def _paired_augment(target, source):
        transform_id = random.randrange(8)
        rotation = transform_id % 4
        if rotation:
            target = torch.rot90(target, rotation, dims=(-2, -1))
            source = torch.rot90(source, rotation, dims=(-2, -1))
        if transform_id >= 4:
            target = torch.flip(target, dims=(-1,))
            source = torch.flip(source, dims=(-1,))
        return target, source

    def __getitem__(self, index):
        source_path, target_path = self.pairs[index]
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(target_path) as image:
            target = image.convert("RGB")

        if source.size != target.size:
            raise RuntimeError(
                f"Pair size mismatch: {source_path} {source.size} != "
                f"{target_path} {target.size}"
            )

        crop_box = self._crop_box(*target.size)
        resize_shape = (self.output_size, self.output_size)
        source = source.crop(crop_box).resize(resize_shape, _BILINEAR)
        target = target.crop(crop_box).resize(resize_shape, _BILINEAR)
        source = _pil_to_tensor(source)
        target = _pil_to_tensor(target)

        if self.train and self.augment:
            target, source = self._paired_augment(target, source)

        sample_name = source_path.name[: -len("_source.png")]
        return target, source, sample_name


def get_tip2018_training_data(data_root, augment=True):
    return TIP2018Dataset(
        Path(data_root) / "trainData", train=True, augment=augment
    )


def get_tip2018_validation_data(data_root):
    return TIP2018Dataset(
        Path(data_root) / "testData", train=False, augment=False
    )
