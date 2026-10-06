"""FHDMi paired dataset with the preprocessing used by the official loader."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class FHDMiDataset(Dataset):
    """Load aligned ``src_XXXXX.png``/``tar_XXXXX.png`` FHDMi pairs.

    Training follows the official FHDMi loader: one aligned random crop from
    the original 1920x1080 pair, without resizing, flips, or rotations.
    Evaluation returns the complete full-resolution pair.
    """

    def __init__(self, split_dir, train=False, crop_size=512):
        self.split_dir = Path(split_dir)
        self.train = train
        self.crop_size = int(crop_size)
        if self.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"FHDMi split not found: {self.split_dir}")

        source_dir = self.split_dir / "source"
        target_dir = self.split_dir / "target"
        if not source_dir.is_dir() or not target_dir.is_dir():
            raise FileNotFoundError(
                f"Expected source/ and target/ under {self.split_dir}"
            )

        target_paths = sorted(target_dir.glob("tar_*.png"))
        if not target_paths:
            raise RuntimeError(f"No tar_*.png files found in {target_dir}")

        pairs = []
        missing_sources = []
        for target_path in target_paths:
            sample_id = target_path.stem[len("tar_") :]
            source_path = source_dir / f"src_{sample_id}.png"
            if not source_path.is_file():
                missing_sources.append(str(source_path))
            else:
                pairs.append((source_path, target_path, sample_id))
        if missing_sources:
            raise RuntimeError(
                f"Missing {len(missing_sources)} FHDMi source images; "
                f"examples: {missing_sources[:3]}"
            )

        source_paths = sorted(source_dir.glob("src_*.png"))
        expected_sources = {source for source, _, _ in pairs}
        extra_sources = [str(path) for path in source_paths if path not in expected_sources]
        if extra_sources:
            raise RuntimeError(
                f"Found {len(extra_sources)} unpaired FHDMi source images; "
                f"examples: {extra_sources[:3]}"
            )
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        source_path, target_path, sample_id = self.pairs[index]
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(target_path) as image:
            target = image.convert("RGB")
        if source.size != target.size:
            raise RuntimeError(
                f"Pair size mismatch: {source_path} {source.size} != "
                f"{target_path} {target.size}"
            )

        if self.train:
            width, height = target.size
            if width < self.crop_size or height < self.crop_size:
                raise RuntimeError(
                    f"Image {target_path} with size {target.size} is smaller than "
                    f"crop size {self.crop_size}"
                )
            left = random.randint(0, width - self.crop_size)
            top = random.randint(0, height - self.crop_size)
            crop_box = (left, top, left + self.crop_size, top + self.crop_size)
            source = source.crop(crop_box)
            target = target.crop(crop_box)

        return _pil_to_tensor(target), _pil_to_tensor(source), sample_id


def get_fhdmi_training_data(data_root, crop_size=512):
    return FHDMiDataset(Path(data_root) / "train", train=True, crop_size=crop_size)


def get_fhdmi_validation_data(data_root):
    return FHDMiDataset(Path(data_root) / "test", train=False, crop_size=512)
