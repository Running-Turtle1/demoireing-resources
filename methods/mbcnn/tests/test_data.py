from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mbcnn.data import PairedImageDataset, discover_image_pairs
from mbcnn.metrics import batch_psnr, batch_ssim


def write_pair(root: Path, sample_id: str, size: int = 24) -> None:
    values = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    Image.fromarray(values).save(root / f"{sample_id}_moire.jpg", quality=100)
    Image.fromarray(values).save(root / f"{sample_id}_gt.jpg", quality=100)


def test_discover_and_load_aligned_pair(tmp_path: Path) -> None:
    nested = tmp_path / "pair_00"
    nested.mkdir()
    write_pair(nested, "0001")

    pairs = discover_image_pairs(tmp_path)
    dataset = PairedImageDataset(
        tmp_path,
        crop_size=16,
        random_crop=True,
        augment=True,
    )
    sample = dataset[0]

    assert len(pairs) == 1
    assert pairs[0].sample_id == "pair_00/0001"
    assert sample["input"].shape == (3, 16, 16)
    torch.testing.assert_close(sample["input"], sample["target"])


def test_missing_pair_is_reported(tmp_path: Path) -> None:
    Image.new("RGB", (16, 16)).save(tmp_path / "0001_gt.jpg")
    with pytest.raises(FileNotFoundError, match="Missing 1 paired"):
        discover_image_pairs(tmp_path)


def test_perfect_metrics() -> None:
    image = torch.rand(2, 3, 16, 16)
    assert torch.all(batch_psnr(image, image) >= 100)
    torch.testing.assert_close(batch_ssim(image, image), torch.ones(2))

