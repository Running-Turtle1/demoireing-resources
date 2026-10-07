import pytest
import torch

from mbcnn.evaluation import pad_esdnet_style, remove_padding
from mbcnn.metrics import batch_uhdm_psnr, batch_uhdm_ssim


def test_esdnet_padding_and_removal() -> None:
    image = torch.rand(1, 3, 15, 17)

    padded, padding = pad_esdnet_style(image)

    assert padded.shape == (1, 3, 32, 32)
    assert padding.left == 7
    assert padding.right == 8
    assert padding.top == 8
    assert padding.bottom == 9
    torch.testing.assert_close(remove_padding(padded, padding), image)
    expected_values = (0.3827, 0.4141, 0.3912)
    for channel, value in enumerate(expected_values):
        assert padded[0, channel, 0, 0].item() == pytest.approx(value)


def test_esdnet_padding_is_noop_at_multiple() -> None:
    image = torch.rand(1, 3, 32, 64)

    padded, padding = pad_esdnet_style(image)

    torch.testing.assert_close(padded, image)
    assert (padding.left, padding.right, padding.top, padding.bottom) == (0, 0, 0, 0)


def test_uhdm_metrics_are_perfect_for_identical_images() -> None:
    image = torch.rand(2, 3, 24, 32)

    psnr = batch_uhdm_psnr(image, image)
    ssim = batch_uhdm_ssim(image, image)

    assert torch.isinf(psnr).all()
    torch.testing.assert_close(ssim, torch.ones_like(ssim))

