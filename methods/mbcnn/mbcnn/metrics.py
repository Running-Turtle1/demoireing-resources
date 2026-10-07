from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def batch_psnr(prediction: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    prediction, target = _validate_images(prediction, target)
    mse = (prediction - target).square().flatten(1).mean(1)
    maximum = prediction.new_tensor(data_range).square()
    return 10.0 * torch.log10(maximum / mse.clamp_min(1e-12))


def batch_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    prediction, target = _validate_images(prediction, target)
    height, width = prediction.shape[-2:]
    window_size = min(window_size, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires images at least 3x3")

    coordinates = torch.arange(
        window_size,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    coordinates -= (window_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    channels = prediction.shape[1]
    window = window.expand(channels, 1, window_size, window_size)
    padding = window_size // 2

    mean_prediction = F.conv2d(prediction, window, padding=padding, groups=channels)
    mean_target = F.conv2d(target, window, padding=padding, groups=channels)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_cross = mean_prediction * mean_target

    variance_prediction = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mean_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mean_cross
    )

    constant1 = (0.01 * data_range) ** 2
    constant2 = (0.03 * data_range) ** 2
    numerator = (2 * mean_cross + constant1) * (2 * covariance + constant2)
    denominator = (mean_prediction_sq + mean_target_sq + constant1) * (
        variance_prediction + variance_target + constant2
    )
    return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(1)


def batch_uhdm_psnr(prediction: Tensor, target: Tensor) -> Tensor:
    """Match the full-RGB PSNR used by UniDemoire for UHDM."""

    prediction, target = _validate_images(prediction, target)
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (prediction - target).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse)


def batch_uhdm_ssim(
    prediction: Tensor,
    target: Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Match UniDemoire's valid-window PyTorch SSIM for UHDM."""

    prediction, target = _validate_images(prediction, target)
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    height, width = prediction.shape[-2:]
    real_size = min(window_size, height, width)
    if real_size <= 0:
        raise ValueError("SSIM requires non-empty images")

    coordinates = torch.arange(
        real_size,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    coordinates -= real_size // 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    window = torch.outer(gaussian, gaussian)
    channels = prediction.shape[1]
    window = window.expand(channels, 1, real_size, real_size).contiguous()

    mean_prediction = F.conv2d(prediction, window, groups=channels)
    mean_target = F.conv2d(target, window, groups=channels)
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_cross = mean_prediction * mean_target

    variance_prediction = (
        F.conv2d(prediction.square(), window, groups=channels)
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(target.square(), window, groups=channels) - mean_target_sq
    )
    covariance = (
        F.conv2d(prediction * target, window, groups=channels) - mean_cross
    )

    constant1 = 0.01**2
    constant2 = 0.03**2
    contrast_numerator = 2.0 * covariance + constant2
    contrast_denominator = variance_prediction + variance_target + constant2
    ssim_map = (
        (2.0 * mean_cross + constant1) * contrast_numerator
    ) / (
        (mean_prediction_sq + mean_target_sq + constant1)
        * contrast_denominator
    )
    return ssim_map.flatten(1).mean(1)


def _validate_images(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.ndim != 4:
        raise ValueError("Expected NCHW image batches")
    return prediction.float(), target.float()

