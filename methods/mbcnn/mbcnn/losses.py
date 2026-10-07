from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BatchSummedL1Loss(nn.Module):
    """L1 sum divided by batch size, matching the reference implementation."""

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.abs(prediction - target).sum() / prediction.shape[0]


class AdvancedSobelL1Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor(
            [
                [
                    [2, 1, 0],
                    [1, 0, -1],
                    [0, -1, -2],
                ]
            ]
            * 3,
            dtype=torch.float32,
        ).unsqueeze(0)
        kernel_y = torch.tensor(
            [
                [
                    [0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0],
                ]
            ]
            * 3,
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("kernel_x", kernel_x, persistent=False)
        self.register_buffer("kernel_y", kernel_y, persistent=False)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_edges = self._edges(prediction)
        target_edges = self._edges(target)
        return torch.abs(prediction_edges - target_edges).sum() / prediction.shape[0]

    def _edges(self, image: Tensor) -> Tensor:
        kernel_x = self.kernel_x.to(dtype=image.dtype)
        kernel_y = self.kernel_y.to(dtype=image.dtype)
        return torch.abs(F.conv2d(image, kernel_x)) + torch.abs(
            F.conv2d(image, kernel_y)
        )


class MBCNNLoss(nn.Module):
    """Reference MBCNN loss over quarter, half, and full-resolution outputs."""

    def __init__(self, edge_weight: float = 0.25) -> None:
        super().__init__()
        self.edge_weight = edge_weight
        self.criterion_l1 = BatchSummedL1Loss()
        self.criterion_advanced_sobel_l1 = AdvancedSobelL1Loss()

    def forward(
        self,
        output_quarter: Tensor,
        output_half: Tensor,
        output_full: Tensor,
        target: Tensor,
    ) -> Tensor:
        target_half = F.interpolate(
            target, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        target_quarter = F.interpolate(
            target, scale_factor=0.25, mode="bilinear", align_corners=False
        )

        loss_full = self._scale_loss(output_full, target)
        loss_half = self._scale_loss(output_half, target_half)
        loss_quarter = self._scale_loss(output_quarter, target_quarter)
        return loss_full + loss_half + loss_quarter

    def _scale_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        pixel_loss = self.criterion_l1(prediction, target)
        edge_loss = self.criterion_advanced_sobel_l1(prediction, target)
        return pixel_loss + self.edge_weight * edge_loss

