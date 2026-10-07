from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import Conv, ConvReLU, GlobalBlock, PostBlock, PreBlock


class MBCNN(nn.Module):
    """Multi-scale bandpass convolutional neural network for demoireing.

    The model returns predictions from coarse to fine: quarter resolution,
    half resolution, and full resolution.
    """

    def __init__(
        self,
        n_filters: int = 64,
        *,
        nFilters: int | None = None,
    ) -> None:
        super().__init__()
        if nFilters is not None:
            if n_filters != 64 and n_filters != nFilters:
                raise ValueError("n_filters and nFilters disagree")
            n_filters = nFilters
        if n_filters != 64:
            raise ValueError("This MBCNN architecture is fixed to n_filters=64")

        self.imagesize = 256
        self.sigmoid = nn.Sigmoid()
        self.Space2Depth1 = nn.PixelUnshuffle(2)
        self.Depth2space1 = nn.PixelShuffle(2)

        self.conv_func1 = ConvReLU(12, n_filters * 2, 3, padding=1)
        self.pre_block1 = PreBlock((1, 2, 3, 2, 1))
        self.conv_func2 = ConvReLU(128, n_filters * 2, 3, padding=0, stride=2)
        self.pre_block2 = PreBlock((1, 2, 3, 2, 1))

        self.conv_func3 = ConvReLU(128, n_filters * 2, 3, padding=0, stride=2)
        self.pre_block3 = PreBlock((1, 2, 2, 2, 1))
        self.global_block1 = GlobalBlock(self.imagesize // 8)
        self.pos_block1 = PostBlock((1, 2, 2, 2, 1))
        self.conv1 = Conv(128, 12, 3, us=(True, False))

        self.conv_func4 = ConvReLU(
            131,
            n_filters * 2,
            1,
            padding=0,
            cat_shape=(3, n_filters * 2),
            set_cat_mul=(False, True),
        )
        self.global_block2 = GlobalBlock(self.imagesize // 4)
        self.pre_block4 = PreBlock((1, 2, 3, 2, 1))
        self.global_block3 = GlobalBlock(self.imagesize // 4)
        self.pos_block2 = PostBlock((1, 2, 3, 2, 1))
        self.conv2 = Conv(128, 12, 3, us=(True, False))

        self.conv_func5 = ConvReLU(
            131,
            n_filters * 2,
            1,
            padding=0,
            cat_shape=(3, n_filters * 2),
            set_cat_mul=(False, True),
        )
        self.global_block4 = GlobalBlock(self.imagesize // 2)
        self.pre_block5 = PreBlock((1, 2, 3, 2, 1))
        self.global_block5 = GlobalBlock(self.imagesize // 2)
        self.pos_block3 = PostBlock((1, 2, 3, 2, 1))
        self.conv3 = Conv(128, 12, 3, us=(True, False))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_input(inputs)

        level1 = self.Space2Depth1(inputs)
        level1 = self.pre_block1(self.conv_func1(level1))

        level2 = self.conv_func2(F.pad(level1, (1, 1, 1, 1)))
        level2 = self.pre_block2(level2)

        level3 = self.conv_func3(F.pad(level2, (1, 1, 1, 1)))
        level3 = self.pre_block3(level3)
        level3 = self.global_block1(level3)
        level3 = self.pos_block1(level3)
        output_quarter = torch.sigmoid(self.Depth2space1(self.conv1(level3)))

        level2 = torch.cat((output_quarter, level2), dim=1)
        level2 = self.conv_func4(level2)
        level2 = self.global_block2(level2)
        level2 = self.pre_block4(level2)
        level2 = self.global_block3(level2)
        level2 = self.pos_block2(level2)
        output_half = torch.sigmoid(self.Depth2space1(self.conv2(level2)))

        level1 = torch.cat((level1, output_half), dim=1)
        level1 = self.conv_func5(level1)
        level1 = self.global_block4(level1)
        level1 = self.pre_block5(level1)
        level1 = self.global_block5(level1)
        level1 = self.pos_block3(level1)
        output_full = self.Depth2space1(self.conv3(level1))
        output_full = self.sigmoid(output_full) + output_full.new_tensor(1e-10)

        return output_quarter, output_half, output_full

    @staticmethod
    def _validate_input(inputs: Tensor) -> None:
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected an NCHW tensor, got shape {tuple(inputs.shape)}"
            )
        if inputs.shape[1] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {inputs.shape[1]}")
        height, width = inputs.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                f"Input height and width must be divisible by 8, got {height}x{width}"
            )

