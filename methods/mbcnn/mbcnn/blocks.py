from __future__ import annotations

from math import cos, pi, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import USConv2d


class ScaleLayer2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = nn.Parameter(torch.ones((64, 1, 1, 1)))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.ReLU(inputs * self.it_weights)


class Kernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel = torch.zeros((64, 64, 1, 1))
        low_frequency_scale = sqrt(1.0 / 8)
        frequency_scale = sqrt(2.0 / 8)

        for frequency_y in range(8):
            odd_y = 2 * frequency_y + 1
            for frequency_x in range(8):
                odd_x = 2 * frequency_x + 1
                frequency_index = frequency_y * 8 + frequency_x
                for spatial_y in range(8):
                    for spatial_x in range(8):
                        spatial_index = spatial_y * 8 + spatial_x
                        value = cos(odd_y * spatial_y * pi / 16)
                        value *= cos(odd_x * spatial_x * pi / 16)
                        value *= (
                            low_frequency_scale if spatial_y == 0 else frequency_scale
                        )
                        value *= (
                            low_frequency_scale if spatial_x == 0 else frequency_scale
                        )
                        kernel[spatial_index, frequency_index, 0, 0] = value

        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self) -> Tensor:
        return self.kernel


class AdaptiveImplicitTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ReLU = nn.ReLU()
        self.it_weights = ScaleLayer2()
        self.kernel = Kernel()

    def forward(self, inputs: Tensor) -> Tensor:
        kernel = self.it_weights(self.kernel())
        return F.conv2d(inputs, kernel)


class ScaleLayer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.tensor(scale))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.kernel


class ConvReLU(nn.Module):
    def __init__(
        self,
        channels: int,
        filters: int,
        kernel: int,
        padding: int = 1,
        stride: int = 1,
        set_cat_mul: tuple[bool, ...] | None = None,
        us: tuple[bool, bool] = (True, True),
        cat_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.channel = channels
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv = USConv2d(
            channels,
            filters,
            kernel,
            stride=stride,
            padding=padding,
            us=us,
            cat_shape=cat_shape,
            set_cat_mul=set_cat_mul,
        )
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(self.conv(inputs))


class BlockConvReLU(nn.Module):
    def __init__(
        self,
        channels: int,
        filters: int,
        kernel: int,
        padding: int = 1,
        stride: int = 1,
        dilation: int = 1,
        us: tuple[bool, bool] = (True, True),
        cat_shape: tuple[int, ...] | list[int] | None = None,
        set_cat_mul: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        self.channel = channels
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.relu = nn.ReLU()

        if kernel == 1:
            actual_padding = 0
        elif stride == 2:
            actual_padding = 1
        else:
            actual_padding = dilation

        self.conv = USConv2d(
            channels,
            filters,
            kernel,
            stride=stride,
            padding=actual_padding,
            dilation=dilation,
            us=us,
            cat_shape=cat_shape,
            set_cat_mul=set_cat_mul,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(self.conv(inputs))


class Conv(nn.Module):
    def __init__(
        self,
        channels: int,
        filters: int,
        kernel: int,
        padding: int = 1,
        stride: int = 1,
        us: tuple[bool, bool] = (True, True),
        cat_shape: tuple[int, ...] | list[int] | None = None,
        set_cat_mul: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        self.channel = channels
        self.filters = filters
        self.kernel = kernel
        self.padding = padding
        self.conv = USConv2d(
            channels,
            filters,
            kernel,
            stride=stride,
            padding=padding,
            us=us,
            cat_shape=cat_shape,
            set_cat_mul=set_cat_mul,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PreBlock(nn.Module):
    def __init__(self, dilation: tuple[int, int, int, int, int]) -> None:
        super().__init__()
        self.nFilters = 64
        self.dilation = dilation
        self.add = True

        cat_shape = [128]
        self.conv_relu1 = BlockConvReLU(
            128, 64, 3, dilation=dilation[0], cat_shape=cat_shape
        )
        cat_shape2 = [64] + cat_shape
        self.conv_relu2 = BlockConvReLU(
            192, 64, 3, dilation=dilation[1], cat_shape=cat_shape2
        )
        cat_shape3 = [64] + cat_shape2
        self.conv_relu3 = BlockConvReLU(
            256, 64, 3, dilation=dilation[2], cat_shape=cat_shape3
        )
        cat_shape4 = [64] + cat_shape3
        self.conv_relu4 = BlockConvReLU(
            320, 64, 3, dilation=dilation[3], cat_shape=cat_shape4
        )
        cat_shape5 = [64] + cat_shape4
        self.conv_relu5 = BlockConvReLU(
            384, 64, 3, dilation=dilation[4], cat_shape=cat_shape5
        )

        cat_shape6 = [64] + cat_shape5
        self.conv1 = Conv(448, 64, 3, us=(True, False), cat_shape=cat_shape6)
        self.conv2 = Conv(64, 128, 1, padding=0, us=(False, True))
        self.relu = nn.ReLU()
        self.adaptive_implicit_trans1 = AdaptiveImplicitTransform()
        self.ScaleLayer1 = ScaleLayer(0.1)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        features = inputs
        for layer in (
            self.conv_relu1,
            self.conv_relu2,
            self.conv_relu3,
            self.conv_relu4,
            self.conv_relu5,
        ):
            output = layer(features)
            features = torch.cat((output, features), dim=1)

        features = self.conv1(features)
        features = self.adaptive_implicit_trans1(features)
        features = self.conv2(features)
        features = self.ScaleLayer1(features)
        return residual + features


class GlobalBlock(nn.Module):
    def __init__(self, nominal_size: int) -> None:
        super().__init__()
        self.size = (nominal_size + 2) // 2
        self.avgkernel_size = nominal_size
        self.nFilters = 64
        self.conv_func1 = BlockConvReLU(128, 256, 3, stride=2, us=(True, False))
        self.GlobalAveragePooling2D = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(256, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.conv_func2 = BlockConvReLU(128, 256, 1, padding=0, us=(True, False))
        self.conv_func3 = BlockConvReLU(256, 128, 1, padding=0, us=(False, True))
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        context = F.pad(inputs, (1, 1, 1, 1))
        context = self.conv_func1(context)
        context = self.GlobalAveragePooling2D(context).flatten(1)
        context = self.relu(self.dense1(context))
        context = self.relu(self.dense2(context))
        context = self.dense3(context).unsqueeze(2).unsqueeze(3)

        features = self.conv_func2(inputs)
        features = features * context
        return self.conv_func3(features)


class PostBlock(nn.Module):
    def __init__(self, dilation: tuple[int, int, int, int, int]) -> None:
        super().__init__()
        self.nFilters = 64
        self.dilation = dilation

        cat_shape = [128]
        self.conv_func1 = BlockConvReLU(
            128, 64, 3, dilation=dilation[0], cat_shape=cat_shape
        )
        cat_shape2 = [64] + cat_shape
        self.conv_func2 = BlockConvReLU(
            192, 64, 3, dilation=dilation[1], cat_shape=cat_shape2
        )
        cat_shape3 = [64] + cat_shape2
        self.conv_func3 = BlockConvReLU(
            256, 64, 3, dilation=dilation[2], cat_shape=cat_shape3
        )
        cat_shape4 = [64] + cat_shape3
        self.conv_func4 = BlockConvReLU(
            320, 64, 3, dilation=dilation[3], cat_shape=cat_shape4
        )
        cat_shape5 = [64] + cat_shape4
        self.conv_func5 = BlockConvReLU(
            384, 64, 3, dilation=dilation[4], cat_shape=cat_shape5
        )
        cat_shape6 = [64] + cat_shape5
        self.conv_func_last = BlockConvReLU(
            448, 128, 1, padding=0, cat_shape=cat_shape6
        )

    def forward(self, inputs: Tensor) -> Tensor:
        features = inputs
        for layer in (
            self.conv_func1,
            self.conv_func2,
            self.conv_func3,
            self.conv_func4,
            self.conv_func5,
        ):
            output = layer(features)
            features = torch.cat((output, features), dim=1)
        return self.conv_func_last(features)

