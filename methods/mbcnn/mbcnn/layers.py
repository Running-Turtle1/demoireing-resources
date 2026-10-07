from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class USConv2d(nn.Conv2d):
    """The fixed-width convolution used by the PyTorch MBCNN port.

    UniDemoire embeds MBCNN in a universally slimmable network utility. MBCNN
    itself always runs at width 1.0, where that layer is exactly ``Conv2d``.
    The extra constructor arguments are retained so upstream parameter names
    and module structure stay compatible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        us: tuple[bool, bool] | list[bool] = (False, False),
        cat_shape: tuple[int, ...] | list[int] | None = None,
        set_cat_mul: tuple[bool, ...] | list[bool] | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.width_mult = 1.0
        self.us = tuple(us)
        self.cat_shape = tuple(cat_shape) if cat_shape is not None else None
        self.set_cat_mul = tuple(set_cat_mul) if set_cat_mul is not None else None
        self.in_channels_index_list = [None] * 4
        self.unrank = False

    def forward(self, inputs: Tensor) -> Tensor:
        if self.width_mult != 1.0:
            raise RuntimeError("Standalone MBCNN only supports width_mult=1.0")
        return F.conv2d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

