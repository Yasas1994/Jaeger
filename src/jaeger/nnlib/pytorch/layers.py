import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeLU(nn.Module):
    """Tanh-approximated GELU for TFLite-compatible graph export."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate="tanh")


class MaskedConv1D(nn.Module):
    """1D convolution over the sequence axis of (B, F, L, C) inputs.

    Masked positions are zeroed before convolution and the output mask is
    propagated based on whether the full kernel window contained valid values.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "valid",
        dilation_rate: int = 1,
        activation: Optional[str] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[float] = None,
    ):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.lower()
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels=filters,  # placeholder; set in build
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=0,  # handled manually
            dilation=dilation_rate,
            bias=use_bias,
        )
        # Actual in_channels will be determined at first forward; override then.

    def _resolve_padding(self, length: int) -> int:
        if self.padding == "same":
            dilated = self.dilation_rate * (self.kernel_size - 1)
            return (length + self.strides - 1) // self.strides * self.strides - length + dilated
        return 0

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, f, l, c = x.shape
        # Lazy set conv in_channels on first call
        if self.conv.in_channels != c:
            self.conv = nn.Conv1d(
                in_channels=c,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding=0,
                dilation=self.dilation_rate,
                bias=self.use_bias,
            ).to(x.device)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        # Merge batch and frame dims: (B*F, C, L)
        x_2d = x.reshape(b * f, c, l)
        if self.padding == "same":
            pad_total = self._resolve_padding(l)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_2d = F.pad(x_2d, (pad_left, pad_right))

        out = self.conv(x_2d)
        _, _, l_out = out.shape
        out = out.reshape(b, f, l_out, self.filters)

        if self.activation:
            out = getattr(F, self.activation)(out)

        out_mask = None
        if mask is not None:
            mask_f = mask.reshape(b * f, 1, l).to(x.dtype)
            if self.padding == "same":
                mask_f = F.pad(mask_f, (pad_left, pad_right))
            with torch.no_grad():
                kernel = torch.ones(
                    (1, 1, self.kernel_size), dtype=mask_f.dtype, device=mask_f.device
                )
                out_mask = F.conv1d(
                    mask_f,
                    kernel,
                    stride=self.strides,
                    dilation=self.dilation_rate,
                )
                out_mask = (out_mask >= self.kernel_size).squeeze(1)
            out_mask = out_mask.reshape(b, f, l_out)

        return out, out_mask
