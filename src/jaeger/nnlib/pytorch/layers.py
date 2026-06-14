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
    Bias is added at all positions, so downstream code should apply the
    returned mask if masked positions must remain suppressed.

    For API compatibility with Keras-style configs, ``kernel_initializer``,
    ``bias_initializer`` and ``kernel_regularizer`` are accepted in the
    signature but only their default values are supported; passing a non-default
    value raises ``NotImplementedError``.
    """

    _SUPPORTED_ACTIVATIONS = {
        "relu": F.relu,
        "gelu": F.gelu,
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
    }

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
        padding_norm = padding.lower()
        if padding_norm not in ("same", "valid"):
            raise ValueError(
                f"padding must be 'same' or 'valid', got {padding!r}"
            )

        if kernel_initializer != "glorot_uniform":
            raise NotImplementedError(
                f"kernel_initializer {kernel_initializer!r} is not supported; "
                "only 'glorot_uniform' is implemented."
            )
        if bias_initializer != "zeros":
            raise NotImplementedError(
                f"bias_initializer {bias_initializer!r} is not supported; "
                "only 'zeros' is implemented."
            )
        if kernel_regularizer is not None:
            raise NotImplementedError(
                f"kernel_regularizer {kernel_regularizer!r} is not supported."
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding_norm
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias

        if activation is None:
            self.activation: Optional[callable] = None
        else:
            act_norm = activation.lower()
            if act_norm not in self._SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"activation must be one of "
                    f"{list(self._SUPPORTED_ACTIVATIONS.keys())} or None, "
                    f"got {activation!r}"
                )
            self.activation = self._SUPPORTED_ACTIVATIONS[act_norm]

        # Placeholder conv so that ``parameters()`` and ``state_dict()`` are
        # available before the first forward.  ``in_channels`` is set to the
        # real value on the first forward call.
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=0,  # handled manually
            dilation=dilation_rate,
            bias=use_bias,
        )
        self._in_channels: Optional[int] = None

    def _resolve_padding(self, length: int) -> int:
        if self.padding == "same":
            dilated = self.dilation_rate * (self.kernel_size - 1)
            return (
                (length + self.strides - 1) // self.strides * self.strides
                - length
                + dilated
            )
        return 0

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input of shape (B, F, L, C), got {x.ndim}D"
            )

        b, f, l, c = x.shape

        if self._in_channels is None:
            # First forward: create the real conv with the correct in_channels.
            self.conv = nn.Conv1d(
                in_channels=c,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding=0,
                dilation=self.dilation_rate,
                bias=self.use_bias,
            ).to(x.device)
            self._in_channels = c
        elif c != self._in_channels:
            raise ValueError(
                f"Expected input channels {self._in_channels}, got {c}. "
                "MaskedConv1D does not support variable channel dimensions."
            )

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

        if self.activation is not None:
            out = self.activation(out)

        out_mask = None
        if mask is not None:
            mask_f = mask.reshape(b * f, 1, l).to(x.dtype)
            if self.padding == "same":
                mask_f = F.pad(mask_f, (pad_left, pad_right))
            with torch.no_grad():
                kernel = torch.ones(
                    (1, 1, self.kernel_size),
                    dtype=mask_f.dtype,
                    device=mask_f.device,
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
