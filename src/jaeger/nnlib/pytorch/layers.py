from typing import Callable, Optional, Tuple

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
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
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
            self.activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
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

        b, f, length, c = x.shape

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
        x_2d = x.reshape(b * f, c, length)
        if self.padding == "same":
            pad_total = self._resolve_padding(length)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_2d = F.pad(x_2d, (pad_left, pad_right))

        out = self.conv(x_2d)
        _, _, length_out = out.shape
        out = out.reshape(b, f, length_out, self.filters)

        if self.activation is not None:
            out = self.activation(out)

        out_mask = None
        if mask is not None:
            mask_f = mask.reshape(b * f, 1, length).to(x.dtype)
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
            out_mask = out_mask.reshape(b, f, length_out)

        return out, out_mask


class MaskedBatchNorm(nn.Module):
    """Batch normalization that excludes masked positions from statistics.

    Can optionally return normalized mean difference (nmd) vectors.

    The running statistics are updated with Keras-style momentum:
    ``running = momentum * running + (1 - momentum) * batch``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.9,
        return_nmd: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.return_nmd = return_nmd

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        xf = x.to(torch.float32)
        b, f, length, c = xf.shape

        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(torch.float32)
            masked_x = xf * mask_f
            valid_count = mask_f.sum(dim=(0, 1, 2)) + self.eps
            mean = masked_x.sum(dim=(0, 1, 2)) / valid_count
            var = ((masked_x - mean) * mask_f).pow(2).sum(dim=(0, 1, 2)) / valid_count
        else:
            mean = xf.mean(dim=(0, 1, 2))
            var = xf.var(dim=(0, 1, 2), unbiased=False)

        if self.training:
            self.running_mean.lerp_(mean.detach(), 1 - self.momentum)
            self.running_var.lerp_(var.detach(), 1 - self.momentum)
            mean_use, var_use = mean, var
        else:
            mean_use, var_use = self.running_mean, self.running_var

        mean_use = mean_use.view(1, 1, 1, -1)
        var_use = var_use.view(1, 1, 1, -1)
        normalized = (xf - mean_use) / torch.sqrt(var_use + self.eps)
        out = normalized * self.gamma.view(1, 1, 1, -1) + self.beta.view(1, 1, 1, -1)
        out = out.to(x.dtype)

        if self.return_nmd:
            if mask is not None:
                per_ex_sum = masked_x.sum(dim=(1, 2))
                per_ex_count = mask_f.sum(dim=(1, 2)) + self.eps
                mean_channel = per_ex_sum / per_ex_count
            else:
                mean_channel = xf.mean(dim=(1, 2))
            nmd = (mean_channel - mean).to(x.dtype)
            return out, nmd

        return out, None


class MaskedLayerNorm(nn.Module):
    """Layer normalization that excludes masked positions.

    Normalization is performed over the channel dimension for each
    (batch, frame, position) tuple. Positions excluded by ``mask`` are
    omitted from the mean and variance, and the corresponding output is
    zeroed.
    """

    def __init__(self, num_features: int, eps: float = 1e-3):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = x.to(torch.float32)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(torch.float32)
            masked_x = xf * mask_f
            count = mask_f.sum(dim=-1, keepdim=True) * x.shape[-1]
            count = count.masked_fill(count == 0, self.eps)
            mean = masked_x.sum(dim=-1, keepdim=True) / count
            var = ((masked_x - mean) * mask_f).pow(2).sum(dim=-1, keepdim=True) / count
        else:
            mean = xf.mean(dim=-1, keepdim=True)
            var = xf.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (xf - mean) / torch.sqrt(var + self.eps)
        out = normalized * self.gamma.view(1, 1, 1, -1) + self.beta.view(1, 1, 1, -1)
        if mask is not None:
            out = out * mask_f
        return out.to(x.dtype)
