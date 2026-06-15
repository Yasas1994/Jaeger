import math
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        kernel_regularizer: Optional[str] = None,
        kernel_regularizer_w: float = 0.0,
        bias_regularizer: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        padding_norm = padding.lower()
        if padding_norm not in ("same", "valid"):
            raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")

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

        # Regularizers are accepted for Keras config compatibility but are not
        # implemented at the layer level; use optimizer weight_decay instead.
        self.kernel_regularizer = kernel_regularizer
        self.kernel_regularizer_w = kernel_regularizer_w
        self.bias_regularizer = bias_regularizer

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
            raise ValueError(f"Expected 4D input of shape (B, F, L, C), got {x.ndim}D")

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

        # Merge batch and frame dims. The input layout is (B, F, L, C); Conv1d
        # expects (N, C, L), so permute before reshaping to keep channels and
        # length in the right order.
        x_2d = x.permute(0, 1, 3, 2).reshape(b * f, c, length)
        if self.padding == "same":
            pad_total = self._resolve_padding(length)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_2d = F.pad(x_2d, (pad_left, pad_right))

        out = self.conv(x_2d)
        _, _, length_out = out.shape
        # Conv1d output is (B*F, C_out, L_out); restore to (B, F, L_out, C_out).
        out = out.permute(0, 2, 1).reshape(b, f, length_out, self.filters)

        if self.activation is not None:
            out = self.activation(out)

        out_mask = None
        if mask is not None:
            if self.padding == "same" and self.strides == 1:
                # With stride-1 "same" padding, output positions map one-to-one
                # to input positions, so the center of the kernel falls on a
                # valid position exactly when the input mask is True. This
                # avoids marking boundary positions as invalid because of the
                # zero padding used to implement "same".
                out_mask = mask[:, :, :length_out]
            else:
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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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


class GatedFrameGlobalMaxPooling(nn.Module):
    """Frame-aware global max pooling. Input (B,F,L,D) -> output (B,D)."""

    def __init__(self, return_gate: bool = False):
        super().__init__()
        self.return_gate = return_gate
        self.score_dense = nn.LazyLinear(1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, F, L, D)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        per_frame = x.max(dim=2)[0]  # (B, F, D)
        logits = self.score_dense(per_frame).squeeze(-1)  # (B, F)
        gates = F.softmax(logits, dim=1)
        pooled = (per_frame * gates.unsqueeze(-1)).sum(dim=1)  # (B, D)
        if self.return_gate:
            return pooled, gates
        return pooled


class AxialAttention(nn.Module):
    """Axial attention over the sequence axis.

    Supports Keras-style configs with ``feed_forward_dim`` (defaults to
    ``4 * embed_dim``), ``num_blocks`` (defaults to 1), and ``dropout_rate``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        feed_forward_dim: Optional[int] = None,
        num_blocks: int = 1,
        dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        dropout = dropout_rate if dropout_rate is not None else dropout
        feed_forward_dim = feed_forward_dim or embed_dim * 4
        layers = []
        for _ in range(num_blocks):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=feed_forward_dim,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=False,
                )
            )
        self.blocks = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, f, seq_len, d = x.shape
        # Reshape to (B*F, L, D)
        x_2d = x.reshape(b * f, seq_len, d)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.reshape(b * f, seq_len)
        out = x_2d
        for block in self.blocks:
            out = block(out, src_key_padding_mask=key_padding_mask)
        return out.reshape(b, f, seq_len, d), mask


class MaskedGlobalAvgPooling(nn.Module):
    """Global average pooling over the sequence axis, respecting masks."""

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, F, L, D)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(x.dtype)
            masked_x = x * mask_f
            # Compute the global masked average in one step to match the
            # TensorFlow GlobalAveragePooling2D behavior and reduce rounding
            # differences from two-stage averaging.
            count = mask_f.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            pooled = masked_x.sum(dim=(1, 2), keepdim=True) / count
            return pooled.squeeze(dim=(1, 2))
        return x.mean(dim=(1, 2))


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_wavelength: int = 10000):
        super().__init__()
        self.max_wavelength = max_wavelength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, L, D)
        b, f, length, d = x.shape
        position = torch.arange(length, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d, 2, device=x.device).float()
            * (-math.log(self.max_wavelength) / d)
        )
        pe = torch.zeros(length, d, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return x + pe.view(1, 1, length, d)


class ResidualBlock(nn.Module):
    """Residual wrapper around a layer or a small stack of MaskedConv1D layers.

    For Keras-style configs, ``config`` may contain ``block_size``,
    ``filters``, ``kernel_size``, ``use_bias``, and regularizer keys to build
    a stack of ``MaskedConv1D`` layers. If ``layer`` is provided directly,
    it is wrapped as before.
    """

    def __init__(self, layer: Optional[nn.Module] = None, config: Optional[dict] = None):
        super().__init__()
        if layer is not None:
            self.layer = layer
        elif config is not None:
            cfg = dict(config)
            block_size = cfg.pop("block_size", 1)
            cfg.setdefault("padding", "same")
            cfg.setdefault("activation", None)
            layers: List[nn.Module] = []
            for _ in range(block_size):
                layers.append(MaskedConv1D(**cfg))
            self.layer = nn.ModuleList(layers)
        else:
            raise ValueError("ResidualBlock requires either ``layer`` or ``config``.")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        residual = x
        out = x
        out_mask = mask
        if isinstance(self.layer, nn.ModuleList):
            for layer in self.layer:
                out, out_mask = layer(out, out_mask)
        elif "mask" in self.layer.forward.__code__.co_varnames:
            out = self.layer(out, mask)
            if isinstance(out, tuple):
                out, out_mask = out
        else:
            out = self.layer(out)

        # If dimensions differ, project the residual with a 1x1 conv.
        if out.shape[-1] != residual.shape[-1]:
            proj = nn.Conv1d(
                residual.shape[-1], out.shape[-1], kernel_size=1, bias=False
            ).to(out.device)
            residual = proj(
                residual.permute(0, 1, 3, 2).reshape(-1, residual.shape[-1], residual.shape[2])
            )
            residual = residual.permute(0, 2, 1).reshape_as(out)

        return out + residual, out_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, num_layers: int = 1, dropout: float = 0.0
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, f, length, d = x.shape
        x_2d = x.reshape(b * f, length, d)
        key_mask = None
        if mask is not None:
            key_mask = ~mask.reshape(b * f, length)
        out = self.encoder(x_2d, src_key_padding_mask=key_mask)
        return out.reshape(b, f, length, d), mask


class CrossFrameAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, f, length, d = x.shape
        # Treat frames as sequence: (B*L, F, D)
        x_t = x.permute(0, 2, 1, 3).reshape(b * length, f, d)
        key_mask = None
        if mask is not None:
            key_mask = ~mask.permute(0, 2, 1).reshape(b * length, f)
        out, _ = self.attn(x_t, x_t, x_t, key_padding_mask=key_mask, need_weights=False)
        out = self.norm(x_t + out).reshape(b, length, f, d).permute(0, 2, 1, 3)
        return out, mask


# ---------------------------------------------------------------------------
# Shape helpers for native PyTorch layer pipelines
# ---------------------------------------------------------------------------
class Permute(nn.Module):
    """Permute tensor dimensions."""

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class Flatten(nn.Module):
    """Flatten contiguous dimensions, defaulting to all dims after batch."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class SqueezeLast(nn.Module):
    """Squeeze the last dimension if it is size 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == 1:
            return x.squeeze(-1)
        return x


# ---------------------------------------------------------------------------
# Native PyTorch layer registry
# ---------------------------------------------------------------------------
def _build_native_layer(name: str, config: Dict[str, Any]) -> nn.Module:
    """Build a native ``torch.nn`` module from a config dict.

    The registry maps Keras-style layer names to PyTorch constructors.
    All constructor arguments are passed through ``config``.
    """
    config = config or {}
    name = name.lower()

    registry: Dict[str, Callable[..., nn.Module]] = {
        # Linear / dense
        "linear": nn.Linear,
        "dense": nn.Linear,
        # Convolutions
        "conv1d": nn.Conv1d,
        "conv2d": nn.Conv2d,
        # Normalization
        "batchnorm1d": nn.BatchNorm1d,
        "batchnorm2d": nn.BatchNorm2d,
        "layernorm": nn.LayerNorm,
        # Pooling
        "adaptive_avg_pool1d": nn.AdaptiveAvgPool1d,
        "adaptive_max_pool1d": nn.AdaptiveMaxPool1d,
        "avg_pool1d": nn.AvgPool1d,
        "max_pool1d": nn.MaxPool1d,
        # Regularization
        "dropout": nn.Dropout,
        "dropout1d": nn.Dropout1d,
        "dropout2d": nn.Dropout2d,
        # Activations
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax,
        "tanh": nn.Tanh,
        # Shape helpers
        "permute": Permute,
        "flatten": Flatten,
        "squeeze_last": SqueezeLast,
    }

    if name not in registry:
        raise ValueError(
            f"Unknown native layer: {name!r}. "
            f"Supported layers: {sorted(registry.keys())}"
        )

    return registry[name](**config)


class NativeSequential(nn.Module):
    """Sequential container for native PyTorch layers.

    Accepts an optional mask argument for compatibility with Jaeger's masked
    layer pipeline; the mask is returned unchanged. Shape-helper layers
    (``permute``, ``flatten``, ``squeeze_last``) can be interleaved with
    native ``torch.nn`` modules.
    """

    def __init__(self, layers: List[Dict[str, Any]]):
        super().__init__()
        modules: List[nn.Module] = []
        for layer_cfg in layers:
            name = layer_cfg.get("name")
            config = layer_cfg.get("config", {})
            modules.append(_build_native_layer(name, config))
        self.layers = nn.ModuleList(modules)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        for layer in self.layers:
            x = layer(x)
        return x, mask
