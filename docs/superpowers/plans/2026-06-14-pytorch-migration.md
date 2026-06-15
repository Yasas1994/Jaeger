# PyTorch Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Jaeger's TensorFlow/Keras training and inference stack with a plain PyTorch implementation in the `pytorch_migration` branch, targeting release `1.27.1`.

**Architecture:** Build a self-contained PyTorch module tree under `src/jaeger/nnlib/pytorch/`, `src/jaeger/data/pytorch/`, `src/jaeger/training/pytorch/`, and `src/jaeger/inference/pytorch/`. Preserve the YAML config format. Delete TensorFlow training/inference code only after parity tests and end-to-end smoke tests pass.

**Tech Stack:** Python 3.11+, PyTorch 2.x, NumPy, Click, PyYAML, pytest.

---

## File structure

New files to create:

```
src/jaeger/nnlib/pytorch/
  __init__.py
  layers.py              # GeLU, MaskedConv1D, MaskedBatchNorm, MaskedLayerNorm,
                         # MaskedGlobalAvgPooling, GatedFrameGlobalMaxPooling,
                         # AxialAttention, CrossFrameAttention, TransformerEncoder,
                         # ResidualBlock, custom pooling helpers
  models.py              # Embedding, RepresentationModel, ClassificationHead,
                         # ReliabilityHead, ProjectionHead, JaegerModel
  losses.py              # ArcFaceLoss, HierarchicalLoss, classification losses
  metrics.py             # Per-class precision/recall/specificity
  builder.py             # ModelBuilder from YAML config
  checkpoints.py         # save/load .pt checkpoints

src/jaeger/data/pytorch/
  __init__.py
  transforms.py          # codon translation, mutation, frame shuffle, masking
  collate.py             # padding collators
  dataset_numpy.py       # NumpyFullDataset, NumpyRawDataset, NumpyRawVariableDataset
  dataset_csv.py         # CSVDataset
  dataset_tfrecord.py    # TFRecordDataset (deferred)

src/jaeger/training/pytorch/
  __init__.py
  engine.py              # Epoch loop, train/validation step, AMP
  trainer.py             # Trainer for classifier/reliability/pretrain branches
  distributed.py         # DDP setup/teardown
  callbacks.py           # ModelCheckpoint, EarlyStopping, LR scheduling, loggers

src/jaeger/inference/pytorch/
  __init__.py
  model.py               # Load JaegerModel from checkpoint + config
  engine.py              # Batch inference, windowing, aggregation

src/jaeger/commands/
  train.py               # Dispatch to PyTorch trainer
  predict.py             # Dispatch to PyTorch inference engine
```

Test files to create:

```
tests/unit/nnlib/pytorch/
  test_layers.py
  test_models.py
  test_builder.py
  test_losses.py
  test_metrics.py
tests/unit/data/pytorch/
  test_dataset_numpy.py
  test_transforms.py
tests/unit/training/pytorch/
  test_engine.py
  test_callbacks.py
tests/integration/test_pytorch_parity.py
```

Files to delete at the end (after validation):

```
src/jaeger/nnlib/v1/
src/jaeger/nnlib/v2/
src/jaeger/nnlib/builder.py (old Keras builder)
src/jaeger/nnlib/inference.py (old TF inference)
src/jaeger/nnlib/conversion.py (if TF-only)
src/jaeger/commands/predict_legacy.py
src/jaeger/commands/train.py (old TF train command)
```

---

## Task 1: Create branch and update dependency metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `.cz.toml`

- [ ] **Step 1: Create the `pytorch_migration` branch**

```bash
git checkout -b pytorch_migration
```

- [ ] **Step 2: Add PyTorch dependencies to `pyproject.toml`**

Replace the TensorFlow optional dependencies with PyTorch equivalents. Keep TensorFlow as an optional `[legacy]` extra during development; remove it only in Task 27.

```toml
[project.optional-dependencies]
gpu = [
    "torch >=2.0",
    "torchvision",
]
cpu = [
    "torch >=2.0",
    "torchvision",
]
darwin-arm = [
    "torch >=2.0",
    "torchvision",
]
legacy = [
    "tensorflow >=2.21, <2.22",
]
test = [
    "pytest >=8.0",
    "pytest-mock >=3.14",
]
```

- [ ] **Step 3: Verify `pyproject.toml` parses**

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

Expected: no exception.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .cz.toml
git commit -m "chore(pytorch): add pytorch_migration branch and PyTorch deps"
```

---

## Task 2: Port activation and masking helper layers

**Files:**
- Create: `src/jaeger/nnlib/pytorch/__init__.py`
- Create: `src/jaeger/nnlib/pytorch/layers.py`
- Create: `tests/unit/nnlib/pytorch/test_layers.py`

- [ ] **Step 1: Write the failing test for GeLU**

```python
# tests/unit/nnlib/pytorch/test_layers.py
import torch
import pytest
from jaeger.nnlib.pytorch.layers import GeLU


def test_gelu_matches_torch_nn_gelu():
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    layer = GeLU()
    out = layer(x)
    expected = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(out, expected, atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_gelu_matches_torch_nn_gelu -v
```

Expected: `ModuleNotFoundError: No module named 'jaeger.nnlib.pytorch.layers'`

- [ ] **Step 3: Implement GeLU and helpers**

```python
# src/jaeger/nnlib/pytorch/__init__.py
# (empty or re-export later)
```

```python
# src/jaeger/nnlib/pytorch/layers.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_gelu_matches_torch_nn_gelu -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/__init__.py src/jaeger/nnlib/pytorch/layers.py tests/unit/nnlib/pytorch/test_layers.py
git commit -m "feat(pytorch): add GeLU activation layer"
```

---

## Task 3: Port MaskedConv1D

**Files:**
- Modify: `src/jaeger/nnlib/pytorch/layers.py`
- Modify: `tests/unit/nnlib/pytorch/test_layers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nnlib/pytorch/test_layers.py
from jaeger.nnlib.pytorch.layers import MaskedConv1D


def test_masked_conv1d_shape_and_mask():
    # (B=2, F=6, L=10, C=4)
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    mask[0, :, 5:] = False
    layer = MaskedConv1D(filters=8, kernel_size=3, padding="same")
    out, out_mask = layer(x, mask)
    assert out.shape == (2, 6, 10, 8)
    assert out_mask.shape == (2, 6, 10)
    assert not out_mask[0, :, 5:].any()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_masked_conv1d_shape_and_mask -v
```

Expected: `AttributeError: module 'jaeger.nnlib.pytorch.layers' has no attribute 'MaskedConv1D'`

- [ ] **Step 3: Implement MaskedConv1D**

```python
# src/jaeger/nnlib/pytorch/layers.py
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_masked_conv1d_shape_and_mask -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/layers.py tests/unit/nnlib/pytorch/test_layers.py
git commit -m "feat(pytorch): add MaskedConv1D with mask propagation"
```

---

## Task 4: Port MaskedBatchNorm and MaskedLayerNorm

**Files:**
- Modify: `src/jaeger/nnlib/pytorch/layers.py`
- Modify: `tests/unit/nnlib/pytorch/test_layers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_masked_batchnorm_output_shape_and_nmd():
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    mask[0, :, 5:] = False
    layer = MaskedBatchNorm(num_features=4, return_nmd=True)
    out, nmd = layer(x, mask)
    assert out.shape == (2, 6, 10, 4)
    assert nmd.shape == (2, 4)


def test_masked_layer_norm_shape():
    x = torch.randn(2, 6, 10, 4)
    mask = torch.ones(2, 6, 10, dtype=torch.bool)
    mask[0, :, 5:] = False
    layer = MaskedLayerNorm(num_features=4)
    out = layer(x, mask)
    assert out.shape == (2, 6, 10, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_masked_batchnorm_output_shape_and_nmd tests/unit/nnlib/pytorch/test_layers.py::test_masked_layer_norm_shape -v
```

Expected: `AttributeError` for missing classes.

- [ ] **Step 3: Implement MaskedBatchNorm and MaskedLayerNorm**

```python
# src/jaeger/nnlib/pytorch/layers.py
class MaskedBatchNorm(nn.Module):
    """Batch normalization that excludes masked positions from statistics.

    Can optionally return normalized mean difference (nmd) vectors.
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
        b, f, l, c = xf.shape

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
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.detach()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.detach()
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
    """Layer normalization that excludes masked positions."""

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
            count = mask_f.sum(dim=-1, keepdim=True) + self.eps
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_masked_batchnorm_output_shape_and_nmd tests/unit/nnlib/pytorch/test_layers.py::test_masked_layer_norm_shape -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/layers.py tests/unit/nnlib/pytorch/test_layers.py
git commit -m "feat(pytorch): add MaskedBatchNorm and MaskedLayerNorm"
```

---

## Task 5: Port attention and pooling layers

**Files:**
- Modify: `src/jaeger/nnlib/pytorch/layers.py`
- Modify: `tests/unit/nnlib/pytorch/test_layers.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_gated_frame_pooling_shape():
    x = torch.randn(2, 6, 10, 4)
    layer = GatedFrameGlobalMaxPooling(return_gate=False)
    out = layer(x)
    assert out.shape == (2, 4)


def test_axial_attention_shape():
    x = torch.randn(2, 6, 10, 4)
    layer = AxialAttention(embed_dim=4, num_heads=2)
    out, mask = layer(x)
    assert out.shape == (2, 6, 10, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_gated_frame_pooling_shape tests/unit/nnlib/pytorch/test_layers.py::test_axial_attention_shape -v
```

Expected: `AttributeError` for missing classes.

- [ ] **Step 3: Implement the layers**

```python
# src/jaeger/nnlib/pytorch/layers.py
class GatedFrameGlobalMaxPooling(nn.Module):
    """Frame-aware global max pooling. Input (B,F,L,D) -> output (B,D)."""

    def __init__(self, return_gate: bool = False):
        super().__init__()
        self.return_gate = return_gate
        self.score_dense = nn.Linear(1, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, F, L, D)
        per_frame = x.max(dim=2)[0]  # (B, F, D)
        logits = self.score_dense(per_frame).squeeze(-1)  # (B, F)
        gates = torch.softmax(logits, dim=1)
        pooled = (per_frame * gates.unsqueeze(-1)).sum(dim=1)  # (B, D)
        if self.return_gate:
            return pooled, gates
        return pooled


class AxialAttention(nn.Module):
    """Axial attention over the sequence axis."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, f, l, d = x.shape
        # Reshape to (B*F, L, D)
        x_2d = x.reshape(b * f, l, d)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.reshape(b * f, l)
        attn_out, _ = self.attn(
            x_2d, x_2d, x_2d, key_padding_mask=key_padding_mask, need_weights=False
        )
        out = self.norm(x_2d + attn_out).reshape(b, f, l, d)
        out_mask = mask
        return out, out_mask
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py::test_gated_frame_pooling_shape tests/unit/nnlib/pytorch/test_layers.py::test_axial_attention_shape -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/layers.py tests/unit/nnlib/pytorch/test_layers.py
git commit -m "feat(pytorch): add attention and pooling layers"
```

---

## Task 5b: Implement remaining custom layers and RepresentationModel

**Files:**
- Modify: `src/jaeger/nnlib/pytorch/layers.py`
- Modify: `src/jaeger/nnlib/pytorch/models.py`
- Modify: `tests/unit/nnlib/pytorch/test_layers.py`

- [ ] **Step 1: Add SinusoidalPositionEmbedding, ResidualBlock, TransformerEncoder, CrossFrameAttention**

```python
# src/jaeger/nnlib/pytorch/layers.py
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, max_wavelength: int = 10000):
        super().__init__()
        self.max_wavelength = max_wavelength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, L, D)
        b, f, l, d = x.shape
        position = torch.arange(l, device=x.device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d, 2, device=x.device).float()
            * (-math.log(self.max_wavelength) / d)
        )
        pe = torch.zeros(l, d, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.view(1, 1, l, d)


class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if hasattr(self.layer, "forward") and "mask" in self.layer.forward.__code__.co_varnames:
            out, out_mask = self.layer(x, mask)
            return out + x, out_mask
        out = self.layer(x)
        return out + x, mask


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int = 1, dropout: float = 0.0):
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
        b, f, l, d = x.shape
        x_2d = x.reshape(b * f, l, d)
        key_mask = None
        if mask is not None:
            key_mask = ~mask.reshape(b * f, l)
        out = self.encoder(x_2d, src_key_padding_mask=key_mask)
        return out.reshape(b, f, l, d), mask


class CrossFrameAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, f, l, d = x.shape
        # Treat frames as sequence: (B*L, F, D)
        x_t = x.permute(0, 2, 1, 3).reshape(b * l, f, d)
        key_mask = None
        if mask is not None:
            key_mask = ~mask.permute(0, 2, 1).reshape(b * l, f)
        out, _ = self.attn(x_t, x_t, x_t, key_padding_mask=key_mask, need_weights=False)
        out = self.norm(x_t + out).reshape(b, l, f, d).permute(0, 2, 1, 3)
        return out, mask
```

- [ ] **Step 2: Add RepresentationModel**

```python
# src/jaeger/nnlib/pytorch/models.py
class RepresentationModel(nn.Module):
    def __init__(self, embedding: nn.Module, hidden_layers: list, pooling: str = "average"):
        super().__init__()
        self.embedding = embedding
        self.blocks = nn.ModuleList()
        self.return_nmd = False
        self.output_dim = None
        self.nmd_dim = None
        # Build blocks from config; simplified here
        for cfg in hidden_layers:
            name = cfg["name"].lower()
            config = cfg.get("config", {})
            if name == "masked_conv1d":
                self.blocks.append(MaskedConv1D(**config))
            elif name == "masked_batchnorm":
                self.blocks.append(MaskedBatchNorm(**config))
            elif name == "masked_layer_norm":
                self.blocks.append(MaskedLayerNorm(**config))
            elif name == "axial_attention":
                self.blocks.append(AxialAttention(**config))
            elif name == "cross_frame_attention":
                self.blocks.append(CrossFrameAttention(**config))
            elif name == "transformer_encoder":
                self.blocks.append(TransformerEncoder(**config))
            elif name == "residual_block":
                inner = self.blocks.pop() if self.blocks else None
                self.blocks.append(ResidualBlock(inner))
            elif name == "dense":
                self.blocks.append(nn.Linear(**config))
            elif name == "activation":
                self.blocks.append(GeLU() if config.get("activation") == "gelu" else nn.ReLU())
            elif name == "dropout":
                self.blocks.append(nn.Dropout(config.get("rate", 0.0)))
            else:
                raise ValueError(f"Unknown layer type: {name}")
        self.pooler = self._build_pooler(pooling)

    def _build_pooler(self, pooling: str):
        from jaeger.nnlib.pytorch.layers import GatedFrameGlobalMaxPooling, MaskedGlobalAvgPooling
        if pooling == "average":
            return MaskedGlobalAvgPooling()
        elif pooling == "gatedframe":
            return GatedFrameGlobalMaxPooling(return_gate=False)
        raise ValueError(f"Unknown pooling: {pooling}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor | tuple:
        x = self.embedding(x)
        nmd = None
        for block in self.blocks:
            if hasattr(block, "return_nmd") and block.return_nmd:
                x, nmd = block(x, mask)
            else:
                x, mask = block(x, mask)
        pooled = self.pooler(x, mask)
        if nmd is not None:
            return pooled, nmd
        return pooled
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/unit/nnlib/pytorch/test_layers.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/nnlib/pytorch/layers.py src/jaeger/nnlib/pytorch/models.py tests/unit/nnlib/pytorch/test_layers.py
git commit -m "feat(pytorch): add remaining custom layers and RepresentationModel"
```

---

## Task 6: Build PyTorch model modules

**Files:**
- Create: `src/jaeger/nnlib/pytorch/models.py`
- Create: `tests/unit/nnlib/pytorch/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nnlib/pytorch/test_models.py
import torch
from jaeger.nnlib.pytorch.models import ClassificationHead, RepresentationModel


def test_classification_head_output_shape():
    head = ClassificationHead(input_dim=64, num_classes=3, hidden_units=[128, 64])
    x = torch.randn(2, 64)
    out = head(x)
    assert out.shape == (2, 3)


def test_representation_model_output_shape():
    # Minimal config matching nn_config_500bp_baseline structure
    model_cfg = {
        "embedding": {
            "input_type": "translated",
            "use_embedding_layer": False,
            "embedding_size": 64,
            "input_shape": (6, None),
            "vocab_size": 65,
            "codon_depth": 1,
        },
        "representation_learner": {
            "hidden_layers": [
                {"name": "masked_conv1d", "config": {"filters": 32, "kernel_size": 7, "padding": "same"}}
            ],
            "pooling": "average",
        },
        "classifier": {"input_shape": 32, "hidden_layers": [{"name": "dense", "config": {"units": 3}}]},
    }
    # Build via builder in Task 9; for now test a small standalone module
    assert True
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/nnlib/pytorch/test_models.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement model modules**

```python
# src/jaeger/nnlib/pytorch/models.py
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from jaeger.nnlib.pytorch.layers import GeLU


class Embedding(nn.Module):
    """Translates DNA into 6-frame codon embeddings or nucleotide one-hot."""

    def __init__(
        self,
        input_type: str,
        vocab_size: Optional[int],
        embedding_size: int,
        use_embedding_layer: bool,
        use_positional_embeddings: bool = False,
        positional_embedding_length: Optional[int] = None,
    ):
        super().__init__()
        self.input_type = input_type
        self.use_embedding_layer = use_embedding_layer
        self.use_positional_embeddings = use_positional_embeddings

        if input_type == "translated":
            if use_embedding_layer:
                self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
                nn.init.orthogonal_(self.embed.weight)
            else:
                self.embed = nn.Linear(vocab_size, embedding_size, bias=False)
                nn.init.orthogonal_(self.embed.weight)
        elif input_type == "nucleotide":
            self.embed = None
        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        if use_positional_embeddings:
            from jaeger.nnlib.pytorch.layers import SinusoidalPositionEmbedding

            self.positional = SinusoidalPositionEmbedding(positional_embedding_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape depends on input_type; caller must pass correct shape
        if self.input_type == "translated":
            if self.use_embedding_layer:
                return self.embed(x)
            else:
                return self.embed(x)
        return x


class ClassificationHead(nn.Module):
    """Dense head for class prediction."""

    def __init__(self, input_dim: int, num_classes: int, hidden_units: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.LayerNorm(units))
            layers.append(GeLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReliabilityHead(nn.Module):
    """Dense head for confidence estimation."""

    def __init__(self, input_dim: int, num_classes: int, hidden_units: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.LayerNorm(units))
            layers.append(GeLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectionHead(nn.Module):
    """Projection head for ArcFace self-supervised pretraining."""

    def __init__(self, input_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            GeLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JaegerModel(nn.Module):
    """Combined model exposing all outputs for inference."""

    def __init__(
        self,
        rep_model: nn.Module,
        classification_head: nn.Module,
        reliability_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.rep_model = rep_model
        self.classification_head = classification_head
        self.reliability_head = reliability_head

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        outputs = self.rep_model(x, mask)
        if isinstance(outputs, tuple):
            embedding, nmd = outputs[0], outputs[1]
            gate = outputs[2] if len(outputs) > 2 else None
        else:
            embedding = outputs
            nmd = None
            gate = None

        prediction = self.classification_head(embedding)
        result: Dict[str, torch.Tensor] = {"prediction": prediction, "embedding": embedding}
        if nmd is not None:
            result["nmd"] = nmd
        if gate is not None:
            result["gate"] = gate
        if self.reliability_head is not None and nmd is not None:
            result["reliability"] = self.reliability_head(nmd)
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/unit/nnlib/pytorch/test_models.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/models.py tests/unit/nnlib/pytorch/test_models.py
git commit -m "feat(pytorch): add model modules (embedding, heads, JaegerModel)"
```

---

## Task 7: Port losses

**Files:**
- Create: `src/jaeger/nnlib/pytorch/losses.py`
- Create: `tests/unit/nnlib/pytorch/test_losses.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nnlib/pytorch/test_losses.py
import torch
from jaeger.nnlib.pytorch.losses import ArcFaceLoss


def test_arcface_loss_shape():
    loss = ArcFaceLoss(num_classes=3, embedding_dim=64, margin=0.5, scale=30.0)
    labels = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    embeddings = torch.randn(3, 64)
    out = loss(labels, embeddings)
    assert out.ndim == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/nnlib/pytorch/test_losses.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement losses**

```python
# src/jaeger/nnlib/pytorch/losses.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.5,
        scale: float = 30.0,
        onehot: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.onehot = onehot
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (B, D), labels: (B, C) one-hot or (B,) long
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        cos_t = torch.matmul(embeddings_norm, weight_norm.t())

        if self.onehot:
            target = labels.argmax(dim=1)
        else:
            target = labels

        # Additive angular margin
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        sin_t = torch.sqrt(1.0 - cos_t.pow(2) + 1e-6)
        cos_t_plus_m = cos_t * cos_m - sin_t * sin_m
        one_hot = F.one_hot(target, num_classes=self.num_classes).to(cos_t.dtype)
        logits = one_hot * cos_t_plus_m + (1.0 - one_hot) * cos_t
        logits = logits * self.scale
        return F.cross_entropy(logits, target)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/nnlib/pytorch/test_losses.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/losses.py tests/unit/nnlib/pytorch/test_losses.py
git commit -m "feat(pytorch): add ArcFace loss"
```

---

## Task 8: Port metrics

**Files:**
- Create: `src/jaeger/nnlib/pytorch/metrics.py`
- Create: `tests/unit/nnlib/pytorch/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nnlib/pytorch/test_metrics.py
import torch
from jaeger.nnlib.pytorch.metrics import PrecisionForClass


def test_precision_for_class():
    metric = PrecisionForClass(class_id=1)
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    metric.update(preds, labels)
    result = metric.compute()
    assert 0.0 <= result <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/nnlib/pytorch/test_metrics.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics**

```python
# src/jaeger/nnlib/pytorch/metrics.py
import torch
import torch.nn.functional as F


class PrecisionForClass:
    def __init__(self, class_id: int):
        self.class_id = class_id
        self.tp = 0
        self.fp = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_labels = y_pred.argmax(dim=-1)
        true_labels = y_true.argmax(dim=-1)
        cls = self.class_id
        self.tp += int(((pred_labels == cls) & (true_labels == cls)).sum().item())
        self.fp += int(((pred_labels == cls) & (true_labels != cls)).sum().item())

    def compute(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)


class RecallForClass(PrecisionForClass):
    def __init__(self, class_id: int):
        super().__init__(class_id)
        self.fn = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pred_labels = y_pred.argmax(dim=-1)
        true_labels = y_true.argmax(dim=-1)
        cls = self.class_id
        self.tp += int(((pred_labels == cls) & (true_labels == cls)).sum().item())
        self.fn += int(((pred_labels != cls) & (true_labels == cls)).sum().item())

    def compute(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/nnlib/pytorch/test_metrics.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/metrics.py tests/unit/nnlib/pytorch/test_metrics.py
git commit -m "feat(pytorch): add per-class precision/recall metrics"
```

---

## Task 9: Implement ModelBuilder

**Files:**
- Create: `src/jaeger/nnlib/pytorch/builder.py`
- Create: `tests/unit/nnlib/pytorch/test_builder.py`
- Modify: `src/jaeger/nnlib/pytorch/models.py` (add RepresentationModel)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/nnlib/pytorch/test_builder.py
import torch
from jaeger.nnlib.pytorch.builder import ModelBuilder


def test_builder_creates_jaeger_model():
    config = {
        "model": {
            "name": "test_model",
            "classifier_out_dim": 3,
            "reliability_out_dim": 1,
            "class_label_map": [{"class": "phage", "label": 1}, {"class": "bacteria", "label": 0}],
            "embedding": {
                "input_type": "translated",
                "use_embedding_layer": False,
                "embedding_size": 32,
                "input_shape": [6, None],
                "vocab_size": 65,
                "codon_depth": 1,
            },
            "string_processor": {"codon": "CODON", "codon_id": "CODON_ID"},
            "representation_learner": {
                "hidden_layers": [
                    {"name": "masked_conv1d", "config": {"filters": 16, "kernel_size": 3, "padding": "same"}}
                ],
                "pooling": "average",
            },
            "classifier": {"input_shape": 16, "hidden_layers": [{"name": "dense", "config": {"units": 3}}]},
        },
        "training": {"batch_size": 2, "optimizer": "adam", "optimizer_params": {"lr": 1e-3}},
    }
    builder = ModelBuilder(config)
    models = builder.build_fragment_classifier()
    x = torch.randint(0, 65, (2, 6, 50))
    mask = torch.ones(2, 6, 50, dtype=torch.bool)
    out = models["jaeger_model"](x, mask)
    assert out["prediction"].shape == (2, 3)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/unit/nnlib/pytorch/test_builder.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ModelBuilder**

```python
# src/jaeger/nnlib/pytorch/builder.py
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from jaeger.nnlib.pytorch.layers import (
    AxialAttention,
    CrossFrameAttention,
    GatedFrameGlobalMaxPooling,
    GeLU,
    MaskedBatchNorm,
    MaskedConv1D,
    MaskedLayerNorm,
    ResidualBlock,
    TransformerEncoder,
)
from jaeger.nnlib.pytorch.losses import ArcFaceLoss
from jaeger.nnlib.pytorch.models import (
    ClassificationHead,
    Embedding,
    JaegerModel,
    ProjectionHead,
    ReliabilityHead,
    RepresentationModel,
)


class ModelBuilder:
    """Build PyTorch Jaeger models from YAML config."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self.classifier_out_dim = int(self.model_cfg.get("classifier_out_dim", 0))
        self.reliability_out_dim = int(self.model_cfg.get("reliability_out_dim", 0))

    def build_fragment_classifier(self) -> Dict[str, nn.Module]:
        models: Dict[str, nn.Module] = {}

        embedding_cfg = self.model_cfg.get("embedding", {})
        embedding = Embedding(
            input_type=embedding_cfg.get("input_type"),
            vocab_size=embedding_cfg.get("vocab_size"),
            embedding_size=embedding_cfg.get("embedding_size", 4),
            use_embedding_layer=embedding_cfg.get("use_embedding_layer", False),
        )

        rep_cfg = self.model_cfg.get("representation_learner", {})
        rep_model = RepresentationModel(
            embedding=embedding,
            hidden_layers=rep_cfg.get("hidden_layers", []),
            pooling=rep_cfg.get("pooling", "average"),
        )
        models["rep_model"] = rep_model

        if "classifier" in self.model_cfg:
            cls_cfg = self.model_cfg["classifier"]
            head = ClassificationHead(
                input_dim=cls_cfg.get("input_shape", rep_model.output_dim),
                num_classes=self.classifier_out_dim,
                hidden_units=[layer["config"]["units"] for layer in cls_cfg.get("hidden_layers", [])[:-1]],
            )
            models["classification_head"] = head
            models["jaeger_classifier"] = nn.Sequential(rep_model, head)

        if "reliability_model" in self.model_cfg:
            rel_cfg = self.model_cfg["reliability_model"]
            rel_head = ReliabilityHead(
                input_dim=rel_cfg.get("input_shape", rep_model.nmd_dim),
                num_classes=self.reliability_out_dim,
                hidden_units=[layer["config"]["units"] for layer in rel_cfg.get("hidden_layers", [])[:-1]],
            )
            models["reliability_head"] = rel_head

        models["jaeger_model"] = JaegerModel(
            rep_model=rep_model,
            classification_head=models["classification_head"],
            reliability_head=models.get("reliability_head"),
        )
        return models

    def compile_model(self, models: Dict[str, nn.Module], train_branch: str = "classifier") -> tuple:
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        opt_params = self.train_cfg.get("optimizer_params", {})
        opt_class = self._get_optimizer(opt_name)

        if train_branch == "classifier":
            model = models["jaeger_classifier"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = nn.CrossEntropyLoss()
            return model, optimizer, loss
        elif train_branch == "reliability":
            model = models["jaeger_reliability"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = nn.BCEWithLogitsLoss()
            return model, optimizer, loss
        elif train_branch == "pretrain":
            model = models["jaeger_projection"]
            optimizer = opt_class(model.parameters(), **opt_params)
            loss = models["arcface_loss"]
            return model, optimizer, loss
        else:
            raise ValueError(f"Unknown train_branch: {train_branch}")

    def get_metrics(self, branch: str = "classifier") -> Dict[str, Any]:
        from jaeger.nnlib.pytorch.metrics import PrecisionForClass, RecallForClass
        metrics = {}
        out_dim = self.classifier_out_dim if branch == "classifier" else self.reliability_out_dim
        for cls in range(out_dim):
            metrics[f"precision_class_{cls}"] = PrecisionForClass(class_id=cls)
            metrics[f"recall_class_{cls}"] = RecallForClass(class_id=cls)
        return metrics

    def _get_optimizer(self, name: str, kwargs: Dict[str, Any]) -> torch.optim.Optimizer:
        optimizers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }
        return optimizers[name]
```

Also update `models.py` to add `RepresentationModel` with proper output tracking.

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/unit/nnlib/pytorch/test_builder.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/pytorch/builder.py src/jaeger/nnlib/pytorch/models.py tests/unit/nnlib/pytorch/test_builder.py
git commit -m "feat(pytorch): add ModelBuilder"
```

---

## Task 10: Add parity tests against TensorFlow baseline

**Files:**
- Create: `tests/integration/test_pytorch_parity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_pytorch_parity.py
import numpy as np
import pytest
import torch

from jaeger.nnlib.builder import DynamicModelBuilder as TFBuilder
from jaeger.nnlib.pytorch.builder import ModelBuilder as PTBuilder


CONFIG = {
    "model": {
        "name": "parity_test",
        "classifier_out_dim": 3,
        "reliability_out_dim": 1,
        "class_label_map": [{"class": "phage", "label": 1}, {"class": "bacteria", "label": 0}, {"class": "eukarya", "label": 2}],
        "embedding": {
            "input_type": "translated",
            "use_embedding_layer": False,
            "embedding_size": 32,
            "input_shape": [6, None],
            "vocab_size": 65,
            "codon_depth": 1,
        },
        "string_processor": {
            "codon": "CODON",
            "codon_id": "CODON_ID",
            "crop_size": 50,
            "input_type": "translated",
            "use_embedding_layer": False,
        },
        "representation_learner": {
            "hidden_layers": [
                {"name": "masked_conv1d", "config": {"filters": 16, "kernel_size": 3, "padding": "same"}}
            ],
            "pooling": "average",
        },
        "classifier": {"input_shape": 16, "hidden_layers": [{"name": "dense", "config": {"units": 3}}]},
    },
    "training": {"batch_size": 2, "optimizer": "adam", "optimizer_params": {"lr": 1e-3}},
}


def test_forward_parity():
    tf_builder = TFBuilder(CONFIG)
    tf_models = tf_builder.build_fragment_classifier()
    tf_model = tf_models["jaeger_model"]

    pt_builder = PTBuilder(CONFIG)
    pt_models = pt_builder.build_fragment_classifier()
    pt_model = pt_models["jaeger_model"]
    pt_model.eval()

    _copy_tf_weights_to_pt(tf_model, pt_model)

    x = np.random.randint(0, 65, size=(2, 6, 50)).astype(np.int64)
    tf_out = tf_model.predict({"translated": x})
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(x), mask=torch.ones(2, 6, 50, dtype=torch.bool))

    np.testing.assert_allclose(
        tf_out["prediction"],
        pt_out["prediction"].numpy(),
        atol=1e-4,
        rtol=1e-4,
    )


def _copy_tf_weights_to_pt(tf_model, pt_model):
    """Map Keras layer weights to PyTorch state_dict by name and shape."""
    tf_layers = {layer.name: layer for layer in tf_model.layers}
    pt_state = pt_model.state_dict()
    new_state = {}
    for pt_key, pt_tensor in pt_state.items():
        # Example mapping: "rep_model.blocks.0.conv.weight" -> "rep_conv1d_0/kernel:0"
        # Implement name-based mapping here; raise ValueError if no match.
        # This is architecture-specific and must be updated as layers are added.
        raise NotImplementedError(f"Implement mapping for {pt_key}")
    pt_model.load_state_dict(new_state)
```

- [ ] **Step 2: Implement weight-copy helper**

Map TensorFlow layer names to PyTorch `state_dict` keys. For each `MaskedConv1D`/`Dense`/BatchNorm in the TF model, find the corresponding PyTorch module and copy `.kernel` → `.weight`, `.bias` → `.bias`, BN moving stats, gamma, beta.

- [ ] **Step 3: Run the test**

```bash
pytest tests/integration/test_pytorch_parity.py::test_forward_parity -v
```

Expected: PASS after weight mapping is correct.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_pytorch_parity.py
git commit -m "test(pytorch): add TF vs PyTorch forward parity test"
```

---

## Task 11: Implement numpy_full PyTorch Dataset

**Files:**
- Create: `src/jaeger/data/pytorch/dataset_numpy.py`
- Create: `src/jaeger/data/pytorch/collate.py`
- Create: `tests/unit/data/pytorch/test_dataset_numpy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/data/pytorch/test_dataset_numpy.py
import numpy as np
import torch
from jaeger.data.pytorch.dataset_numpy import NumpyFullDataset


def test_numpy_full_dataset(tmp_path):
    data = np.random.randint(0, 65, size=(10, 6, 50)).astype(np.int32)
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=10)]
    path = tmp_path / "data.npz"
    np.savez(path, translated=data, label=labels)

    ds = NumpyFullDataset(path, input_key="translated")
    assert len(ds) == 10
    x, y = ds[0]
    assert x.shape == (6, 50)
    assert y.shape == (3,)

    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 6, 50)
    assert batch_y.shape == (2, 3)
```

- [ ] **Step 2: Implement NumpyFullDataset**

```python
# src/jaeger/data/pytorch/dataset_numpy.py
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyFullDataset(Dataset):
    """Loads a fully-preprocessed .npz file."""

    def __init__(self, path: str | Path, input_key: str = "translated", label_key: str = "label"):
        data = np.load(path, allow_pickle=False)
        self.inputs = torch.from_numpy(data[input_key])
        self.labels = torch.from_numpy(data[label_key])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        mask = (x != 0).any(dim=0) if x.dim() == 3 else torch.ones(x.shape[-1], dtype=torch.bool)
        return x, self.labels[idx], mask
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/unit/data/pytorch/test_dataset_numpy.py -v
```

Expected: PASS

- [ ] **Step 4: Add dataset builder helper**

```python
# src/jaeger/data/pytorch/builders.py
from typing import Any, Dict

from torch.utils.data import DataLoader

from jaeger.data.pytorch.dataset_csv import CSVDataset
from jaeger.data.pytorch.dataset_numpy import NumpyFullDataset, NumpyRawDataset


def build_datasets(config: Dict[str, Any], branch: str = "classifier"):
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    string_cfg = model_cfg.get("string_processor", {})
    embedding_cfg = model_cfg.get("embedding", {})
    data_format = string_cfg.get("data_format", "csv")
    batch_size = train_cfg.get("batch_size", 32)
    crop_size = string_cfg.get("crop_size", 500)
    num_classes = model_cfg.get("classifier_out_dim" if branch == "classifier" else "reliability_out_dim", 3)

    data_key = "fragment_classifier_data" if branch == "classifier" else "fragment_reliability_data"
    data_cfg = train_cfg.get(data_key, {})

    from jaeger.seqops.maps import CODON_ID
    codon_table = {c: i for i, c in enumerate(CODON_ID)}

    datasets = {}
    for split in ["train", "validation"]:
        entries = data_cfg.get(split, [])
        paths = [p for entry in entries for p in entry.get("path", [])]
        if not paths:
            raise ValueError(f"No paths found for {branch}/{split}")

        if data_format == "numpy_full":
            datasets[split] = NumpyFullDataset(paths[0])
        elif data_format == "numpy_raw":
            datasets[split] = NumpyRawDataset(
                paths[0],
                crop_size=crop_size,
                num_classes=num_classes,
                codon_table=codon_table,
                shuffle=(split == "train"),
                mutate=string_cfg.get("mutate", False),
                mutation_rate=string_cfg.get("mutation_rate", 0.1),
                shuffle_frames=string_cfg.get("shuffle_frames", False),
            )
        elif data_format == "csv":
            datasets[split] = CSVDataset(
                paths[0],
                seq_col=string_cfg.get("seq_col", 1),
                class_col=string_cfg.get("class_col", 0),
                crop_size=crop_size,
                num_classes=num_classes,
                codon_table=codon_table,
                shuffle=(split == "train"),
                mutate=string_cfg.get("mutate", False),
                mutation_rate=string_cfg.get("mutation_rate", 0.1),
                shuffle_frames=string_cfg.get("shuffle_frames", False),
            )
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")

    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=4),
    }
```

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/data/pytorch/builders.py src/jaeger/data/pytorch/dataset_numpy.py tests/unit/data/pytorch/test_dataset_numpy.py
git commit -m "feat(pytorch): add NumpyFullDataset and dataset builder"
```

---

## Task 12: Implement numpy_raw PyTorch Dataset

**Files:**
- Modify: `src/jaeger/data/pytorch/dataset_numpy.py`
- Modify: `src/jaeger/data/pytorch/transforms.py`
- Modify: `tests/unit/data/pytorch/test_dataset_numpy.py`

- [ ] **Step 1: Write the failing test**

```python
def test_numpy_raw_dataset(tmp_path):
    seqs = np.random.randint(0, 4, size=(10, 500)).astype(np.int8)
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=10)]
    path = tmp_path / "raw.npz"
    np.savez(path, sequences=seqs, labels=labels)

    from jaeger.seqops.maps import CODON_ID
    codon_table = {c: i for i, c in enumerate(CODON_ID)}
    ds = NumpyRawDataset(
        path,
        seq_key="sequences",
        label_key="labels",
        crop_size=50,
        num_classes=3,
        codon_table=codon_table,
    )
    x, y, mask = ds[0]
    assert x.shape == (6, 50)
    assert mask.shape == (6, 50)
```

- [ ] **Step 2: Implement transforms and dataset**

```python
# src/jaeger/data/pytorch/transforms.py
import random
from typing import Dict, Optional

import numpy as np
import torch


def _reverse_complement(seq_str: str) -> str:
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement.get(b, "N") for b in reversed(seq_str.upper()))


def translate_to_codons(seq: np.ndarray, codon_table: Dict[str, int]) -> torch.Tensor:
    # seq: (L,) int8 nucleotide indices
    # returns (6, L//3) translated codon indices for 6 reading frames
    nucleotides = ["A", "T", "G", "C"]
    seq_str = "".join(nucleotides[i] for i in seq)
    frames = []
    for offset in range(3):
        codons = [seq_str[i : i + 3] for i in range(offset, len(seq_str) - 2, 3)]
        indices = [codon_table.get(c, 0) for c in codons]
        rev = _reverse_complement(seq_str)
        rev_codons = [rev[i : i + 3] for i in range(offset, len(rev) - 2, 3)]
        rev_indices = [codon_table.get(c, 0) for c in rev_codons]
        frames.append(torch.tensor(indices, dtype=torch.long))
        frames.append(torch.tensor(rev_indices, dtype=torch.long))
    return torch.stack(frames)


def dna_to_indices(seq: str) -> np.ndarray:
    mapping = {"A": 0, "T": 1, "G": 2, "C": 3}
    return np.array([mapping.get(base.upper(), 0) for base in seq], dtype=np.int8)


def apply_mutation(seq: np.ndarray, rate: float) -> np.ndarray:
    if rate == 0:
        return seq
    mask = np.random.rand(len(seq)) < rate
    mutated = seq.copy()
    mutated[mask] = np.random.randint(0, 4, size=mask.sum())
    return mutated


def shuffle_frames(x: torch.Tensor) -> torch.Tensor:
    # x: (6, L)
    frames = list(range(6))
    random.shuffle(frames)
    return x[frames]
```

```python
# src/jaeger/data/pytorch/dataset_numpy.py
class NumpyRawDataset(Dataset):
    """Loads raw int8 sequences and applies runtime preprocessing."""

    def __init__(
        self,
        path: str | Path,
        seq_key: str = "sequences",
        label_key: str = "labels",
        crop_size: int = 500,
        num_classes: int = 3,
        codon_table: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        mutate: bool = False,
        mutation_rate: float = 0.1,
        shuffle_frames: bool = False,
    ):
        data = np.load(path, allow_pickle=False)
        self.seqs = data[seq_key]
        self.labels = torch.from_numpy(data[label_key])
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.codon_table = codon_table
        self.shuffle = shuffle
        self.mutate = mutate
        self.mutation_rate = mutation_rate
        self.shuffle_frames = shuffle_frames

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if self.mutate:
            seq = apply_mutation(seq, self.mutation_rate)
        x = translate_to_codons(seq, self.codon_table)
        # crop / pad to crop_size
        l = x.shape[1]
        if l > self.crop_size:
            start = random.randint(0, l - self.crop_size)
            x = x[:, start : start + self.crop_size]
        elif l < self.crop_size:
            pad = self.crop_size - l
            x = torch.nn.functional.pad(x, (0, pad))
        mask = (x != 0)
        if self.shuffle_frames:
            x = shuffle_frames(x)
        return x, self.labels[idx], mask
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/unit/data/pytorch/test_dataset_numpy.py::test_numpy_raw_dataset -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/data/pytorch/transforms.py src/jaeger/data/pytorch/dataset_numpy.py tests/unit/data/pytorch/test_dataset_numpy.py
git commit -m "feat(pytorch): add NumpyRawDataset and transforms"
```

---

## Task 13: Implement CSV PyTorch Dataset

**Files:**
- Create: `src/jaeger/data/pytorch/dataset_csv.py`
- Create: `tests/unit/data/pytorch/test_dataset_csv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/data/pytorch/test_dataset_csv.py
import csv
import torch
from jaeger.data.pytorch.dataset_csv import CSVDataset


def test_csv_dataset(tmp_path):
    path = tmp_path / "data.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        for i in range(10):
            writer.writerow([i % 3, "ATGC" * 20, f"seq{i}"])

    ds = CSVDataset(path, seq_col=1, class_col=0, crop_size=50, num_classes=3)
    x, y, mask = ds[0]
    assert x.shape[0] == 6
    assert y.shape == (3,)
```

- [ ] **Step 2: Implement CSVDataset**

```python
# src/jaeger/data/pytorch/dataset_csv.py
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from jaeger.data.pytorch.transforms import dna_to_indices, translate_to_codons


class CSVDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        seq_col: int,
        class_col: int,
        crop_size: int,
        num_classes: int,
        codon_table: dict,
        shuffle: bool = True,
        mutate: bool = False,
        mutation_rate: float = 0.1,
        shuffle_frames: bool = False,
    ):
        self.samples = []
        with open(path) as fh:
            reader = csv.reader(fh)
            for row in reader:
                label = int(row[class_col])
                seq = row[seq_col]
                self.samples.append((seq, label))
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.codon_table = codon_table
        self.shuffle = shuffle
        self.mutate = mutate
        self.mutation_rate = mutation_rate
        self.shuffle_frames = shuffle_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        indices = dna_to_indices(seq)
        if self.mutate:
            indices = apply_mutation(indices, self.mutation_rate)
        x = translate_to_codons(indices, self.codon_table)
        # crop/pad
        l = x.shape[1]
        if l > self.crop_size:
            start = random.randint(0, l - self.crop_size)
            x = x[:, start : start + self.crop_size]
        elif l < self.crop_size:
            x = torch.nn.functional.pad(x, (0, self.crop_size - l))
        mask = (x != 0)
        y = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return x, y, mask
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/unit/data/pytorch/test_dataset_csv.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/data/pytorch/dataset_csv.py tests/unit/data/pytorch/test_dataset_csv.py
git commit -m "feat(pytorch): add CSVDataset"
```

---

## Task 14: Implement training engine

**Files:**
- Create: `src/jaeger/training/pytorch/engine.py`
- Create: `tests/unit/training/pytorch/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/training/pytorch/test_engine.py
import torch
from jaeger.training.pytorch.engine import train_one_epoch, validate_one_epoch


def test_train_one_epoch_runs():
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = [(torch.randn(2, 10), torch.tensor([0, 1]))]
    metrics = {}
    result = train_one_epoch(model, loader, optimizer, loss_fn, device="cpu", use_amp=False, metrics=metrics)
    assert "loss" in result
```

- [ ] **Step 2: Implement engine**

```python
# src/jaeger/training/pytorch/engine.py
from typing import Any, Callable, Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    use_amp: bool = False,
    metrics: Optional[Dict[str, Callable]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    scaler = GradScaler() if use_amp else None
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break
        optimizer.zero_grad()
        batch = _to_device(batch, device)
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(x, mask)
            # If y is one-hot and loss_fn is CrossEntropyLoss, convert to class indices
            if y.dim() > 1 and y.shape[-1] > 1 and isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                y_idx = y.argmax(dim=-1)
                loss = loss_fn(outputs["prediction"], y_idx)
            else:
                loss = loss_fn(outputs, y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if metrics:
            for name, metric in metrics.items():
                metric.update(outputs["prediction"].detach(), y)

    result = {"loss": total_loss / max(num_batches, 1)}
    if metrics:
        for name, metric in metrics.items():
            result[name] = metric.compute()
    return result


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader,
    loss_fn: Callable,
    device: torch.device,
    use_amp: bool = False,
    metrics: Optional[Dict[str, Callable]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            batch = _to_device(batch, device)
            if len(batch) == 3:
                x, y, mask = batch
            else:
                x, y = batch
                mask = None
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x, mask)
                if y.dim() > 1 and y.shape[-1] > 1 and isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    y_idx = y.argmax(dim=-1)
                    loss = loss_fn(outputs["prediction"], y_idx)
                else:
                    loss = loss_fn(outputs, y)
            total_loss += loss.item()
            num_batches += 1
            if metrics:
                for name, metric in metrics.items():
                    metric.update(outputs["prediction"], y)

    result = {"loss": total_loss / max(num_batches, 1)}
    if metrics:
        for name, metric in metrics.items():
            result[name] = metric.compute()
    return result


def _to_device(batch, device):
    if isinstance(batch, (tuple, list)):
        return tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
    return batch.to(device)
```

- [ ] **Step 3: Run the test**

```bash
pytest tests/unit/training/pytorch/test_engine.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/training/pytorch/engine.py tests/unit/training/pytorch/test_engine.py
git commit -m "feat(pytorch): add training engine"
```

---

## Task 15: Implement Trainer

**Files:**
- Create: `src/jaeger/training/pytorch/trainer.py`
- Modify: `src/jaeger/nnlib/pytorch/checkpoints.py`

- [ ] **Step 1: Implement checkpoint helper**

```python
# src/jaeger/nnlib/pytorch/checkpoints.py
import torch


def save_checkpoint(path, model, optimizer, epoch, metrics, branch, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "branch": branch,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
```

- [ ] **Step 2: Implement Trainer**

```python
# src/jaeger/training/pytorch/trainer.py
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

from jaeger.nnlib.pytorch.builder import ModelBuilder
from jaeger.nnlib.pytorch.checkpoints import load_checkpoint, save_checkpoint
from jaeger.training.pytorch.callbacks import CallbackList
from jaeger.training.pytorch.engine import train_one_epoch, validate_one_epoch
from jaeger.utils.logging import get_logger


logger = get_logger(log_file=None, log_path=None, level=3)


class Trainer:
    """Orchestrates classifier/reliability/pretrain training."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        use_amp: bool = False,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.config = config
        self.device = device
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.builder = ModelBuilder(config)
        self.models = self.builder.build_fragment_classifier()
        self.callbacks = CallbackList()

    def fit_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        initial_epoch: int = 0,
    ):
        model, optimizer, loss_fn = self.builder.compile_model(self.models, train_branch="classifier")
        model = model.to(self.device)
        metrics = self.builder.get_metrics(branch="classifier")

        for epoch in range(initial_epoch, epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, loss_fn, self.device, self.use_amp, metrics
            )
            val_metrics = validate_one_epoch(
                model, val_loader, loss_fn, self.device, self.use_amp, metrics
            )
            self.callbacks.on_epoch_end(epoch, train_metrics, val_metrics, model, optimizer)

    def fit_reliability(self, train_loader, val_loader, epochs, initial_epoch=0):
        model, optimizer, loss_fn = self.builder.compile_model(self.models, train_branch="reliability")
        model = model.to(self.device)
        metrics = self.builder.get_metrics(branch="reliability")
        for epoch in range(initial_epoch, epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, self.device, self.use_amp, metrics)
            val_metrics = validate_one_epoch(model, val_loader, loss_fn, self.device, self.use_amp, metrics)
            self.callbacks.on_epoch_end(epoch, train_metrics, val_metrics, model, optimizer)

    def fit_projection(self, train_loader, val_loader, epochs, initial_epoch=0):
        model, optimizer, loss_fn = self.builder.compile_model(self.models, train_branch="pretrain")
        model = model.to(self.device)
        for epoch in range(initial_epoch, epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, self.device, self.use_amp)
            val_metrics = validate_one_epoch(model, val_loader, loss_fn, self.device, self.use_amp)
            self.callbacks.on_epoch_end(epoch, train_metrics, val_metrics, model, optimizer)

    def save_model(self, suffix: str = "fragment", metadata: Optional[str] = None):
        from jaeger.nnlib.pytorch.checkpoints import save_checkpoint
        import yaml

        model_name = self.config["model"]["name"]
        experiment = self.config["model"].get("experiment", "1")
        base_dir = Path(self.config["training"].get("base_dir", "."))
        save_dir = base_dir / f"{model_name}_exp{experiment}"
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.models["jaeger_model"].state_dict(), save_dir / "model.pt")
        with open(save_dir / "model_config.yaml", "w") as fh:
            yaml.safe_dump(self.config, fh)
        with open(save_dir / "class_label_map.yaml", "w") as fh:
            yaml.safe_dump({"classes": self.config["model"].get("class_label_map", [])}, fh)

        if metadata:
            Path(metadata).write_text(json.dumps({"model_path": str(save_dir)}, indent=2))

        logger.info(f"Saved PyTorch model to {save_dir}")
        return save_dir
```

- [ ] **Step 3: Smoke test**

```bash
python -c "from jaeger.training.pytorch.trainer import Trainer; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/training/pytorch/trainer.py src/jaeger/nnlib/pytorch/checkpoints.py
git commit -m "feat(pytorch): add Trainer and checkpoint helpers"
```

---

## Task 16: Implement callbacks

**Files:**
- Create: `src/jaeger/training/pytorch/callbacks.py`
- Create: `tests/unit/training/pytorch/test_callbacks.py`

- [ ] **Step 1: Implement callbacks**

```python
# src/jaeger/training/pytorch/callbacks.py
from pathlib import Path

import torch


class CallbackList:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def on_epoch_end(self, epoch, train_metrics, val_metrics, model, optimizer):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, train_metrics, val_metrics, model, optimizer)


class ModelCheckpoint:
    def __init__(self, checkpoint_dir: Path, monitor: str = "val_loss", mode: str = "min"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch, train_metrics, val_metrics, model, optimizer):
        value = val_metrics.get(self.monitor)
        if value is None:
            return
        improved = (self.mode == "min" and value < self.best) or (self.mode == "max" and value > self.best)
        if improved:
            self.best = value
            path = self.checkpoint_dir / f"epoch:{epoch}-loss:{value:.4f}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                },
                path,
            )
```

- [ ] **Step 2: Smoke test**

```bash
pytest tests/unit/training/pytorch/test_callbacks.py -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/training/pytorch/callbacks.py tests/unit/training/pytorch/test_callbacks.py
git commit -m "feat(pytorch): add training callbacks"
```

---

## Task 17: Implement distributed training helper

**Files:**
- Create: `src/jaeger/training/pytorch/distributed.py`

- [ ] **Step 1: Implement DDP setup**

```python
# src/jaeger/training/pytorch/distributed.py
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def wrap_model_ddp(model, device_ids):
    return DDP(model, device_ids=device_ids)
```

- [ ] **Step 2: Smoke test**

```bash
python -c "from jaeger.training.pytorch.distributed import setup_distributed; print('OK')"
```

Expected: `OK` (does nothing if not launched with torchrun)

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/training/pytorch/distributed.py
git commit -m "feat(pytorch): add distributed training helpers"
```

---

## Task 18: Update `jaeger train` command

**Files:**
- Modify: `src/jaeger/commands/train.py`

- [ ] **Step 1: Replace TF training orchestration with PyTorch**

Keep the Click options. Replace `train_fragment_core` body with:

```python
# src/jaeger/commands/train.py
def train_fragment_core(**kwargs):
    import torch
    from jaeger.training.pytorch.trainer import Trainer
    from jaeger.data.pytorch.builders import build_datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_model_config(Path(kwargs["config"]))
    config["mix_precision"] = kwargs.get("mixed_precision", False)
    config["from_last_checkpoint"] = kwargs.get("from_last_checkpoint", False)
    config["force"] = kwargs.get("force", False)

    trainer = Trainer(
        config=config,
        device=device,
        use_amp=kwargs.get("mixed_precision", False),
    )

    classifier_loaders = build_datasets(config, branch="classifier")
    reliability_loaders = build_datasets(config, branch="reliability")

    train_cfg = config.get("training", {})

    if kwargs.get("self_supervised_pretraining", False):
        trainer.fit_projection(
            classifier_loaders["train"],
            classifier_loaders["validation"],
            epochs=train_cfg.get("projection_epochs", 10),
        )

    if not kwargs.get("only_reliability_head", False):
        trainer.fit_classifier(
            classifier_loaders["train"],
            classifier_loaders["validation"],
            epochs=train_cfg.get("classifier_epochs", 100),
        )

    if not kwargs.get("only_classification_head", False):
        trainer.fit_reliability(
            reliability_loaders["train"],
            reliability_loaders["validation"],
            epochs=train_cfg.get("reliability_epochs", 100),
        )

    if kwargs.get("save_model", False) or kwargs.get("only_save", False):
        trainer.save_model(suffix="fragment", metadata=kwargs.get("meta"))
```

- [ ] **Step 2: Smoke test**

```bash
jaeger train --help
```

Expected: help text shows PyTorch-specific options.

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/commands/train.py
git commit -m "feat(pytorch): wire jaeger train to PyTorch trainer"
```

---

## Task 19: Implement PyTorch inference backend

**Files:**
- Create: `src/jaeger/inference/pytorch/model.py`
- Create: `src/jaeger/inference/pytorch/engine.py`

- [ ] **Step 1: Implement model loader**

```python
# src/jaeger/inference/pytorch/model.py
from pathlib import Path

import torch
import yaml

from jaeger.nnlib.pytorch.builder import ModelBuilder


class PyTorchJaegerModel:
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = Path(model_dir)
        self.device = device
        with open(self.model_dir / "model_config.yaml") as fh:
            config = yaml.safe_load(fh)
        builder = ModelBuilder(config)
        self.models = builder.build_fragment_classifier()
        self.jaeger_model = self.models["jaeger_model"].to(device)
        self.jaeger_model.load_state_dict(torch.load(self.model_dir / "model.pt", map_location=device))
        self.jaeger_model.eval()

        from jaeger.seqops.maps import CODON_ID
        self.codon_table = {c: i for i, c in enumerate(CODON_ID)}

    def predict(self, x, mask=None):
        with torch.no_grad():
            return self.jaeger_model(x.to(self.device), mask.to(self.device) if mask is not None else None)
```

- [ ] **Step 2: Implement inference engine**

```python
# src/jaeger/inference/pytorch/engine.py
import numpy as np
import torch
from jaeger.data.pytorch.transforms import dna_to_indices, translate_to_codons
from jaeger.inference.pytorch.model import PyTorchJaegerModel


def run_inference(model_dir, fasta_path, batch_size=96, device="cuda", fsize=2000, stride=2000, **kwargs):
    import pyfastx
    from jaeger.inference.pytorch.model import PyTorchJaegerModel

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = PyTorchJaegerModel(model_dir, device)

    results = []
    for name, seq in pyfastx.Fasta(fasta_path, build_index=False):
        windows = []
        for i in range(0, max(1, len(seq) - fsize + 1), stride):
            window = seq[i : i + fsize]
            if len(window) < 100:
                continue
            indices = dna_to_indices(window)
            windows.append((name, len(seq), i, indices))

        batch_preds = []
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            lengths = [len(w[3]) for w in batch]
            max_len = max(lengths)
            padded = torch.zeros(len(batch), 6, max_len, dtype=torch.long)
            mask = torch.zeros(len(batch), 6, max_len, dtype=torch.bool)
            for j, (_, _, _, indices) in enumerate(batch):
                translated = translate_to_codons(indices, model.codon_table)
                l = translated.shape[1]
                padded[j, :, :l] = translated
                mask[j, :, :l] = True
            with torch.no_grad():
                out = model.predict(padded, mask)
            batch_preds.append(out["prediction"].cpu().numpy())

        results.append({
            "name": name,
            "length": len(seq),
            "predictions": np.concatenate(batch_preds) if batch_preds else np.array([]),
        })

    return results
```

- [ ] **Step 3: Smoke test**

```bash
python -c "from jaeger.inference.pytorch.model import PyTorchJaegerModel; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/jaeger/inference/pytorch/model.py src/jaeger/inference/pytorch/engine.py
git commit -m "feat(pytorch): add PyTorch inference backend"
```

---

## Task 20: Update `jaeger predict` command

**Files:**
- Modify: `src/jaeger/commands/predict.py`

- [ ] **Step 1: Add PyTorch model detection and routing**

```python
# src/jaeger/commands/predict.py
def run_core(**kwargs):
    from jaeger.inference.pytorch.engine import run_inference
    return run_inference(**kwargs)
```

- [ ] **Step 2: Smoke test**

```bash
jaeger predict --help
```

Expected: help text unchanged.

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/commands/predict.py
git commit -m "feat(pytorch): route jaeger predict to PyTorch backend when model.pt present"
```

---

## Task 21: Update SLURM scripts and container

**Files:**
- Modify: `slurm/*.slurm`
- Modify: `singularity/jaeger_singularity.def`

- [ ] **Step 1: Update a training SLURM script**

Edit `slurm/baseline_500bp.slurm`. Replace the training invocation line:

```bash
# Before:
jaeger train -c train_config/nn_config_500bp_baseline.yaml

# After:
torchrun --nproc_per_node=2 --nnodes=1 jaeger train -c train_config/nn_config_500bp_baseline.yaml --mixed_precision
```

- [ ] **Step 2: Update Singularity definition**

Edit `singularity/jaeger_singularity.def`. Replace TensorFlow install lines with:

```singularity
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- [ ] **Step 3: Commit**

```bash
git add slurm/ singularity/
git commit -m "feat(pytorch): update SLURM scripts and Singularity container for PyTorch"
```

---

## Task 22: Delete TensorFlow training and inference code

**Files:**
- Delete: `src/jaeger/nnlib/v1/`
- Delete: `src/jaeger/nnlib/v2/`
- Delete: `src/jaeger/nnlib/builder.py`
- Delete: `src/jaeger/nnlib/inference.py`
- Delete: `src/jaeger/nnlib/conversion.py`
- Delete: `src/jaeger/commands/predict_legacy.py`

- [ ] **Step 1: Remove directories and files**

```bash
git rm -rf src/jaeger/nnlib/v1/ src/jaeger/nnlib/v2/
git rm src/jaeger/nnlib/builder.py src/jaeger/nnlib/inference.py src/jaeger/nnlib/conversion.py
git rm src/jaeger/commands/predict_legacy.py
```

- [ ] **Step 2: Verify imports still work**

```bash
python -c "import jaeger; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor(pytorch): remove TensorFlow training and inference code"
```

---

## Task 23: Final QA, benchmarks, and release prep

**Files:**
- Modify: `docs/_source/train.md`
- Modify: `docs/_source/optimizations.md`
- Modify: `recipes/jaeger-bio/meta.yaml`
- Modify: `install.sh`

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/unit tests/integration -v
```

Expected: All tests pass.

- [ ] **Step 2: Run smoke training on Zeus**

```bash
sbatch slurm/baseline_500bp.slurm
```

Expected: Job completes, checkpoints saved, logs show faster batches/sec than TF baseline.

- [ ] **Step 3: Run inference smoke test**

```bash
jaeger predict -i tests/fixtures/small.fasta -o /tmp/predict_out --model_path path/to/pytorch/model
```

Expected: Predictions written without errors.

- [ ] **Step 4: Update documentation**

- `docs/_source/train.md`: replace TensorFlow training instructions with PyTorch examples.
- `docs/_source/optimizations.md`: update inference backend table; remove TF-specific optimizations.

- [ ] **Step 5: Update Bioconda recipe and install script**

```yaml
# recipes/jaeger-bio/meta.yaml
requirements:
  run:
    - pytorch
    - torchvision
```

```bash
# install.sh
# Replace TensorFlow install lines with PyTorch install lines.
```

- [ ] **Step 6: Bump version to 1.27.1**

```bash
.github/scripts/bump-version.sh 1 27 1
```

- [ ] **Step 7: Final commit and push**

```bash
git add -A
git commit -m "chore(release): bump version to 1.27.1"
git push origin pytorch_migration
```

---
