# Short-Fragment Architecture Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in `MultiScaleConv1D`, `LocalAttention`, and masked global pooling support to Jaeger's v2 layer library and builder, plus a new training config and tests, so models can better handle 300 bp–2000 bp DNA fragments.

**Architecture:** Add three new layer primitives to `src/jaeger/nnlib/v2/layers.py`, register them in `DynamicModelBuilder`, expose them through a new YAML config, and cover each with unit tests plus one integration test that builds the full model.

**Tech Stack:** Python 3.11+, TensorFlow 2.21+/Keras 3.12+, pytest, YAML.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/jaeger/nnlib/v2/layers.py` | Contains new `MultiScaleConv1D`, `LocalAttention`, and an updated `MaskedGlobalAvgPooling` class. |
| `src/jaeger/nnlib/builder.py` | Registers `multi_scale_conv`, `local_attention`, and `masked_average` pooler in `DynamicModelBuilder`. |
| `train_config/nn_config_300-2000bp_multiscale.yaml` | New training config that uses the new layers. |
| `tests/unit/test_nnlib_v2_layers_short_fragment.py` | Unit tests for the three new layer primitives. |
| `tests/integration/test_builder_short_fragment_config.py` | Integration test that loads the new config and builds/compiles/runs the model. |

---

## Task 1: Add `MultiScaleConv1D` layer

**Files:**
- Modify: `src/jaeger/nnlib/v2/layers.py`
- Test: `tests/unit/test_nnlib_v2_layers_short_fragment.py`

### Step 1.1: Write the failing test

Append the following test class to `tests/unit/test_nnlib_v2_layers_short_fragment.py` (create the file if it does not exist):

```python
"""Tests for short-fragment layers."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2 import layers


class TestMultiScaleConv1D:
    def test_concat_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                {"filters": 16, "kernel_size": 5, "dilation_rate": 1},
                {"filters": 8, "kernel_size": 3, "dilation_rate": 3},
            ],
            merge="concat",
        )
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 32, 32]

    def test_add_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                {"filters": 8, "kernel_size": 5, "dilation_rate": 1},
            ],
            merge="add",
        )
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 32, 8]

    def test_add_mismatched_filters_raises(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3},
                {"filters": 16, "kernel_size": 3},
            ],
            merge="add",
        )
        with pytest.raises(ValueError):
            layer(x, mask=mask)

    def test_mask_propagation(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(
            mask, [[0, 0, 15], [0, 0, 16]], [False, False]
        )
        layer = layers.MultiScaleConv1D(
            branches=[{"filters": 8, "kernel_size": 3, "padding": "same"}],
            merge="concat",
        )
        _ = layer(x, mask=mask)
        assert layer.compute_mask(x, mask=mask) is not None
```

### Step 1.2: Run the failing test

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py::TestMultiScaleConv1D -v
```

Expected: 4 failures (`MultiScaleConv1D` not defined).

### Step 1.3: Implement `MultiScaleConv1D`

Append the following class to `src/jaeger/nnlib/v2/layers.py` near the other custom layers (after `MaskedConv1D`):

```python
class MultiScaleConv1D(tf.keras.layers.Layer):
    """Parallel masked 1D convolutions at multiple scales.

    Input shape: (batch, frames, length, channels)
    Output shape: (batch, frames, length, total_filters) for merge="concat", or
                  (batch, frames, length, branch_filters) for merge="add".

    Each branch is configured by a dict passed to `MaskedConv1D`. Branch
    sequence lengths must align, which is enforced by using ``padding="same"``
    and ``strides=1`` by default.
    """

    def __init__(
        self,
        branches: list[dict],
        merge: str = "concat",
        kernel_initializer: str | tf.keras.initializers.Initializer = "glorot_uniform",
        kernel_regularizer: str | tf.keras.regularizers.Regularizer | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.branches = list(branches)
        self.merge = merge.lower()
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.supports_masking = True
        self._convs: list[MaskedConv1D] = []

        if self.merge not in {"concat", "add"}:
            raise ValueError(f"merge must be 'concat' or 'add', got {merge!r}")

    def _resolve(self, value, kind: str):
        """Convert string regularizer/initializer names to objects."""
        if kind == "regularizer" and isinstance(value, str):
            return tf.keras.regularizers.get(value)
        if kind == "initializer" and isinstance(value, str):
            return tf.keras.initializers.get(value)
        return value

    def build(self, input_shape):
        if self.merge == "add":
            filters = [b.get("filters") for b in self.branches]
            if len(set(filters)) != 1:
                raise ValueError(
                    "All branches must have the same filters when merge='add'"
                )

        for i, cfg in enumerate(self.branches):
            branch_cfg = dict(cfg)
            branch_cfg.setdefault("padding", "same")
            branch_cfg.setdefault("strides", 1)
            branch_cfg.setdefault("name", f"{self.name}_branch_{i}")
            branch_cfg.setdefault(
                "kernel_initializer",
                self._resolve(self.kernel_initializer, "initializer"),
            )
            branch_cfg.setdefault(
                "kernel_regularizer",
                self._resolve(self.kernel_regularizer, "regularizer"),
            )
            branch_cfg.setdefault("use_bias", self.use_bias)
            self._convs.append(MaskedConv1D(**branch_cfg))
        super().build(input_shape)

    def call(self, inputs, mask=None):
        outputs = [conv(inputs, mask=mask) for conv in self._convs]
        if self.merge == "concat":
            x = tf.concat(outputs, axis=-1)
        else:
            x = tf.add_n(outputs)
        self._output_mask = mask
        return x

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        if self.merge == "concat":
            total = sum(b.get("filters", 0) for b in self.branches)
        else:
            total = self.branches[0].get("filters", 0)
        return (input_shape[0], input_shape[1], input_shape[2], total)

    def get_config(self):
        config = super().get_config()

        def _serialize(value, kind):
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if kind == "initializer":
                return tf.keras.initializers.serialize(value)
            if kind == "regularizer":
                return tf.keras.regularizers.serialize(value)
            return value

        config.update(
            {
                "branches": self.branches,
                "merge": self.merge,
                "kernel_initializer": _serialize(
                    self._resolve(self.kernel_initializer, "initializer"), "initializer"
                ),
                "kernel_regularizer": _serialize(
                    self._resolve(self.kernel_regularizer, "regularizer"), "regularizer"
                ),
                "use_bias": self.use_bias,
            }
        )
        return config
```

### Step 1.4: Run the test

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py::TestMultiScaleConv1D -v
```

Expected: 4 passes.

### Step 1.5: Commit

```bash
git add src/jaeger/nnlib/v2/layers.py tests/unit/test_nnlib_v2_layers_short_fragment.py
git commit -m "feat(nnlib): add MultiScaleConv1D layer for short fragments"
```

---

## Task 2: Register `multi_scale_conv` in the builder

**Files:**
- Modify: `src/jaeger/nnlib/builder.py`

### Step 2.1: Write the failing integration test

Create `tests/integration/test_builder_short_fragment_config.py` with a minimal failing test:

```python
"""Integration test for short-fragment config support."""

from __future__ import annotations

import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder


def test_multiscale_layer_registered():
    config = {
        "model": {
            "name": "test_multiscale",
            "experiment": "test",
            "seed": 42,
            "classifier_out_dim": 3,
            "reliability_out_dim": 0,
            "class_label_map": [
                {"class": "chromosome", "label": 0},
                {"class": "virus", "label": 1},
                {"class": "plasmid", "label": 2},
            ],
            "embedding": {
                "use_embedding_layer": True,
                "input_type": "translated",
                "strands": 2,
                "frames": 6,
                "input_shape": [6, None],
                "embedding_size": 64,
            },
            "string_processor": {
                "data_format": "numpy_full",
                "seq_onehot": False,
                "codon": "CODON",
                "codon_id": "CODON_ID",
                "crop_size": 500,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "hidden_layers": [
                    {
                        "name": "multi_scale_conv",
                        "config": {
                            "branches": [
                                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                                {"filters": 8, "kernel_size": 5, "dilation_rate": 1},
                            ],
                            "merge": "concat",
                        },
                    },
                    {"name": "masked_batchnorm", "config": {"return_nmd": False}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "average",
            },
            "classifier": {
                "input_shape": 16,
                "hidden_layers": [
                    {"name": "dense", "config": {"units": 3, "activation": None, "dtype": "float32"}}
                ],
            },
        },
        "training": {
            "optimizer": "adam",
            "optimizer_params": {"learning_rate": 0.001},
            "loss_classifier": "categorical_crossentropy",
            "loss_params_classifier": {"from_logits": True},
            "metrics_classifier": [{"name": "categorical_accuracy", "params": None}],
            "callbacks": {"directories": []},
            "model_saving": {"path": "/tmp/jaeger_test", "save_weights": False, "save_exec_graph": False},
            "fragment_classifier_data": {
                "train": [{"class": ["chromosome"], "label": [0], "path": []}]
            },
        },
        "config_path": "/tmp/jaeger_test_config.yaml",
    }
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    assert "jaeger_model" in models
```

### Step 2.2: Run the failing test

```bash
python -m pytest tests/integration/test_builder_short_fragment_config.py::test_multiscale_layer_registered -v
```

Expected: Failure (`Unknown layer type: multi_scale_conv`).

### Step 2.3: Register the layer

In `src/jaeger/nnlib/builder.py`, update the `self._layers` dict inside `DynamicModelBuilder.__init__`:

```python
self._layers = {
    "masked_conv1d": MaskedConv1D,
    "conv1d": tf.keras.layers.Conv1D,
    "masked_batchnorm": MaskedBatchNorm,
    "masked_layernorm": MaskedLayerNormalization,
    "layernorm": tf.keras.layers.LayerNormalization,
    "batchnorm": tf.keras.layers.BatchNormalization,
    "transformer_encoder": TransformerEncoder,
    "cross_frame_attention": CrossFrameAttention,
    "axial_attention": AxialAttention,
    "residual_block": ResidualBlock_wrapper,
    "dense": tf.keras.layers.Dense,
    "activation": tf.keras.layers.Activation,
    "dropout": tf.keras.layers.Dropout,
    "crop": tf.keras.layers.Cropping2D,
    "multi_scale_conv": MultiScaleConv1D,  # NEW
}
```

Also update the imports at the top of `builder.py`:

```python
from jaeger.nnlib.v2.layers import (
    AxialAttention,
    CrossFrameAttention,
    GatedFrameGlobalMaxPooling,
    MaskedBatchNorm,
    MaskedLayerNormalization,
    MaskedConv1D,
    MetricModel,
    MultiScaleConv1D,  # NEW
    ResidualBlock_wrapper,
    TransformerEncoder,
)
```

### Step 2.4: Run the test

```bash
python -m pytest tests/integration/test_builder_short_fragment_config.py::test_multiscale_layer_registered -v
```

Expected: Pass.

### Step 2.5: Commit

```bash
git add src/jaeger/nnlib/builder.py tests/integration/test_builder_short_fragment_config.py
git commit -m "feat(nnlib): register multi_scale_conv in DynamicModelBuilder"
```

---

## Task 3: Add masked global average pooler support

**Files:**
- Modify: `src/jaeger/nnlib/v2/layers.py`
- Modify: `src/jaeger/nnlib/builder.py`
- Test: `tests/unit/test_nnlib_v2_layers_short_fragment.py`

### Note on existing code

`MaskedGlobalAvgPooling` already exists in `src/jaeger/nnlib/v2/layers.py`. This task only needs to add `get_config` to it and register it as `masked_average` in the builder.

### Step 3.1: Write the failing test

Append to `tests/unit/test_nnlib_v2_layers_short_fragment.py`:

```python
class TestMaskedGlobalAvgPooling:
    def test_masked_average_ignores_padding(self):
        x = tf.constant([
            [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]],
            [[[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]],
        ])  # (2, 1, 3, 2)
        mask = tf.constant([
            [[True, True, False]],
            [[True, False, False]],
        ])  # (2, 1, 3)
        layer = layers.MaskedGlobalAvgPooling()
        out = layer(x, mask=mask)
        expected = tf.constant([[2.0, 3.0], [5.0, 6.0]])
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_pooler_registered_in_builder(self):
        from jaeger.nnlib.builder import DynamicModelBuilder
        builder = DynamicModelBuilder({"model": {}, "training": {"callbacks": {"directories": []}}})
        pooler = builder._get_pooler("masked_average")
        assert pooler is layers.MaskedGlobalAvgPooling
```

### Step 3.2: Run the failing test

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py::TestMaskedGlobalAvgPooling -v
```

Expected: 2 failures (`get_config` missing or pooler not registered).

### Step 3.3: Update `MaskedGlobalAvgPooling` and register it

Update the existing `MaskedGlobalAvgPooling` class in `src/jaeger/nnlib/v2/layers.py`:

```python
class MaskedGlobalAvgPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True  # NEW

    def call(self, inputs, mask=None):
        ...  # existing implementation unchanged

    def compute_output_shape(self, input_shape):
        ...  # existing implementation unchanged

    def get_config(self):  # NEW
        return super().get_config()
```

In `src/jaeger/nnlib/builder.py`, update `_get_pooler`:

```python
def _get_pooler(self, name: str):
    poolers = {
        "max": tf.keras.layers.GlobalMaxPooling2D,
        "average": tf.keras.layers.GlobalAveragePooling2D,
        "gatedframe": GatedFrameGlobalMaxPooling,
        "masked_average": MaskedGlobalAvgPooling,  # NEW
    }
    return poolers[name]
```

Also add `MaskedGlobalAvgPooling` to the imports from `jaeger.nnlib.v2.layers` in `builder.py`:

```python
from jaeger.nnlib.v2.layers import (
    AxialAttention,
    CrossFrameAttention,
    GatedFrameGlobalMaxPooling,
    MaskedBatchNorm,
    MaskedGlobalAvgPooling,  # NEW
    MaskedLayerNormalization,
    MaskedConv1D,
    MetricModel,
    MultiScaleConv1D,
    ResidualBlock_wrapper,
    TransformerEncoder,
)
```

### Step 3.4: Run the test

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py::TestMaskedGlobalAvgPooling -v
```

Expected: 2 passes.

### Step 3.5: Commit

```bash
git add src/jaeger/nnlib/v2/layers.py src/jaeger/nnlib/builder.py tests/unit/test_nnlib_v2_layers_short_fragment.py
git commit -m "feat(nnlib): expose MaskedGlobalAvgPooling as masked_average pooler"
```

---

## Task 4: Add `LocalAttention` layer

**Files:**
- Modify: `src/jaeger/nnlib/v2/layers.py`
- Test: `tests/unit/test_nnlib_v2_layers_short_fragment.py`

### Step 4.1: Write the failing test

Append to `tests/unit/test_nnlib_v2_layers_short_fragment.py`:

```python
class TestLocalAttention:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 64, 16))
        mask = tf.ones((2, 6, 64), dtype=tf.bool)
        layer = layers.LocalAttention(
            embed_dim=16, num_heads=4, feed_forward_dim=32, window_size=16, num_blocks=2
        )
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 64, 16]

    def test_gradient_flow(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.LocalAttention(
            embed_dim=8, num_heads=2, feed_forward_dim=16, window_size=8, num_blocks=1
        )
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = layer(x, mask=mask)
            loss = tf.reduce_mean(out ** 2)
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.all(np.isfinite(grads.numpy()))

    def test_invalid_config_raises(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.LocalAttention(
            embed_dim=8, num_heads=3, feed_forward_dim=16, window_size=8
        )
        with pytest.raises(ValueError):
            layer(x, mask=mask)
```

### Step 4.2: Run the failing test

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py::TestLocalAttention -v
```

Expected: 3 failures (`LocalAttention` not defined).

### Step 4.3: Implement `LocalAttention`

Append the following class to `src/jaeger/nnlib/v2/layers.py` near the other attention layers:

```python
class LocalAttention(tf.keras.layers.Layer):
    """Windowed self-attention along the sequence-length axis.

    Input shape: (batch, frames, length, channels)
    Output shape: (batch, frames, length, channels)

    Each position attends only to neighbors within ``window_size // 2`` on
    either side. This is cheaper and more appropriate than full self-attention
    for short sequences where long-range dependencies are noisy.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        window_size: int,
        dropout_rate: float = 0.1,
        num_blocks: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.supports_masking = True

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                {
                    "ln1": tf.keras.layers.LayerNormalization(
                        epsilon=1e-6, name=f"{self.name}_ln1_{i}"
                    ),
                    "mha": tf.keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=embed_dim // num_heads,
                        dropout=dropout_rate,
                        name=f"{self.name}_mha_{i}",
                    ),
                    "ln2": tf.keras.layers.LayerNormalization(
                        epsilon=1e-6, name=f"{self.name}_ln2_{i}"
                    ),
                    "ffn1": tf.keras.layers.Dense(
                        feed_forward_dim,
                        activation="gelu",
                        name=f"{self.name}_ffn1_{i}",
                    ),
                    "ffn2": tf.keras.layers.Dense(
                        embed_dim, name=f"{self.name}_ffn2_{i}"
                    ),
                }
            )

    def _local_attention_mask(self, length: tf.Tensor):
        """Return a boolean (1, length, length) mask for the local window."""
        half_window = self.window_size // 2
        row = tf.range(length)[:, None]
        col = tf.range(length)[None, :]
        mask = tf.abs(row - col) <= half_window
        return mask[None, ...]

    def call(self, inputs, mask=None, training=None):
        shape = tf.shape(inputs)
        batch = shape[0]
        frames = shape[1]
        length = shape[2]
        channels = shape[3]

        # Reshape to (batch*frames, length, channels) for length-wise attention.
        x = tf.reshape(inputs, [batch * frames, length, channels])

        attn_mask = self._local_attention_mask(length)

        if mask is not None:
            # mask: (batch, frames, length) -> (batch*frames, length)
            seq_mask = tf.reshape(mask, [batch * frames, length])
            # Combine local window mask with sequence validity mask.
            key_mask = seq_mask[:, None, :]  # (B*F, 1, L)
            attn_mask = attn_mask & key_mask

        for block in self.blocks:
            x_norm = block["ln1"](x)
            attn = block["mha"](
                x_norm,
                x_norm,
                attention_mask=attn_mask,
                training=training,
            )
            x = x + attn

            x_norm = block["ln2"](x)
            ffn = block["ffn2"](block["ffn1"](x_norm))
            x = x + ffn

        return tf.reshape(x, [batch, frames, length, channels])

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "feed_forward_dim": self.feed_forward_dim,
                "window_size": self.window_size,
                "dropout_rate": self.dropout_rate,
                "num_blocks": self.num_blocks,
            }
        )
        return config
```

### Step 4.4: Run the test

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py::TestLocalAttention -v
```

Expected: 3 passes.

### Step 4.5: Commit

```bash
git add src/jaeger/nnlib/v2/layers.py tests/unit/test_nnlib_v2_layers_short_fragment.py
git commit -m "feat(nnlib): add LocalAttention layer for windowed self-attention"
```

---

## Task 5: Register `local_attention` in the builder

**Files:**
- Modify: `src/jaeger/nnlib/builder.py`
- Test: `tests/integration/test_builder_short_fragment_config.py`

### Step 5.1: Update the integration test

Replace the representation_learner section in `tests/integration/test_builder_short_fragment_config.py` with one that uses `local_attention` and `masked_average` pooling:

```python
            "representation_learner": {
                "hidden_layers": [
                    {
                        "name": "multi_scale_conv",
                        "config": {
                            "branches": [
                                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                                {"filters": 8, "kernel_size": 5, "dilation_rate": 1},
                            ],
                            "merge": "concat",
                        },
                    },
                    {"name": "masked_batchnorm", "config": {"return_nmd": False}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                    {
                        "name": "local_attention",
                        "config": {
                            "embed_dim": 16,
                            "num_heads": 2,
                            "feed_forward_dim": 32,
                            "window_size": 16,
                            "dropout_rate": 0.1,
                            "num_blocks": 1,
                        },
                    },
                    {"name": "masked_layernorm"},
                ],
                "pooling": "masked_average",
            },
            "classifier": {
                "input_shape": 16,
                ...
            },
```

### Step 5.2: Run the failing test

```bash
python -m pytest tests/integration/test_builder_short_fragment_config.py -v
```

Expected: Failure (`Unknown layer type: local_attention`).

### Step 5.3: Register the layer

Add `"local_attention": LocalAttention` to `self._layers` in `src/jaeger/nnlib/builder.py`:

```python
self._layers = {
    ...
    "local_attention": LocalAttention,  # NEW
    ...
}
```

Also add `LocalAttention` to the imports from `jaeger.nnlib.v2.layers` at the top of `builder.py`:

```python
from jaeger.nnlib.v2.layers import (
    AxialAttention,
    CrossFrameAttention,
    GatedFrameGlobalMaxPooling,
    LocalAttention,  # NEW
    MaskedBatchNorm,
    MaskedGlobalAvgPooling,  # added in Task 3
    MaskedLayerNormalization,
    MaskedConv1D,
    MetricModel,
    MultiScaleConv1D,
    ResidualBlock_wrapper,
    TransformerEncoder,
)
```

### Step 5.4: Run the test

```bash
python -m pytest tests/integration/test_builder_short_fragment_config.py -v
```

Expected: Pass.

### Step 5.5: Commit

```bash
git add src/jaeger/nnlib/builder.py tests/integration/test_builder_short_fragment_config.py
git commit -m "feat(nnlib): register local_attention and masked_average in builder"
```

---

## Task 6: Create the new training config

**Files:**
- Create: `train_config/nn_config_300-2000bp_multiscale.yaml`

### Step 6.1: Copy and modify the axial config

Copy `train_config/nn_config_500bp_axial.yaml` to `train_config/nn_config_300-2000bp_multiscale.yaml` and update only the `model.name`, `model.experiment`, and `representation_learner` sections.

Use the following representation_learner block:

```yaml
  representation_learner:
    hidden_layers:
    - name: multi_scale_conv
      config:
        branches:
          - filters: 32
            kernel_size: 3
            dilation_rate: 1
          - filters: 32
            kernel_size: 5
            dilation_rate: 1
          - filters: 32
            kernel_size: 3
            dilation_rate: 3
        merge: concat
    - name: masked_batchnorm
      config:
        return_nmd: false
    - name: activation
      config:
        activation: gelu
    - name: local_attention
      config:
        embed_dim: 96
        num_heads: 4
        feed_forward_dim: 128
        window_size: 32
        dropout_rate: 0.1
        num_blocks: 2
    - name: masked_layernorm
    pooling: masked_average
```

Set `classifier.input_shape: 96` to match the concatenated multi-scale output.

Update `model.name` and `model.experiment` to `jaeger_300-2000bp_multiscale`.

### Step 6.2: Validate the config loads

```python
from pathlib import Path
from jaeger.utils.misc import load_model_config
from jaeger.nnlib.builder import DynamicModelBuilder
import tempfile

cfg = load_model_config(Path("train_config/nn_config_300-2000bp_multiscale.yaml"))
with tempfile.TemporaryDirectory() as tmp:
    cfg["training"]["model_saving"]["path"] = tmp
    cfg["training"]["callbacks"]["directories"] = []
    builder = DynamicModelBuilder(cfg)
    models = builder.build_fragment_classifier()
    print(models["jaeger_model"].output_shape)
```

Expected: No exception and a dict-shaped output.

### Step 6.3: Commit

```bash
git add train_config/nn_config_300-2000bp_multiscale.yaml
git commit -m "feat(config): add 300-2000bp multiscale short-fragment config"
```

---

## Task 7: Add integration test for the full config

**Files:**
- Modify: `tests/integration/test_builder_short_fragment_config.py`

### Step 7.1: Add the full-config test

Append to `tests/integration/test_builder_short_fragment_config.py`:

```python
from pathlib import Path
import tempfile

from jaeger.utils.misc import load_model_config


def test_full_multiscale_config_builds():
    config_path = Path(__file__).parents[2] / "train_config" / "nn_config_300-2000bp_multiscale.yaml"
    cfg = load_model_config(config_path)
    cfg["config_path"] = str(config_path)
    with tempfile.TemporaryDirectory() as tmp:
        cfg["training"]["model_saving"]["path"] = tmp
        cfg["training"]["callbacks"]["directories"] = []
        # Reduce data paths to empty to avoid needing real CSVs/NPZs.
        cfg["training"]["fragment_classifier_data"]["train"][0]["path"] = []
        cfg["training"]["fragment_classifier_data"]["validation"][0]["path"] = []
        builder = DynamicModelBuilder(cfg)
        models = builder.build_fragment_classifier()
        model = models["jaeger_model"]
        # Synthetic forward pass: (batch, frames, length, channels)
        x = tf.random.normal((1, 6, 500, 64))
        out = model(x)
        assert "prediction" in out
        assert out["prediction"].shape.as_list() == [1, 3]
```

### Step 7.2: Run the test

```bash
python -m pytest tests/integration/test_builder_short_fragment_config.py -v
```

Expected: Pass.

### Step 7.3: Commit

```bash
git add tests/integration/test_builder_short_fragment_config.py
git commit -m "test(integration): verify full multiscale config builds and runs"
```

---

## Task 8: Run the full test suite and lint

### Step 8.1: Run targeted tests

```bash
python -m pytest tests/unit/test_nnlib_v2_layers_short_fragment.py tests/integration/test_builder_short_fragment_config.py -v
```

Expected: All tests pass.

### Step 8.2: Run ruff

```bash
ruff check src/jaeger/nnlib/v2/layers.py src/jaeger/nnlib/builder.py tests/unit/test_nnlib_v2_layers_short_fragment.py tests/integration/test_builder_short_fragment_config.py
ruff format --check src/jaeger/nnlib/v2/layers.py src/jaeger/nnlib/builder.py tests/unit/test_nnlib_v2_layers_short_fragment.py tests/integration/test_builder_short_fragment_config.py
```

Expected: No lint errors. If format check fails, run `ruff format` on the files.

### Step 8.3: Run the broader unit/integration suite (optional but recommended)

```bash
python -m pytest tests/unit tests/integration -v --tb=short
```

Expected: Existing tests still pass; any failures unrelated to this work should be documented.

### Step 8.4: Commit

```bash
git commit -m "style: ruff formatting for short-fragment layers" --allow-empty
```

---

## Self-Review Checklist

Before handoff, verify:

- [ ] `MultiScaleConv1D` handles `concat` and `add` merge modes and propagates masks.
- [ ] `LocalAttention` enforces `embed_dim % num_heads == 0` and `window_size > 0`.
- [ ] `MaskedGlobalAvgPooling` is registered as `masked_average` and `average` remains unchanged.
- [ ] All three layers are registered in `DynamicModelBuilder`.
- [ ] New config uses the new layers and pools with `masked_average`.
- [ ] Unit tests cover shapes, mask behavior, and invalid configs.
- [ ] Integration test builds the full model from the real YAML file.
- [ ] No placeholders, TBDs, or missing file paths remain in this plan.
