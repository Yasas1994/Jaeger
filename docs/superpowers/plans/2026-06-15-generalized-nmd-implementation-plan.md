# Generalized NMD Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone `NMDLayer`, a configurable `NMDMerge`, and builder wiring so that multiple Neural Mean Discrepancy vectors can be collected from any depth of the representation learner and merged before the reliability head.

**Architecture:** A new `src/jaeger/nnlib/v2/nmd.py` module holds `NMDLayer` and `NMDMerge`. The builder registers `nmd` as a layer name, collects NMD tensors during `_build_block`, applies the configured merge before returning from the representation learner, and feeds the merged tensor to the reliability head. Existing `MaskedBatchNorm(return_nmd=True)` behavior is preserved.

**Tech Stack:** Python 3.13, TensorFlow 2.21+/Keras 3.12+, pytest.

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/jaeger/nnlib/v2/nmd.py` (create) | `NMDLayer` and `NMDMerge` Keras layers. |
| `src/jaeger/nnlib/builder.py` (modify) | Register `nmd` layer, pass merge config through `_build_block`, wire merged NMD to reliability head, validate shapes. |
| `tests/unit/test_nnlib_v2_nmd.py` (create) | Unit tests for `NMDLayer` and `NMDMerge`. |
| `tests/integration/test_builder_nmd_merge.py` (create) | Builder integration tests for multiple NMD layers and all merge modes. |
| `train_config/nn_config_500bp_nmd_merge.yaml` (create) | Example config showing two `nmd` layers and a `reliability_model.merge` block. |

---

## Task 1: Create `NMDLayer`

**Files:**
- Create: `src/jaeger/nnlib/v2/nmd.py`
- Test: `tests/unit/test_nnlib_v2_nmd.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2.nmd import NMDLayer


class TestNMDLayer:
    def test_output_shape_without_mask(self):
        x = tf.random.normal((2, 6, 32, 8))
        layer = NMDLayer()
        out = layer(x)
        assert list(out.shape) == [2, 8]

    def test_output_shape_with_mask(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = NMDLayer()
        out = layer(x, mask=mask)
        assert list(out.shape) == [2, 8]

    def test_get_config_roundtrip(self):
        layer = NMDLayer(epsilon=1e-3, momentum=0.95, dtype="float32")
        config = layer.get_config()
        restored = NMDLayer.from_config(config)
        assert restored.epsilon == 1e-3
        assert restored.momentum == 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nnlib_v2_nmd.py::TestNMDLayer -v`
Expected: `ImportError: cannot import name 'NMDLayer'`

- [ ] **Step 3: Write minimal implementation**

Create `src/jaeger/nnlib/v2/nmd.py`:

```python
"""Neural Mean Discrepancy layers."""

from __future__ import annotations

import tensorflow as tf


class NMDLayer(tf.keras.layers.Layer):
    """Compute a per-example channel mean-discrepancy vector.

    The output is the per-example channel mean (mask-aware) minus a reference
    mean. During training the reference mean is the current batch mean and the
    layer's moving mean is updated; during inference the moving mean is used.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        momentum: float = 0.9,
        dtype=None,
        **kwargs,
    ):
        if dtype is not None:
            kwargs["dtype"] = dtype
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.supports_masking = True

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        if channel_dim is None:
            raise ValueError("The last (channel) dimension must be defined.")
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(channel_dim,),
            initializer="zeros",
            trainable=False,
            dtype=self.variable_dtype,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        x = tf.cast(inputs, tf.float32)
        ndims = x.shape.rank
        if ndims is None:
            raise ValueError("Input rank must be statically known for NMDLayer.")

        example_axes = list(range(1, max(ndims - 1, 1)))

        if mask is not None:
            mask_f = tf.cast(mask, tf.float32)
            if mask_f.shape.rank is None or mask_f.shape.rank < ndims:
                mask_f = tf.expand_dims(mask_f, axis=-1)
            masked_inputs = x * mask_f
            per_ex_sum = tf.reduce_sum(masked_inputs, axis=example_axes)
            per_ex_count = tf.reduce_sum(mask_f, axis=example_axes) + self.epsilon
            mean_channel = per_ex_sum / per_ex_count
        else:
            mean_channel = tf.reduce_mean(x, axis=example_axes)

        mm = tf.cast(self.moving_mean, tf.float32)
        if training:
            mean_batch = tf.reduce_mean(mean_channel, axis=0)
            new_mm = self.momentum * mm + (1.0 - self.momentum) * mean_batch
            self.moving_mean.assign(tf.cast(new_mm, self.moving_mean.dtype))
            mean_to_use = mean_batch
        else:
            mean_to_use = mm

        nmd_f32 = mean_channel - mean_to_use
        return tf.cast(nmd_f32, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
            }
        )
        return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nnlib_v2_nmd.py::TestNMDLayer -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/v2/nmd.py tests/unit/test_nnlib_v2_nmd.py
git commit -m "feat: add standalone NMDLayer"
```

---

## Task 2: Create `NMDMerge`

**Files:**
- Modify: `src/jaeger/nnlib/v2/nmd.py`
- Test: `tests/unit/test_nnlib_v2_nmd.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_nnlib_v2_nmd.py`:

```python
class TestNMDMerge:
    @pytest.fixture
    def nmd_tensors(self):
        return [
            tf.random.normal((4, 8)),
            tf.random.normal((4, 16)),
        ]

    def test_concat(self, nmd_tensors):
        merged = NMDMerge(mode="concat")(nmd_tensors)
        assert list(merged.shape) == [4, 24]

    def test_sum(self, nmd_tensors):
        merged = NMDMerge(mode="sum", target_dim=8)(nmd_tensors)
        assert list(merged.shape) == [4, 8]

    def test_mean(self, nmd_tensors):
        merged = NMDMerge(mode="mean", target_dim=8)(nmd_tensors)
        assert list(merged.shape) == [4, 8]

    def test_max(self, nmd_tensors):
        merged = NMDMerge(mode="max", target_dim=8)(nmd_tensors)
        assert list(merged.shape) == [4, 8]

    def test_weighted(self, nmd_tensors):
        merged = NMDMerge(mode="weighted", target_dim=8)(nmd_tensors)
        assert list(merged.shape) == [4, 8]

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            NMDMerge(mode="unsupported")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nnlib_v2_nmd.py::TestNMDMerge -v`
Expected: `NameError: name 'NMDMerge' is not defined`

- [ ] **Step 3: Write minimal implementation**

Append to `src/jaeger/nnlib/v2/nmd.py`:

```python
class NMDMerge(tf.keras.layers.Layer):
    """Merge a list of NMD vectors into a single tensor."""

    def __init__(
        self,
        mode: str = "concat",
        axis: int = -1,
        target_dim: int | None = None,
        projection_kwargs: dict | None = None,
        dtype=None,
        **kwargs,
    ):
        if dtype is not None:
            kwargs["dtype"] = dtype
        super().__init__(**kwargs)
        if mode not in {"concat", "sum", "mean", "max", "weighted"}:
            raise ValueError(f"Unsupported NMD merge mode: {mode}")
        self.mode = mode
        self.axis = axis
        self.target_dim = target_dim
        self.projection_kwargs = projection_kwargs or {}

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self._num_inputs = len(input_shape)
        dims = [int(shape[-1]) for shape in input_shape]

        if self.mode == "concat":
            self._output_dim = sum(dims)
        else:
            if self.target_dim is None:
                if len(set(dims)) == 1:
                    self.target_dim = dims[0]
                else:
                    raise ValueError(
                        f"target_dim is required for merge mode '{self.mode}' "
                        "when NMD channel dimensions differ."
                    )
            self._output_dim = self.target_dim
            self.projections = [
                tf.keras.layers.Dense(
                    self.target_dim,
                    use_bias=False,
                    name=f"proj_{i}",
                    **self.projection_kwargs,
                )
                for i in range(self._num_inputs)
            ]
            if self.mode == "weighted":
                self.layer_weights = self.add_weight(
                    name="layer_weights",
                    shape=(self._num_inputs,),
                    initializer="ones",
                    trainable=True,
                )
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.mode == "concat":
            return tf.concat(inputs, axis=self.axis)

        projected = [proj(inp) for proj, inp in zip(self.projections, inputs)]

        if self.mode == "sum":
            return tf.add_n(projected)
        if self.mode == "mean":
            return tf.add_n(projected) / tf.cast(self._num_inputs, projected[0].dtype)
        if self.mode == "max":
            stacked = tf.stack(projected, axis=self.axis)
            return tf.reduce_max(stacked, axis=self.axis)
        if self.mode == "weighted":
            stacked = tf.stack(projected, axis=0)
            weights = tf.reshape(tf.nn.softmax(self.layer_weights), [-1, 1, 1])
            return tf.reduce_sum(stacked * weights, axis=0)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        batch = input_shape[0][0]
        if self.mode == "concat":
            channels = sum(int(shape[-1]) for shape in input_shape)
        else:
            channels = self.target_dim
        return (batch, channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "axis": self.axis,
                "target_dim": self.target_dim,
                "projection_kwargs": self.projection_kwargs,
            }
        )
        return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nnlib_v2_nmd.py::TestNMDMerge -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/v2/nmd.py tests/unit/test_nnlib_v2_nmd.py
git commit -m "feat: add NMDMerge layer"
```

---

## Task 3: Register `nmd` in the builder

**Files:**
- Modify: `src/jaeger/nnlib/builder.py`
- Test: `tests/unit/test_nnlib_v2_nmd.py` (existing import) + run full builder smoke tests

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_nnlib_v2_nmd.py`:

```python
def test_builder_knows_nmd_layer():
    from jaeger.nnlib.builder import DynamicModelBuilder

    config = {
        "model": {
            "name": "test_nmd_registration",
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
                "embedding_regularizer": "l2",
                "embedding_regularizer_w": 1e-05,
            },
            "string_processor": {
                "data_format": "numpy_full",
                "seq_onehot": False,
                "codon": "CODON",
                "codon_id": "CODON_ID",
                "crop_size": 100,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "hidden_layers": [
                    {"name": "masked_conv1d", "config": {"filters": 16, "kernel_size": 3}},
                    {"name": "nmd", "config": {}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "max",
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
            "model_saving": {"path": "/tmp/test_nmd_registration", "save_weights": False, "save_exec_graph": False},
            "fragment_classifier_data": {"train": [{"class": ["chromosome"], "label": [0], "path": []}]},
        },
        "config_path": "/tmp/test_nmd_registration_config.yaml",
    }
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    assert "nmd" in builder._layers
    assert "prediction" in models["jaeger_model"].output_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_nnlib_v2_nmd.py::test_builder_knows_nmd_layer -v`
Expected: `ValueError: Unknown layer type: nmd`

- [ ] **Step 3: Write minimal implementation**

In `src/jaeger/nnlib/builder.py`:

1. Add import:

```python
from jaeger.nnlib.v2.nmd import NMDLayer, NMDMerge
```

2. Register layer in `_layers`:

```python
self._layers = {
    # ... existing entries ...
    "nmd": NMDLayer,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_nnlib_v2_nmd.py::test_builder_knows_nmd_layer -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/builder.py tests/unit/test_nnlib_v2_nmd.py
git commit -m "feat: register nmd layer in builder"
```

---

## Task 4: Collect multiple NMDs and apply merge in the representation learner

**Files:**
- Modify: `src/jaeger/nnlib/builder.py`
- Test: `tests/integration/test_builder_nmd_merge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_builder_nmd_merge.py`:

```python
"""Integration tests for multiple NMD layers and merge modes."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder


@pytest.fixture
def base_config():
    return {
        "model": {
            "name": "test_nmd_merge",
            "experiment": "test_nmd_merge",
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
                "embedding_regularizer": "l2",
                "embedding_regularizer_w": 1e-05,
            },
            "string_processor": {
                "data_format": "numpy_full",
                "seq_onehot": False,
                "codon": "CODON",
                "codon_id": "CODON_ID",
                "crop_size": 100,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "hidden_layers": [
                    {"name": "masked_conv1d", "config": {"filters": 16, "kernel_size": 3}},
                    {"name": "nmd", "config": {}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                    {"name": "masked_conv1d", "config": {"filters": 8, "kernel_size": 3}},
                    {"name": "nmd", "config": {}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "max",
            },
            "classifier": {
                "input_shape": 8,
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
            "model_saving": {"path": "/tmp/test_nmd_merge", "save_weights": False, "save_exec_graph": False},
            "fragment_classifier_data": {"train": [{"class": ["chromosome"], "label": [0], "path": []}]},
        },
        "config_path": "/tmp/test_nmd_merge_config.yaml",
    }


def test_multiple_nmds_concat_merge(base_config):
    base_config["model"]["reliability_model"] = {
        "merge": {"mode": "concat"},
        "input_shape": 24,
        "hidden_layers": [
            {"name": "dense", "config": {"units": 1, "activation": None, "dtype": "float32"}}
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        base_config["training"]["model_saving"]["path"] = tmp
        builder = DynamicModelBuilder(base_config)
        models = builder.build_fragment_classifier()
        nmd_out = models["rep_model"].output[1]
        assert list(nmd_out.shape) == [None, 24]
        rel_in = models["reliability_head"].input
        assert list(rel_in.shape) == [None, 24]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_builder_nmd_merge.py::test_multiple_nmds_concat_merge -v`
Expected: `IndexError: tuple index out of range` or shape mismatch because NMDs are not merged.

- [ ] **Step 3: Write minimal implementation**

1. Update `_build_block` signature and aggregation:

```python
def _build_block(self, x, cfg: dict[str, Any], prefix: str, nmd_merge: dict[str, Any] | None = None):
    """Build a stack of layers from *cfg* and return the output tensor(s)."""
    nmd = []
    previous_channels = None
    for i, layer_cfg in enumerate(cfg.get("hidden_layers", [])):
        # ... existing layer lookup and config code ...

        if "block_size" in cfg_layer:
            # ... existing block handling ...
            continue

        if "return_nmd" in cfg_layer:
            # ... existing return_nmd handling ...
            continue

        # New: explicit nmd layer
        if layer_name == "nmd":
            nmd_ = layer_class(**cfg_layer)(x)
            nmd.append(nmd_)
            continue

        x = layer_class(**cfg_layer)(x)
        # ... existing previous_channels update ...

    # Aggregation
    if "pooling" in cfg:
        pooling = cfg.get("pooling", "average").lower()
        pooler = self._get_pooler(pooling)
        has_nmd = len(nmd) > 0
        if has_nmd:
            if nmd_merge is not None:
                nmd = NMDMerge(name=f"{prefix}_nmd_merge", **nmd_merge)(nmd)
            elif len(nmd) > 1:
                nmd = tf.keras.layers.Concatenate(axis=-1, name=f"{prefix}_nmd_concat")(nmd)
            else:
                nmd = nmd[0]
        # ... rest unchanged ...
```

2. In `build_fragment_classifier`, pass the merge config when building the representation learner:

```python
merge_cfg = self.model_cfg.get("reliability_model", {}).get("merge")
rep_output = self._build_block(
    x, self.model_cfg["representation_learner"], prefix="rep", nmd_merge=merge_cfg
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_builder_nmd_merge.py::test_multiple_nmds_concat_merge -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/builder.py tests/integration/test_builder_nmd_merge.py
git commit -m "feat: collect and merge multiple NMD tensors"
```

---

## Task 5: Wire merged NMD to the reliability head with shape validation

**Files:**
- Modify: `src/jaeger/nnlib/builder.py`
- Test: `tests/integration/test_builder_nmd_merge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_builder_nmd_merge.py`:

```python
def test_reliability_input_shape_mismatch_raises(base_config):
    base_config["model"]["reliability_model"] = {
        "merge": {"mode": "sum", "target_dim": 8},
        "input_shape": 999,
        "hidden_layers": [
            {"name": "dense", "config": {"units": 1, "activation": None, "dtype": "float32"}}
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        base_config["training"]["model_saving"]["path"] = tmp
        builder = DynamicModelBuilder(base_config)
        with pytest.raises(ValueError, match="does not match"):
            builder.build_fragment_classifier()


def test_missing_nmd_raises_when_reliability_configured(base_config):
    # Remove all nmd layers
    base_config["model"]["representation_learner"]["hidden_layers"] = [
        {"name": "masked_conv1d", "config": {"filters": 16, "kernel_size": 3}},
        {"name": "activation", "config": {"activation": "gelu"}},
    ]
    base_config["model"]["classifier"]["input_shape"] = 16
    base_config["model"]["reliability_model"] = {
        "merge": {"mode": "concat"},
        "input_shape": 16,
        "hidden_layers": [
            {"name": "dense", "config": {"units": 1, "activation": None, "dtype": "float32"}}
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        base_config["training"]["model_saving"]["path"] = tmp
        builder = DynamicModelBuilder(base_config)
        with pytest.raises(ValueError, match="no NMD tensor"):
            builder.build_fragment_classifier()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_builder_nmd_merge.py::test_reliability_input_shape_mismatch_raises tests/integration/test_builder_nmd_merge.py::test_missing_nmd_raises_when_reliability_configured -v`
Expected: tests fail because the builder does not validate yet.

- [ ] **Step 3: Write minimal implementation**

In `src/jaeger/nnlib/builder.py`, replace the reliability section with:

```python
        # === 4. RELIABILITY ===
        if "reliability_model" in self.model_cfg:
            reliability_cfg = self.model_cfg["reliability_model"]
            input_shape = (reliability_cfg.get("input_shape"),)
            inputs = tf.keras.Input(shape=input_shape, name="reliability_input")
            x_reliability = self._build_block(
                inputs, reliability_cfg, prefix="reliability"
            )
            models["reliability_head"] = tf.keras.Model(
                inputs=inputs, outputs=x_reliability, name="reliability_head"
            )

            rep_out = models["rep_model"].output
            if not isinstance(rep_out, (list, tuple)) or len(rep_out) < 2:
                raise ValueError(
                    "reliability_model is configured but the representation learner "
                    "produced no NMD tensor. Add an `nmd` layer or set "
                    "return_nmd: true on a layer that supports it."
                )
            nmd = rep_out[1]
            expected_dim = reliability_cfg.get("input_shape")
            actual_dim = tf.keras.backend.int_shape(nmd)[-1]
            if expected_dim is not None and actual_dim is not None and actual_dim != expected_dim:
                raise ValueError(
                    f"Merged NMD dimension ({actual_dim}) does not match "
                    f"reliability_model.input_shape ({expected_dim})."
                )
            x = models["reliability_head"](nmd)
            models["jaeger_reliability"] = tf.keras.Model(
                inputs=models["rep_model"].input, outputs=x, name="Jaeger_reliability"
            )
            # ... existing checkpoint loading code ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_builder_nmd_merge.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/jaeger/nnlib/builder.py tests/integration/test_builder_nmd_merge.py
git commit -m "feat: validate and wire merged NMD to reliability head"
```

---

## Task 6: Add integration tests for remaining merge modes

**Files:**
- Modify: `tests/integration/test_builder_nmd_merge.py`

- [ ] **Step 1: Add tests**

Append parametrized tests:

```python
@pytest.mark.parametrize("mode,target_dim,input_shape", [
    ("sum", 8, 8),
    ("mean", 8, 8),
    ("max", 8, 8),
    ("weighted", 8, 8),
])
def test_merge_modes(base_config, mode, target_dim, input_shape):
    base_config["model"]["reliability_model"] = {
        "merge": {"mode": mode, "target_dim": target_dim},
        "input_shape": input_shape,
        "hidden_layers": [
            {"name": "dense", "config": {"units": 1, "activation": None, "dtype": "float32"}}
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        base_config["training"]["model_saving"]["path"] = tmp
        builder = DynamicModelBuilder(base_config)
        models = builder.build_fragment_classifier()
        nmd_out = models["rep_model"].output[1]
        assert list(nmd_out.shape) == [None, input_shape]
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/integration/test_builder_nmd_merge.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_builder_nmd_merge.py
git commit -m "test: cover all NMD merge modes"
```

---

## Task 7: Add example config

**Files:**
- Create: `train_config/nn_config_500bp_nmd_merge.yaml`

- [ ] **Step 1: Create the config**

Copy `train_config/nn_config_500bp_baseline.yaml` and replace the representation/reliability sections with:

```yaml
  representation_learner:
    hidden_layers:
    - name: masked_conv1d
      config:
        filters: 32
        kernel_size: 7
    - name: nmd
      config:
        epsilon: 1e-5
        momentum: 0.9
    - name: activation
      config:
        activation: gelu
    - name: residual_block
      config:
        use_1x1conv: false
        block_size: 2
        filters: 32
        kernel_size: 3
    - name: nmd
      config:
        epsilon: 1e-5
    - name: activation
      config:
        activation: gelu
    pooling: max

  reliability_model:
    merge:
      mode: concat
      axis: -1
    input_shape: 64
    hidden_layers:
    - name: dense
      config:
        units: 8
        activation: gelu
        use_bias: true
        kernel_regularizer: l2
        kernel_regularizer_w: 0.00001
    - name: dropout
      config:
        rate: 0.5
    - name: dense
      config:
        units: 1
        activation: null
        dtype: float32
        use_bias: true
        kernel_regularizer: l2
        kernel_regularizer_w: 0.05
```

- [ ] **Step 2: Validate it builds**

Run a small Python script or add a smoke test that instantiates `DynamicModelBuilder` with this config and calls `build_fragment_classifier()`.

- [ ] **Step 3: Commit**

```bash
git add train_config/nn_config_500bp_nmd_merge.yaml
git commit -m "config: add example NMD merge configuration"
```

---

## Task 8: Run the full test suite

- [ ] **Step 1: Run focused tests**

```bash
pytest tests/unit/test_nnlib_v2_nmd.py tests/integration/test_builder_nmd_merge.py tests/integration/test_builder_projection.py -v
```

Expected: all pass.

- [ ] **Step 2: Run broader smoke tests**

```bash
pytest tests/unit/test_nnlib_v2_layers.py tests/integration/ -v
```

Expected: all pass.

- [ ] **Step 3: Commit any fixes**

```bash
git commit -am "fix: address test fallout from NMD changes"
```

---

## Self-review checklist

1. **Spec coverage:**
   - Standalone `NMDLayer` → Task 1.
   - Multiple NMD layers → Task 4.
   - Configurable merge (`concat`, `sum`, `mean`, `max`, `weighted`) → Tasks 2 and 6.
   - Builder auto-detects and merges → Tasks 3, 4, 5.
   - `reliability_model.merge` config parameter → Task 4/5.
   - Backward compatibility → preserved by leaving `return_nmd` paths untouched.
   - Validation and clear errors → Task 5.
   - Unit + integration tests → Tasks 1, 2, 6, 8.

2. **Placeholder scan:** No TBD/TODO/fill-in-details strings.

3. **Type consistency:** `NMDMerge` accepts `target_dim` (int or None); builder passes `merge` dict values directly.

4. **Scope:** This plan covers one coherent feature. No decomposition needed.
