"""Tests for jaeger.nnlib.builder."""

from __future__ import annotations

import keras
import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder


class TestBuildBlock:
    """Unit tests for DynamicModelBuilder._build_block."""

    @pytest.fixture
    def builder(self, tmp_path):
        """Return a minimal builder instance for testing."""
        builder = DynamicModelBuilder.__new__(DynamicModelBuilder)
        builder.output_dir = tmp_path / "output"
        builder.output_dir.mkdir(parents=True, exist_ok=True)
        builder._layers = {
            "masked_conv1d": type("MaskedConv1D", (), {}),
            "conv1d": tf.keras.layers.Conv1D,
            "dense": tf.keras.layers.Dense,
            "activation": tf.keras.layers.Activation,
            "relu": tf.keras.layers.Activation,
            "gelu": tf.keras.layers.Activation,
            "sigmoid": tf.keras.layers.Activation,
            "softmax": tf.keras.layers.Activation,
            "tanh": tf.keras.layers.Activation,
            "dropout": tf.keras.layers.Dropout,
        }
        builder._regularizer = {}
        builder.model_cfg = {}
        builder.input_shape = (100, 4)
        return builder

    def test_activation_alias_and_max1d_pooling(self, builder):
        """ReLU as a standalone layer alias plus max1d pooling should work."""
        x = tf.keras.Input(shape=(32, 4), name="input")
        cfg = {
            "hidden_layers": [
                {"name": "relu"},
            ],
            "pooling": "max1d",
        }
        out = builder._build_block(x, cfg, prefix="test")
        assert isinstance(out, keras.KerasTensor)
        assert list(out.shape) == [None, 4]
