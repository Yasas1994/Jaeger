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


class TestBranchedBlock:
    """Unit tests for DynamicModelBuilder._build_branched_block."""

    @pytest.fixture
    def builder(self, tmp_path):
        """Return a minimal builder instance for testing branched blocks."""
        builder = DynamicModelBuilder.__new__(DynamicModelBuilder)
        builder.output_dir = tmp_path / "output"
        builder.output_dir.mkdir(parents=True, exist_ok=True)
        builder._layers = {
            "conv1d": tf.keras.layers.Conv1D,
            "activation": tf.keras.layers.Activation,
            "relu": tf.keras.layers.Activation,
        }
        builder._regularizer = {}
        builder.model_cfg = {}
        builder.input_shape = (16, 4)
        return builder

    def test_branched_block_splits_and_merges(self, builder):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(2, 16, 4), name="nucleotide")
        branch_cfg = {
            "hidden_layers": [
                {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                {"name": "relu"},
            ],
            "pooling": "max1d",
        }
        out = builder._build_branched_block(
            inputs, branch_cfg, prefix="test", merge_method="average"
        )
        assert list(out.shape) == [None, 8]

    def test_branched_block_none_returns_list(self, builder):
        inputs = tf.keras.Input(shape=(2, 16, 4), name="nucleotide")
        branch_cfg = {
            "hidden_layers": [
                {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                {"name": "relu"},
            ],
            "pooling": "max1d",
        }
        out = builder._build_branched_block(
            inputs, branch_cfg, prefix="test", merge_method=None
        )
        assert isinstance(out, list)
        assert len(out) == 2
        for tensor in out:
            assert list(tensor.shape) == [None, 8]

    @pytest.mark.parametrize(
        "method,expected_dim",
        [
            ("sum", 8),
            ("concat", 16),
            ("max", 8),
        ],
    )
    def test_branched_block_merge_methods(self, builder, method, expected_dim):
        inputs = tf.keras.Input(shape=(2, 16, 4), name="nucleotide")
        branch_cfg = {
            "hidden_layers": [
                {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                {"name": "relu"},
            ],
            "pooling": "max1d",
        }
        out = builder._build_branched_block(
            inputs, branch_cfg, prefix="test", merge_method=method
        )
        assert list(out.shape) == [None, expected_dim]

    def test_branched_block_invalid_merge_method(self, builder):
        inputs = tf.keras.Input(shape=(2, 16, 4), name="nucleotide")
        branch_cfg = {
            "hidden_layers": [
                {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                {"name": "relu"},
            ],
            "pooling": "max1d",
        }
        with pytest.raises(ValueError, match="Unknown merge method"):
            builder._build_branched_block(
                inputs, branch_cfg, prefix="test", merge_method="unknown"
            )

    def test_branched_block_list_input(self, builder):
        branch1 = tf.keras.Input(shape=(16, 4), name="branch1")
        branch2 = tf.keras.Input(shape=(16, 4), name="branch2")
        branch_cfg = {
            "hidden_layers": [
                {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                {"name": "relu"},
            ],
            "pooling": "max1d",
        }
        out = builder._build_branched_block(
            [branch1, branch2], branch_cfg, prefix="test", merge_method="sum"
        )
        assert list(out.shape) == [None, 8]


def test_dvf_model_builds():
    from jaeger.nnlib.builder import DynamicModelBuilder

    config = {
        "model": {
            "name": "test_dvf",
            "experiment": "test_dvf",
            "classifier_out_dim": 3,
            "embedding": {
                "input_type": "nucleotide",
                "input_shape": (2, None, 4),
                "use_embedding_layer": False,
            },
            "representation_learner": {
                "branch": {
                    "hidden_layers": [
                        {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                        {"name": "relu"},
                    ],
                    "pooling": "max1d",
                }
            },
            "classifier": {
                "branch": {
                    "hidden_layers": [
                        {"name": "dense", "config": {"units": 8}},
                        {"name": "relu"},
                        {"name": "dense", "config": {"units": 3}},
                        {"name": "merge", "config": {"method": "average"}},
                    ]
                }
            },
        },
        "training": {},
    }
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    assert "jaeger_classifier" in models
    assert models["jaeger_classifier"].output.shape[-1] == 3
    assert "jaeger_model" in models
