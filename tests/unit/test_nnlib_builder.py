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


def _reliability_config(mode=None):
    """Return a minimal config for testing reliability model modes."""
    model_cfg = {
        "classifier_out_dim": 3,
        "reliability_out_dim": 2,
        "embedding": {
            "input_type": "nucleotide",
            "input_shape": (None, 4),
            "use_embedding_layer": False,
        },
        "representation_learner": {
            "hidden_layers": [
                {"name": "conv1d", "config": {"filters": 8, "kernel_size": 3}},
                {"name": "relu"},
                {"name": "nmd"},
            ],
            "pooling": "average1d",
        },
        "classifier": {
            "input_shape": 8,
            "hidden_layers": [
                {"name": "dense", "config": {"units": 8}},
                {"name": "relu"},
                {"name": "dense", "config": {"units": 3}},
            ],
        },
        "reliability_model": {
            "hidden_layers": [
                {"name": "dense", "config": {"units": 4}},
                {"name": "relu"},
                {"name": "dense", "config": {"units": 2}},
            ],
        },
    }
    if mode is not None:
        model_cfg["reliability_model"]["mode"] = mode
    return {"model": model_cfg, "training": {"loss_reliability": "binary_crossentropy"}}


class TestReliabilityModes:
    """Tests for reliability_model mode handling in DynamicModelBuilder."""

    def test_reliability_nmd_mode_builds(self):
        """Default nmd mode should produce a jaeger_reliability output."""
        config = _reliability_config(mode="nmd")
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()

        assert "jaeger_reliability" in models
        assert models["jaeger_reliability"].output.shape[-1] == 2
        assert list(models["jaeger_reliability"].output.shape) == [None, 2]
        assert "reliability" in models["jaeger_model"].output

    def test_reliability_nmd_plus_signals_builds(self):
        """nmd_plus_signals mode should concatenate NMD and OOD signals."""
        config = _reliability_config(mode="nmd_plus_signals")
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()

        assert models["jaeger_reliability"].output.shape[-1] == 2
        assert list(models["jaeger_reliability"].output.shape) == [None, 2]

        nmd_dim = int(models["rep_model"].output[1].shape[-1])
        default_signals = [
            "max_prob",
            "entropy",
            "energy",
            "margin",
            "nmd_norm",
        ]
        expected_dim = nmd_dim + len(default_signals)
        assert int(models["reliability_head"].input.shape[-1]) == expected_dim

    def test_reliability_invalid_mode_raises(self):
        """An unsupported reliability mode should raise ValueError."""
        config = _reliability_config(mode="unknown")
        builder = DynamicModelBuilder(config)
        with pytest.raises(ValueError, match="Unsupported reliability_model.mode"):
            builder.build_fragment_classifier()

    def test_compile_reliability_freezes_classifier(self):
        """Compiling for reliability training should freeze earlier branches."""
        config = _reliability_config(mode="nmd_plus_signals")
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()
        builder.compile_model(models, train_branch="reliability")

        assert models["rep_model"].trainable is False
        assert models["classification_head"].trainable is False
