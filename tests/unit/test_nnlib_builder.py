"""Tests for jaeger.nnlib.builder."""

from __future__ import annotations

import json

import keras
import numpy as np
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
    return {
        "model": model_cfg,
        "training": {"loss_reliability": "binary_crossentropy"},
        "generate_reliability_data": True,
    }


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


class TestGetBias:
    """Unit tests for DynamicModelBuilder._get_bias."""

    @pytest.fixture
    def builder(self, tmp_path):
        """Return a minimal builder instance."""
        builder = DynamicModelBuilder.__new__(DynamicModelBuilder)
        builder.output_dir = tmp_path / "output"
        builder.output_dir.mkdir(parents=True, exist_ok=True)
        return builder

    def _labels(self):
        """Shared label distribution: class 0 x2, class 1 x1, class 2 x3."""
        return [0, 0, 1, 2, 2, 2]

    def _expected_softmax(self):
        counts = np.array([2.0, 1.0, 3.0])
        return np.log(counts / counts.sum())

    def test_get_bias_csv(self, builder, tmp_path):
        """CSV first-column labels should produce correct softmax bias."""
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("\n".join(str(label) for label in self._labels()))
        bias = builder._get_bias(str(csv_path), kind="softmax", label_map=[0, 1, 2])
        np.testing.assert_allclose(bias, self._expected_softmax())

    def test_get_bias_npz_labels_array(self, builder, tmp_path):
        """NPZ with a 'labels' array should produce correct softmax bias."""
        npz_path = tmp_path / "data.npz"
        np.savez(npz_path, labels=np.array(self._labels()))
        bias = builder._get_bias(str(npz_path), kind="softmax", label_map=[0, 1, 2])
        np.testing.assert_allclose(bias, self._expected_softmax())

    def test_get_bias_npz_label_array(self, builder, tmp_path):
        """NPZ with a 'label' array should produce correct softmax bias."""
        npz_path = tmp_path / "data.npz"
        np.savez(npz_path, label=np.array(self._labels()))
        bias = builder._get_bias(str(npz_path), kind="softmax", label_map=[0, 1, 2])
        np.testing.assert_allclose(bias, self._expected_softmax())

    def test_get_bias_npz_sharded(self, builder, tmp_path):
        """Sharded NPZ with labels_* arrays should be concatenated."""
        npz_path = tmp_path / "data.npz"
        labels = self._labels()
        np.savez(
            npz_path,
            labels_00000=np.array(labels[:3]),
            labels_00001=np.array(labels[3:]),
        )
        bias = builder._get_bias(str(npz_path), kind="softmax", label_map=[0, 1, 2])
        np.testing.assert_allclose(bias, self._expected_softmax())

    def test_get_bias_npz_onehot_labels(self, builder, tmp_path):
        """One-hot encoded labels in NPZ should be argmaxed to counts."""
        npz_path = tmp_path / "data.npz"
        onehot = np.eye(3, dtype=np.float32)[self._labels()]
        np.savez(npz_path, labels=onehot)
        bias = builder._get_bias(str(npz_path), kind="softmax", label_map=[0, 1, 2])
        np.testing.assert_allclose(bias, self._expected_softmax())

    def test_get_bias_npz_missing_labels_raises(self, builder, tmp_path):
        """NPZ without any recognisable labels should raise ValueError."""
        npz_path = tmp_path / "data.npz"
        np.savez(npz_path, features=np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="contains no 'labels'"):
            builder._get_bias(str(npz_path), kind="softmax", label_map=[])


def _reliability_bias_config(
    rel_path: str,
    generate_reliability_data: bool = False,
    bias_initializer: str | None = "calculate_from_train_data",
):
    """Return a config that exercises reliability bias initialization."""
    hidden_layers = [
        {"name": "dense", "config": {"units": 4}},
        {"name": "relu"},
        {
            "name": "dense",
            "config": {
                "units": 1,
                "activation": None,
                "dtype": "float32",
            },
        },
    ]
    if bias_initializer:
        hidden_layers[-1]["config"]["bias_initializer"] = bias_initializer

    return {
        "model": {
            "classifier_out_dim": 3,
            "reliability_out_dim": 1,
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
                "input_shape": 8,
                "hidden_layers": hidden_layers,
            },
        },
        "training": {
            "loss_reliability": "binary_crossentropy",
            "fragment_reliability_data": {
                "train": [
                    {
                        "class": ["ood", "indist"],
                        "path": [rel_path],
                        "label": [0, 1],
                    }
                ]
            },
        },
        "generate_reliability_data": generate_reliability_data,
    }


class TestReliabilityBiasInitialization:
    """Tests for deferred/skipped reliability bias initialization."""

    def test_reliability_bias_computed_from_existing_npz(self, tmp_path):
        """When reliability NPZ exists, bias should be non-zero from data."""
        rel_path = tmp_path / "reliability.npz"
        np.savez(rel_path, labels=np.array([0, 1, 1, 1, 0, 1]))

        config = _reliability_bias_config(str(rel_path))
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()

        assert "reliability_head" in models
        bias = models["reliability_head"].layers[-1].bias.numpy()
        # Binary sigmoid bias: log(count(positive) / count(negative))
        np.testing.assert_allclose(bias, np.log(4 / 2), atol=1e-6)

    def test_reliability_bias_deferred_when_generating_data(self, tmp_path):
        """If data is missing but --generate_reliability_data is set, bias starts at
        zero and can be updated once the data has been generated."""
        missing_path = tmp_path / "not_yet_generated.npz"
        generated_path = tmp_path / "generated.npz"
        np.savez(generated_path, labels=np.array([0, 1, 1, 1, 0, 1]))

        config = _reliability_bias_config(
            str(missing_path), generate_reliability_data=True
        )
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()

        assert "reliability_head" in models
        # Bias is deferred -> initialized to zero.
        initial_bias = models["reliability_head"].layers[-1].bias.numpy()
        np.testing.assert_allclose(initial_bias, 0.0, atol=1e-7)

        builder._set_reliability_bias(models["reliability_head"], str(generated_path))
        updated_bias = models["reliability_head"].layers[-1].bias.numpy()
        np.testing.assert_allclose(updated_bias, np.log(4 / 2), atol=1e-6)

    def test_reliability_head_skipped_when_data_missing(self, tmp_path):
        """If data is missing and we are not generating it, skip reliability head."""
        missing_path = tmp_path / "missing.npz"
        config = _reliability_bias_config(
            str(missing_path), generate_reliability_data=False
        )
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()

        assert "reliability_head" not in models
        assert "jaeger_reliability" not in models


class TestCheckpointConvergence:
    """Tests for convergence marker handling in checkpoint metadata."""

    @pytest.fixture
    def builder(self, tmp_path):
        builder = DynamicModelBuilder.__new__(DynamicModelBuilder)
        builder.output_dir = tmp_path / "output"
        builder.output_dir.mkdir(parents=True, exist_ok=True)
        return builder

    def test_converged_marker_detected(self, builder, tmp_path):
        checkpoint_dir = tmp_path / "classifier"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "epoch:03-loss:1.23.weights.h5").write_text("dummy")
        (checkpoint_dir / "converged.json").write_text(
            json.dumps({"is_converged": True, "branch": "classifier", "epoch": 3})
        )
        meta = builder.get_latest_h5_with_metadata(checkpoint_dir)
        assert meta["is_converged"] is True
        assert meta["epoch"] == 3

    def test_no_marker_not_converged(self, builder, tmp_path):
        checkpoint_dir = tmp_path / "classifier"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "epoch:05-loss:0.99.weights.h5").write_text("dummy")
        meta = builder.get_latest_h5_with_metadata(checkpoint_dir)
        assert meta["is_converged"] is False
        assert meta["epoch"] == 5

    def test_marker_without_checkpoint(self, builder, tmp_path):
        checkpoint_dir = tmp_path / "classifier"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "converged.json").write_text(
            json.dumps({"is_converged": True, "branch": "classifier", "epoch": 0})
        )
        meta = builder.get_latest_h5_with_metadata(checkpoint_dir)
        assert meta["is_converged"] is True
        assert meta["path"] is None

    def test_malformed_marker_defaults_to_not_converged(self, builder, tmp_path):
        checkpoint_dir = tmp_path / "classifier"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "epoch:02-loss:1.00.weights.h5").write_text("dummy")
        (checkpoint_dir / "converged.json").write_text("not json")
        meta = builder.get_latest_h5_with_metadata(checkpoint_dir)
        assert meta["is_converged"] is False
        assert meta["epoch"] == 2
