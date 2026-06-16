"""Tests for jaeger.nnlib.v2.nmd."""

from __future__ import annotations

import pytest
import tensorflow as tf

from jaeger.nnlib.v2.layers import MaskedBatchNorm
from jaeger.nnlib.v2.nmd import NMDLayer, NMDMerge


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

    def test_output_shape_3d_without_mask(self):
        x = tf.random.normal((2, 12, 8))
        layer = NMDLayer()
        out = layer(x)
        assert list(out.shape) == [2, 8]

    def test_nmd_matches_masked_batch_norm_with_mask(self):
        tf.random.set_seed(42)
        x = tf.random.normal((4, 6, 32, 8))
        # variable-length mask: first two examples are fully valid, last two are partially masked
        mask = tf.concat(
            [
                tf.ones((2, 6, 32), dtype=tf.int32),
                tf.random.stateless_binomial(
                    shape=(2, 6, 32), seed=(12, 34), counts=1, probs=0.7
                ),
            ],
            axis=0,
        )
        mask = tf.cast(mask, tf.bool)

        nmd_layer = NMDLayer(epsilon=1e-5, momentum=0.9)
        nmd_out = nmd_layer(x, mask=mask, training=False)

        bn_layer = MaskedBatchNorm(return_nmd=True, epsilon=1e-5, momentum=0.9)
        # Build the layer so moving statistics are created in the same initial state.
        bn_layer(x, mask=mask, training=False)
        _, bn_nmd = bn_layer(x, mask=mask, training=False)

        max_diff = float(tf.reduce_max(tf.abs(nmd_out - bn_nmd)))
        assert max_diff < 1e-5

    def test_get_config_roundtrip(self):
        layer = NMDLayer(epsilon=1e-3, momentum=0.95, dtype="float32")
        config = layer.get_config()
        restored = NMDLayer.from_config(config)
        assert restored.epsilon == 1e-3
        assert restored.momentum == 0.95


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
                "data_format": "numpy",
                "seq_onehot": False,
                "codon": "CODON",
                "codon_id": "CODON_ID",
                "crop_size": 100,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "hidden_layers": [
                    {
                        "name": "masked_conv1d",
                        "config": {"filters": 16, "kernel_size": 3},
                    },
                    {"name": "nmd", "config": {}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "max",
            },
            "classifier": {
                "input_shape": 16,
                "hidden_layers": [
                    {
                        "name": "dense",
                        "config": {"units": 3, "activation": None, "dtype": "float32"},
                    }
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
            "model_saving": {
                "path": "/tmp/test_nmd_registration",
                "save_weights": False,
                "save_exec_graph": False,
            },
            "fragment_classifier_data": {
                "train": [{"class": ["chromosome"], "label": [0], "path": []}]
            },
        },
        "config_path": "/tmp/test_nmd_registration_config.yaml",
    }
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    assert "nmd" in builder._layers
    assert "prediction" in models["jaeger_model"].output
