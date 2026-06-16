"""Integration tests for projection-head model building."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder


@pytest.fixture
def base_config():
    """Minimal config that builds a model with a projection head."""
    return {
        "model": {
            "name": "test_projection",
            "experiment": "test_projection",
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
                "embedding_regularizer_w": 1.0e-05,
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
                    {
                        "name": "masked_conv1d",
                        "config": {
                            "filters": 32,
                            "kernel_size": 3,
                            "use_bias": True,
                            "kernel_regularizer": "l2",
                            "kernel_regularizer_w": 1.0e-05,
                        },
                    },
                    {"name": "masked_batchnorm", "config": {"return_nmd": False}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "max",
            },
            "projection": {
                "margin": 0.5,
                "scale": 64,
                "input_shape": 32,
                "hidden_layers": [
                    {
                        "name": "dense",
                        "config": {
                            "units": 16,
                            "activation": "gelu",
                            "use_bias": True,
                            "kernel_regularizer": "l2",
                            "kernel_regularizer_w": 1.0e-05,
                        },
                    },
                    {
                        "name": "dense",
                        "config": {
                            "units": 16,
                            "activation": "gelu",
                            "use_bias": True,
                            "kernel_regularizer": "l2",
                            "kernel_regularizer_w": 1.0e-05,
                        },
                    },
                ],
            },
            "classifier": {
                "input_shape": 32,
                "hidden_layers": [
                    {"name": "dropout", "config": {"rate": 0.3}},
                    {
                        "name": "dense",
                        "config": {
                            "units": 3,
                            "activation": None,
                            "dtype": "float32",
                        },
                    },
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
                "path": "/tmp/jaeger_projection_test",
                "save_weights": False,
                "save_exec_graph": False,
            },
            "fragment_classifier_data": {
                "train": [{"class": ["chromosome"], "label": [0], "path": []}]
            },
        },
        "config_path": "/tmp/jaeger_projection_test_config.yaml",
    }


def _single_output(model):
    """Return the single output tensor, handling Keras list/tuple wrappers."""
    out = model.output
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _shape_as_list(tensor):
    """Return tensor.shape as a list, handling both TensorShape and tuple."""
    return list(tensor.shape)


def test_projection_model_preserves_batch_dim(base_config):
    """The projection model must keep the batch dimension from rep_model output."""
    with tempfile.TemporaryDirectory() as tmp:
        base_config["training"]["model_saving"]["path"] = tmp
        builder = DynamicModelBuilder(base_config)
        models = builder.build_fragment_classifier()

        # rep_model output is a single tensor (batch, 32)
        assert _shape_as_list(_single_output(models["rep_model"])) == [None, 32]
        # jaeger_projection output must also be batched (batch, 16)
        assert _shape_as_list(_single_output(models["jaeger_projection"])) == [
            None,
            16,
        ]


def test_projection_forward_pass(base_config):
    """jaeger_projection can run a forward pass on tokenized input."""
    with tempfile.TemporaryDirectory() as tmp:
        base_config["training"]["model_saving"]["path"] = tmp
        builder = DynamicModelBuilder(base_config)
        models = builder.build_fragment_classifier()

        # translated input is integer token IDs, shape (batch, strands, length)
        batch_size = 4
        x = tf.constant(
            np.random.randint(0, 64, size=(batch_size, 6, 100)), dtype=tf.int32
        )
        out = models["jaeger_projection"](x)
        assert out.shape.as_list() == [batch_size, 16]
