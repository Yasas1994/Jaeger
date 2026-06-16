"""Integration tests for multiple NMD layers and merge modes."""

from __future__ import annotations

import tempfile

import pytest

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
