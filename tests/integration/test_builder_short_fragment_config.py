"""Integration test for short-fragment config support."""

from __future__ import annotations

from pathlib import Path
import tempfile

import tensorflow as tf
from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.utils.misc import load_model_config


def _all_layer_names(model: tf.keras.Model) -> set[str]:
    names: set[str] = set()
    for layer in model.layers:
        names.add(layer.name)
        if hasattr(layer, "layers"):
            names.update(_all_layer_names(layer))
    return names


def test_short_fragment_stack_builds():
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "model": {
                "name": "test_short_fragment_stack",
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
                    "embedding_regularizer_w": 1.0e-05,
                },
                "string_processor": {
                    "data_format": "numpy",
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
                                    {
                                        "filters": 8,
                                        "kernel_size": 3,
                                        "dilation_rate": 1,
                                    },
                                    {
                                        "filters": 8,
                                        "kernel_size": 5,
                                        "dilation_rate": 1,
                                    },
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
                    "hidden_layers": [
                        {
                            "name": "dense",
                            "config": {
                                "units": 3,
                                "activation": None,
                                "dtype": "float32",
                            },
                        }
                    ],
                },
            },
            "training": {
                "optimizer": "adam",
                "optimizer_params": {"learning_rate": 0.001},
                "loss_classifier": "categorical_crossentropy",
                "loss_params_classifier": {"from_logits": True},
                "metrics_classifier": [
                    {"name": "categorical_accuracy", "params": None}
                ],
                "callbacks": {"directories": []},
                "model_saving": {
                    "path": tmp,
                    "save_weights": False,
                    "save_exec_graph": False,
                },
                "fragment_classifier_data": {
                    "train": [{"class": ["chromosome"], "label": [0], "path": []}]
                },
            },
            "config_path": str(Path(tmp) / "jaeger_test_config.yaml"),
        }
        builder = DynamicModelBuilder(config)
        models = builder.build_fragment_classifier()
        assert "jaeger_model" in models
        layer_names = _all_layer_names(models["jaeger_model"])
        assert any("local_attention" in name for name in layer_names)


def test_full_multiscale_config_builds():
    config_path = (
        Path(__file__).parents[2]
        / "train_config"
        / "nn_config_300-2000bp_multiscale.yaml"
    )
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
        # Synthetic forward pass: (batch, frames, length) integer tokens
        x = tf.random.uniform((1, 6, 500), maxval=65, dtype=tf.int32)
        out = model(x)
        assert "prediction" in out
        assert out["prediction"].shape.as_list() == [1, 3]
