"""Integration test for short-fragment config support."""

from __future__ import annotations

from jaeger.nnlib.builder import DynamicModelBuilder


def test_local_attention_layer_registered():
    config = {
        "model": {
            "name": "test_local_attention",
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
                "path": "/tmp/jaeger_test",
                "save_weights": False,
                "save_exec_graph": False,
            },
            "fragment_classifier_data": {
                "train": [{"class": ["chromosome"], "label": [0], "path": []}]
            },
        },
        "config_path": "/tmp/jaeger_test_config.yaml",
    }
    builder = DynamicModelBuilder(config)
    models = builder.build_fragment_classifier()
    assert "jaeger_model" in models
