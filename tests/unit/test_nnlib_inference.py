"""Tests for jaeger.nnlib.inference."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.nnlib import inference


class TestEvaluate:
    def test_evaluate(self):
        inputs = tf.keras.Input(shape=(4,))
        logits = tf.keras.layers.Dense(2, name="prediction")(inputs)
        model = tf.keras.Model(inputs=inputs, outputs={"prediction": logits})

        x = tf.data.Dataset.from_tensor_slices(
            (np.random.normal(size=(8, 4)).astype(np.float32),
             np.eye(2, dtype=np.float32)[np.array([0, 1] * 4)])
        ).batch(2)

        result = inference.evaluate(model, x)
        assert "loss" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0


class TestDynamicInferenceModelBuilder:
    @staticmethod
    def _minimal_config():
        # Mirrors the config used in tests/smoke/infer_test.py.
        return {
            "model": {
                "seed": 7,
                "activation": "gelu",
                "class_label_map": {0: "bacteria", 1: "phage"},
                "embedding": {
                    "type": "translated",
                    "input_shape": (6, None, 64),
                    "embedding_size": 8,
                },
                "string_processor": {
                    "input_type": "translated",
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                    "seq_onehot": True,
                },
                "representation_learner": {
                    "masked_conv1d_1_filters": 16,
                    "masked_conv1d_1_kernel_size": 3,
                    "masked_conv1d_1_strides": 1,
                    "masked_conv1d_1_dilation_rate": 1,
                    "masked_conv1d_1_regularizer": None,
                    "masked_conv1d_1_regularizer_w": 0.0,
                    "block_sizes": [1],
                    "block_filters": [32],
                    "block_kernel_size": [3],
                    "block_kernel_dilation": [1],
                    "block_kernel_strides": [2],
                    "block_regularizer": [None],
                    "block_regularizer_w": [0.0],
                    "masked_conv1d_final_kernel_size": 1,
                    "masked_conv1d_final_strides": 1,
                    "masked_conv1d_final_dilation_rate": 1,
                    "masked_conv1d_final_regularizer": None,
                    "masked_conv1d_final_regularizer_w": 0.0,
                    "pooling": None,
                },
                "classifier": {
                    "hidden_layers": [
                        {"units": 16, "use_bias": True, "dropout_rate": 0.0}
                    ],
                    "output_units": 2,
                    "output_use_bias": True,
                    "output_activation": None,
                },
                "reliability_model": {
                    "hidden_layers": [
                        {"units": 16, "use_bias": True, "dropout_rate": 0.0}
                    ],
                    "output_units": 1,
                    "output_use_bias": True,
                    "output_activation": None,
                },
            }
        }

    def test_builds_models(self):
        cfg = self._minimal_config()
        builder = inference.DynamicInferenceModelBuilder(cfg)
        assert "classifier" in builder.models
        assert "reliability" in builder.models
        assert "jaeger_model" in builder.models

    def test_jaeger_model_inference(self):
        cfg = self._minimal_config()
        builder = inference.DynamicInferenceModelBuilder(cfg)
        model = builder.models["jaeger_model"]
        x = {"translated_input": tf.random.normal((2, 6, 96, 64))}
        y = model(x, training=False)
        assert "prediction" in y
        assert y["prediction"].shape.as_list()[:3] == [2, 6, 48]
