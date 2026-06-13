"""Integration test: build a model, save it, and run inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib import inference
from jaeger.postprocess import collect


@pytest.fixture
def tiny_config():
    return {
        "model": {
            "seed": 7,
            "class_label_map": {0: "bacteria", 1: "phage"},
            "embedding": {"type": "translated", "input_shape": (6, 16, 65)},
            "representation_learner": {
                "block_sizes": [1],
                "block_filters": [4],
                "block_kernel_size": [3],
                "block_kernel_dilation": [1],
                "block_kernel_strides": [1],
                "block_regularizer": ["l2"],
                "block_regularizer_w": [0.0],
                "pooling": "max",
                "masked_conv1d_1_filters": 4,
                "masked_conv1d_1_kernel_size": 3,
                "masked_conv1d_1_strides": 1,
                "masked_conv1d_1_dilation_rate": 1,
                "masked_conv1d_1_regularizer": "l2",
                "masked_conv1d_1_regularizer_w": 0.0,
                "masked_conv1d_final_kernel_size": 3,
                "masked_conv1d_final_strides": 1,
                "masked_conv1d_final_dilation_rate": 1,
            },
            "classifier": {
                "output_units": 2,
                "hidden_layers": [{"units": 4}],
            },
            "reliability_model": {
                "output_units": 1,
                "hidden_layers": [],
            },
        }
    }


@pytest.mark.integration
class TestInferencePipeline:
    def test_save_load_and_predict(self, tiny_config, tmp_path: Path):
        builder = inference.DynamicInferenceModelBuilder(tiny_config)
        model = builder.models["jaeger_model"]

        export_dir = tmp_path / "graph"
        tf.saved_model.save(model, str(export_dir))

        loaded = inference.InferModel(
            path_dict={
                "graph": str(export_dir),
                "classes": None,
                "project": None,
            }
        )

        x = {"translated": tf.random.normal((2, 6, 16, 65))}
        # InferModel.predict expects a dataset yielding (inputs, *meta)
        dataset = tf.data.Dataset.from_tensors(
            (x, np.array(["s1", "s2"]), np.array([1, 1]))
        )
        result = loaded.predict(dataset, no_progress=True)
        assert "prediction" in result
        assert result["prediction"].shape[0] == 2

    def test_end_to_end_postprocess(self, tiny_config, tmp_path: Path):
        builder = inference.DynamicInferenceModelBuilder(tiny_config)
        model = builder.models["jaeger_model"]

        # Create fake windowed data for two contigs.
        inputs = {"translated_input": tf.random.normal((4, 6, 16, 65))}
        y = model(inputs, training=False)

        y_pred = {
            "prediction": y["prediction"].numpy(),
            "reliability": y["reliability"].numpy(),
            "meta_0": np.array(["c1", "c1", "c2", "c2"]),
            "meta_2": np.array([0, 1, 0, 1]),
            "meta_4": np.array([500, 500, 500, 500]),
            "meta_5": np.full(4, 125.0, dtype=np.float32),
            "meta_6": np.full(4, 125.0, dtype=np.float32),
            "meta_7": np.full(4, 125.0, dtype=np.float32),
            "meta_8": np.full(4, 125.0, dtype=np.float32),
            "meta_9": np.zeros(4, dtype=np.float32),
        }

        import pandas as pd

        data, _ = collect.pred_to_dict(
            y_pred,
            fsize=500,
            class_map={"num_classes": 2},
            term_repeats=pd.DataFrame(
                {
                    "contig_id": ["c1", "c2"],
                    "terminal_repeats": [0, 0],
                    "repeat_length": [0, 0],
                }
            ),
        )
        assert len(data["headers"]) == 2
