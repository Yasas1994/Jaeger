import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from jaeger.commands.predict import _aggregate_multiscale_data
from jaeger.nnlib.inference import DynamicInferenceModelBuilder, InferModel, evaluate

# ─── Test 1: DynamicInferenceModelBuilder ──────────────────────────────────

config = {
    "model": {
        "seed": 7,
        "activation": "gelu",
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
            "pooling": "max",
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

builder = DynamicInferenceModelBuilder(config)
model = builder.models["jaeger_model"]

x = tf.random.normal([4, 6, 96, 64])
y = tf.one_hot([0, 1, 0, 1], depth=2)

out = model(x, training=False)

print("Model output keys:", out.keys())
print("Prediction shape:", out["prediction"].shape)
print("Reliability shape:", out["reliability"].shape)

ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

metrics = evaluate(model, ds, prediction_key="prediction", no_progress=True)

print("Eval metrics:", metrics)

assert out["prediction"].shape == (4, 2)
assert out["reliability"].shape[0] == 4
assert "loss" in metrics
assert "accuracy" in metrics

# ─── Test 2: InferModel auto-corrects seq_onehot from SavedModel signature ─

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    classes_file = tmpdir / "classes.yaml"
    project_file = tmpdir / "project.yaml"

    # Use the pre-trained model graph
    graph_dir = Path("models/experiment_048_57341/model/jaeger_048_1.5M_fragment_graph")

    # Write class map
    classes_file.write_text(yaml.safe_dump({
        "classes": [
            {"class": "bacteria", "label": 0},
            {"class": "phage", "label": 1},
        ]
    }))

    # Write project WITHOUT seq_onehot (simulates old configs)
    project = {
        "model": {
            "embedding": config["model"]["embedding"],
            "string_processor": {
                "input_type": "translated",
                "codon": "CODON",
                "codon_id": "CODON_ID",
                # intentionally omit seq_onehot
            },
        }
    }
    project_file.write_text(yaml.safe_dump(project))

    infer_model = InferModel({
        "graph": graph_dir,
        "classes": classes_file,
        "project": project_file,
    })

    # Without auto-correction this would be False / 1
    assert infer_model.string_processor_config["seq_onehot"] is True, \
        f"Expected seq_onehot=True, got {infer_model.string_processor_config['seq_onehot']}"
    assert infer_model.string_processor_config["codon_depth"] == 64, \
        f"Expected codon_depth=64, got {infer_model.string_processor_config['codon_depth']}"

    print("InferModel auto-correction test passed")

# ─── Test 3: _aggregate_multiscale_data ────────────────────────────────────

num_classes = 6
class_map = {"num_classes": num_classes}

# Mock two scales with 2 contigs each (multi-class, 6 classes)
scale1_data = {
    "headers": ["c1", "c2"],
    "length": [1000, 2000],
    "ood": np.array([0.8, 0.3], dtype=np.float16),
    "gc": [np.array([0.5, 0.5]), np.array([0.4])],
    "ns": [np.array([0.1, 0.1]), np.array([0.2])],
    "repeats": None,
}
scale1_full = {
    "predictions": [
        np.array([
            [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
        ]),  # c1: 2 windows, phage-heavy
        np.array([
            [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
        ]),  # c2: 1 window, bacteria-heavy
    ],
    "headers": ["c1", "c2"],
    "lengths": [1000, 2000],
    "gc_skews": None,
    "gcs": [np.array([0.5, 0.5]), np.array([0.4])],
}

scale2_data = {
    "headers": ["c1", "c2"],
    "length": [1000, 2000],
    "ood": np.array([0.7, 0.4], dtype=np.float16),
    "gc": [np.array([0.55, 0.55]), np.array([0.45])],
    "ns": [np.array([0.15, 0.15]), np.array([0.25])],
    "repeats": None,
}
scale2_full = {
    "predictions": [
        np.array([
            [0.15, 0.85, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0, 0.0, 0.0],
        ]),  # c1: 2 windows, phage-heavy
        np.array([
            [0.55, 0.45, 0.0, 0.0, 0.0, 0.0],
        ]),  # c2: 1 window, bacteria-heavy
    ],
    "headers": ["c1", "c2"],
    "lengths": [1000, 2000],
    "gc_skews": None,
    "gcs": [np.array([0.55, 0.55]), np.array([0.45])],
}

aggregated_data, _ = _aggregate_multiscale_data(
    [(scale1_data, scale1_full), (scale2_data, scale2_full)],
    class_map=class_map,
    term_repeats=None,
)

assert len(aggregated_data["headers"]) == 2
assert aggregated_data["ood"].shape == (2,)
# OOD should be averaged across scales
assert np.isclose(aggregated_data["ood"][0], 0.75, atol=0.01)
assert np.isclose(aggregated_data["ood"][1], 0.35, atol=0.01)
# c1 should be phage (class 1) because predictions favor phage
assert aggregated_data["consensus"][0] == 1
# c2 should be bacteria (class 0)
assert aggregated_data["consensus"][1] == 0
# pooled predictions for c1 should have 4 windows (2 from each scale)
assert len(aggregated_data["frag_pred"][0]) == 4

print("Multi-scale aggregation test passed")

print("All checks passed.")
