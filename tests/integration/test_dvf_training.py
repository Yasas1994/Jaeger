"""One-epoch DVF-style siamese training smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from jaeger.data import loaders
from jaeger.nnlib.builder import DynamicModelBuilder, set_global_seed


SEED = 42
NUM_SAMPLES = 32
SEQ_LEN = 64
BATCH_SIZE = 8
NUM_CLASSES = 3


@pytest.fixture
def dvf_config(tmp_path):
    """Minimal DVF-style config with shrunk filters/units."""
    return {
        "model": {
            "name": "dvf_smoke",
            "experiment": "dvf_smoke",
            "seed": SEED,
            "classifier_out_dim": NUM_CLASSES,
            "reliability_out_dim": 0,
            "class_label_map": [
                {"class": "chromosome", "label": 0},
                {"class": "virus", "label": 1},
                {"class": "plasmid", "label": 2},
            ],
            "embedding": {
                "use_embedding_layer": False,
                "input_type": "nucleotide",
                "strands": 2,
                "frames": 6,
                "input_shape": [2, SEQ_LEN, 4],
                "embedding_size": 0,
            },
            "string_processor": {
                "data_format": "numpy",
                "seq_onehot": False,
                "crop_size": SEQ_LEN,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "branch": {
                    "hidden_layers": [
                        {
                            "name": "conv1d",
                            "config": {"filters": 8, "kernel_size": 3},
                        },
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
                        {"name": "dense", "config": {"units": NUM_CLASSES}},
                        {"name": "merge", "config": {"method": "average"}},
                    ]
                }
            },
        },
        "training": {
            "optimizer": "adam",
            "optimizer_params": {"learning_rate": 0.01},
            "loss_classifier": "categorical_crossentropy",
            "loss_params_classifier": {"from_logits": True},
            "metrics_classifier": [{"name": "categorical_accuracy", "params": None}],
            "callbacks": {"directories": []},
            "model_saving": {
                "path": str(tmp_path / "model"),
                "save_weights": False,
                "save_exec_graph": False,
            },
            "fragment_classifier_data": {
                "train": [{"class": ["chromosome"], "label": [0], "path": []}]
            },
        },
    }


@pytest.fixture
def synthetic_nucleotide_npz(tmp_path):
    """Synthetic nucleotide one-hot NPZ for one-epoch training."""
    path = tmp_path / "synthetic_nucleotide.npz"
    nucleotide = np.eye(4, dtype=np.float32)[
        np.random.randint(0, 4, size=(NUM_SAMPLES, 2, SEQ_LEN))
    ]
    labels = np.random.randint(0, NUM_CLASSES, size=NUM_SAMPLES, dtype=np.int32)
    np.savez(path, nucleotide=nucleotide, labels=labels)
    return str(path)


def test_dvf_one_epoch_training(tmp_path, dvf_config, synthetic_nucleotide_npz):
    """Train a tiny DVF-style model for one epoch on a synthetic NPZ."""
    set_global_seed(SEED)

    builder = DynamicModelBuilder(dvf_config)
    models = builder.build_fragment_classifier()

    assert "jaeger_classifier" in models
    assert models["jaeger_classifier"].output.shape[-1] == NUM_CLASSES

    ds = loaders._load_numpy_dataset(
        synthetic_nucleotide_npz,
        input_type="nucleotide",
        seq_onehot=False,
        num_classes=NUM_CLASSES,
        one_hot_labels=True,
    ).batch(BATCH_SIZE)

    builder.compile_model(models, train_branch="classifier")
    history = models["jaeger_classifier"].fit(ds, epochs=1, verbose=0)

    assert history is not None
    assert "loss" in history.history
    loss = history.history["loss"][0]
    assert np.isfinite(loss), f"Training loss is not finite: {loss}"
