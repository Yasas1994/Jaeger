"""Smoke tests for convergence-aware resume."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from pathlib import Path

from jaeger import cli

HAS_TF = importlib.util.find_spec("tensorflow") is not None
pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")


def _make_synthetic_npz(path: Path, n: int = 128, length: int = 300) -> None:
    rng = np.random.default_rng(42)
    np.savez(
        path,
        translated=rng.integers(0, 65, size=(n, 6, length), dtype=np.int32),
        labels=rng.integers(0, 3, size=n, dtype=np.int32),
    )


def _write_resume_config(
    path: Path, base_dir: Path, data_dir: Path, epochs: int
) -> None:
    cfg = {
        "model": {
            "name": "jaeger_resume_smoke",
            "experiment": "resume_smoke",
            "seed": 42,
            "classifier_out_dim": 3,
            "reliability_out_dim": 0,
            "base_dir": str(base_dir),
            "activation": "gelu",
            "mode": "training",
            "embedding": {
                "use_embedding_layer": True,
                "input_type": "translated",
                "strands": 2,
                "frames": 6,
                "length": None,
                "input_shape": [6, None],
                "embedding_size": 16,
                "embedding_regularizer": "l2",
                "embedding_regularizer_w": 0.0001,
            },
            "string_processor": {
                "data_format": "numpy",
                "seq_onehot": False,
                "codon": "CODON",
                "codon_id": "CODON_ID",
                "crop_size": 100,
                "buffer_size": 1000,
                "shuffle": True,
                "reshuffle_each_iteration": True,
                "mutate": False,
                "shuffle_frames": False,
                "masking": False,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "hidden_layers": [
                    {
                        "name": "masked_conv1d",
                        "config": {
                            "filters": 8,
                            "kernel_size": 5,
                            "strides": 1,
                            "dilation_rate": 1,
                            "use_bias": True,
                            "activation": None,
                            "kernel_regularizer": "l2",
                            "kernel_regularizer_w": 0.0001,
                        },
                    },
                    {"name": "nmd"},
                    {"name": "masked_batchnorm", "config": {"return_nmd": False}},
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "max",
            },
            "classifier": {
                "input_shape": 8,
                "hidden_layers": [
                    {"name": "dropout", "config": {"rate": 0.1}},
                    {
                        "name": "dense",
                        "config": {
                            "units": 3,
                            "activation": None,
                            "dtype": "float32",
                            "use_bias": True,
                            "kernel_regularizer": "l2",
                            "kernel_regularizer_w": 0.0001,
                        },
                    },
                ],
            },
        },
        "training": {
            "data_dir": str(data_dir),
            "experiment_root": "experiments/experiment_{{ model.experiment }}_{{ model.seed }}",
            "classifier_dir": "{{ model.base_dir }}/{{ training.experiment_root }}/checkpoints/classifier",
            "reliability_dir": "{{ model.base_dir }}/{{ training.experiment_root }}/checkpoints/reliability",
            "classifier_epochs": epochs,
            "reliability_epochs": 0,
            "projection_epochs": 0,
            "classifier_train_steps": 1,
            "reliability_train_steps": 0,
            "classifier_validation_steps": 1,
            "reliability_validation_steps": 0,
            "batch_size": 32,
            "optimizer": "adamw",
            "optimizer_params": {"learning_rate": 0.001, "clipnorm": 5},
            "loss_classifier": "categorical_crossentropy",
            "loss_params_classifier": {"from_logits": True},
            "classifier_class_weights": {0: 1, 1: 1, 2: 1},
            "metrics_classifier": [{"name": "categorical_accuracy", "params": None}],
            "callbacks": {
                "directories": [
                    "{{ model.base_dir }}/{{ training.experiment_root }}/checkpoints/classifier"
                ],
                "classifier": [
                    {
                        "name": "EarlyStopping",
                        "params": {
                            "monitor": "val_loss",
                            "patience": 0,
                            "min_delta": 10.0,
                            "mode": "min",
                            "restore_best_weights": False,
                        },
                    },
                    {
                        "name": "ModelCheckpoint",
                        "params": {
                            "filepath": "{{ model.base_dir }}/{{ training.experiment_root }}/checkpoints/classifier/epoch:{epoch:02d}-loss:{val_loss:.2f}.weights.h5",
                            "monitor": "val_loss",
                            "mode": "min",
                            "save_weights_only": True,
                            "verbose": 0,
                            "save_best_only": False,
                        },
                    },
                    {"name": "TerminateOnNaN"},
                ],
            },
            "model_saving": {
                "path": "{{ model.base_dir }}/{{ training.experiment_root }}/model",
                "save_weights": False,
                "save_exec_graph": False,
            },
            "fragment_classifier_data": {
                "train": [
                    {
                        "class": ["c1", "c2", "c3"],
                        "path": ["{{ training.data_dir }}/train.npz"],
                        "label": [0, 1, 2],
                    }
                ],
                "validation": [
                    {
                        "class": ["c1", "c2", "c3"],
                        "path": ["{{ training.data_dir }}/val.npz"],
                        "label": [0, 1, 2],
                    }
                ],
            },
        },
    }
    path.write_text(yaml.dump(cfg, sort_keys=False))


def test_convergence_resume_skips_and_override():
    runner = CliRunner()
    with runner.isolated_filesystem() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        base_dir = tmp_path / "exp"
        _make_synthetic_npz(data_dir / "train.npz", n=64, length=300)
        _make_synthetic_npz(data_dir / "val.npz", n=32, length=300)

        config_path = tmp_path / "config.yaml"
        _write_resume_config(config_path, base_dir, data_dir, epochs=10)

        # First run: early stopping fires, marker written.
        result = runner.invoke(
            cli.main, ["train", "-c", str(config_path), "--force"]
        )
        assert result.exit_code == 0, result.output
        classifier_dir = (
            base_dir
            / "experiments"
            / "experiment_resume_smoke_42"
            / "checkpoints"
            / "classifier"
        )
        assert (classifier_dir / "converged.json").exists()
        first_checkpoints = sorted(classifier_dir.glob("epoch:*.h5"))
        assert len(first_checkpoints) >= 1

        # Resume without flag: should skip classifier training.
        result = runner.invoke(
            cli.main,
            [
                "train",
                "-c",
                str(config_path),
                "--from_last_checkpoint",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "already converged" in " ".join(result.output.lower().split())
        assert sorted(classifier_dir.glob("epoch:*.h5")) == first_checkpoints

        # Resume with --ignore_convergence: should train again.
        result = runner.invoke(
            cli.main,
            [
                "train",
                "-c",
                str(config_path),
                "--from_last_checkpoint",
                "--ignore_convergence",
            ],
        )
        assert result.exit_code == 0, result.output
        new_checkpoints = sorted(classifier_dir.glob("epoch:*.h5"))
        assert len(new_checkpoints) > len(first_checkpoints)
