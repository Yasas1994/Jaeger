"""Smoke tests for the PyTorch-based ``jaeger train`` command."""

from __future__ import annotations

import numpy as np
import yaml

from jaeger.commands.train import train_fragment_core


def _make_raw_npz(path, n_samples, seq_length=50):
    """Create a minimal numpy_raw dataset archive."""
    seqs = np.random.randint(0, 4, size=(n_samples, seq_length)).astype(np.int8)
    labels = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, size=n_samples)]
    np.savez(path, sequences=seqs, labels=labels)


def _build_config(tmp_path, train_path, val_path, data_format="numpy_raw"):
    return {
        "model": {
            "name": "test_pytorch_train",
            "classifier_out_dim": 3,
            "reliability_out_dim": 0,
            "embedding": {
                "input_type": "translated",
                "use_embedding_layer": True,
                "vocab_size": 65,
                "embedding_size": 4,
            },
            "string_processor": {
                "data_format": data_format,
                "crop_size": seq_length,
                "shuffle": False,
                "mutate": False,
                "shuffle_frames": False,
            },
            "representation_learner": {
                "hidden_layers": [],
                "pooling": "average",
            },
            "classifier": {
                "input_shape": 4,
                "hidden_layers": [{"name": "dense", "config": {"units": 3}}],
            },
        },
        "training": {
            "batch_size": 4,
            "classifier_epochs": 1,
            "classifier_dir": str(tmp_path / "checkpoints" / "classifier"),
            "optimizer": "adam",
            "optimizer_params": {"lr": 1e-3},
            "fragment_classifier_data": {
                "train": [{"path": [str(train_path)]}],
                "validation": [{"path": [str(val_path)]}],
            },
            "model_saving": {
                "path": str(tmp_path / "model"),
            },
        },
    }


seq_length = 50


def test_train_fragment_core_runs(tmp_path):
    """``train_fragment_core`` should complete a single epoch without error."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=False,
        only_classification_head=False,
        only_reliability_head=False,
        only_heads=False,
        only_save=False,
        save_model=False,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
    )


def test_train_fragment_core_only_save(tmp_path):
    """``train_fragment_core`` should save a checkpoint without training."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=False,
        only_classification_head=False,
        only_reliability_head=False,
        only_heads=False,
        only_save=True,
        save_model=False,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
    )

    checkpoint_path = tmp_path / "model" / "classifier.pt"
    assert checkpoint_path.exists()


def test_train_fragment_core_only_classification_head(tmp_path):
    """``train_fragment_core`` should freeze the representation learner when asked."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=False,
        only_classification_head=True,
        only_reliability_head=False,
        only_heads=False,
        only_save=False,
        save_model=False,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
    )


def test_train_fragment_core_progress_bar(tmp_path):
    """``train_fragment_core`` should support the ``--progress-bar`` path."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=False,
        only_classification_head=False,
        only_reliability_head=False,
        only_heads=False,
        only_save=False,
        save_model=False,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
        progress_bar=True,
    )


def test_train_fragment_core_respects_config_steps(tmp_path, monkeypatch):
    """``train_fragment_core`` should pass config train/validation steps to Trainer."""
    from jaeger.training.pytorch import trainer as trainer_module

    captured = {}

    original_fit = trainer_module.Trainer.fit

    def _capturing_fit(self):
        captured["train_steps"] = self.train_steps
        captured["validation_steps"] = self.validation_steps
        return original_fit(self)

    monkeypatch.setattr(trainer_module.Trainer, "fit", _capturing_fit)

    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config["training"]["classifier_train_steps"] = 2
    config["training"]["classifier_validation_steps"] = 1
    config_path.write_text(yaml.safe_dump(config))

    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=False,
        only_classification_head=False,
        only_reliability_head=False,
        only_heads=False,
        only_save=False,
        save_model=False,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
    )

    assert captured["train_steps"] == 2
    assert captured["validation_steps"] == 1
