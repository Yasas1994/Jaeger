"""Smoke tests for the PyTorch-based ``jaeger train`` command."""

from __future__ import annotations

import click
import numpy as np
import pytest
import torch
import yaml

from jaeger.commands.train import _load_checkpoint_if_requested, train_fragment_core


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
            "callbacks": {
                "classifier": [
                    {
                        "name": "ModelCheckpoint",
                        "params": {
                            "filepath": str(
                                tmp_path / "checkpoints" / "classifier" / "checkpoint_epoch_{epoch:02d}.pt"
                            ),
                            "monitor": "val_loss",
                            "mode": "min",
                            "save_best_only": False,
                            "verbose": 0,
                        },
                    }
                ]
            },
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


def test_train_fragment_core_logs_model_summary(tmp_path, monkeypatch, caplog):
    """``train_fragment_core`` should log a Keras-style model summary."""
    from jaeger.commands import train as train_module

    # The jaeger logger disables propagation; re-enable it so caplog can capture
    # the summary emitted by ``_print_model_summary``.
    monkeypatch.setattr(train_module.logger, "propagate", True)

    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    with caplog.at_level("INFO"):
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

    assert "classifier model summary" in caplog.text
    assert "Total params" in caplog.text


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


def test_train_fragment_core_refuses_overwrite_without_force_or_resume(tmp_path):
    """``train_fragment_core`` should stop if checkpoints exist and neither
    ``--force`` nor ``--from_last_checkpoint`` is given."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    classifier_dir = tmp_path / "checkpoints" / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    (classifier_dir / "checkpoint_epoch_1.pt").write_bytes(b"")

    with pytest.raises(click.ClickException):
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


def test_train_fragment_core_force_overwrites_existing_checkpoints(tmp_path):
    """``--force`` should remove existing checkpoints and train."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    classifier_dir = tmp_path / "checkpoints" / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    (classifier_dir / "checkpoint_epoch_1.pt").write_bytes(b"")

    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=True,
        only_classification_head=False,
        only_reliability_head=False,
        only_heads=False,
        only_save=False,
        save_model=False,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
    )

    # Directory was removed and a fresh checkpoint was written.
    assert len(list(classifier_dir.glob("checkpoint_epoch_*.pt"))) == 1


def test_train_fragment_core_resume_from_last_checkpoint(tmp_path):
    """``--from_last_checkpoint`` should load the latest checkpoint and train."""
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config_path.write_text(yaml.safe_dump(config))

    # First training run creates a checkpoint.
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

    checkpoints = list((tmp_path / "checkpoints" / "classifier").glob("*.pt"))
    assert len(checkpoints) == 1

    # Second run with from_last_checkpoint should not raise and should load
    # the existing checkpoint before training.
    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=True,
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

    checkpoints = list((tmp_path / "checkpoints" / "classifier").glob("*.pt"))
    assert len(checkpoints) >= 1


def test_train_fragment_core_resume_respects_checkpoint_epoch(tmp_path, monkeypatch):
    """``--from_last_checkpoint`` should pass the checkpoint epoch to Trainer."""
    from jaeger.training.pytorch import trainer as trainer_module

    captured = {}

    original_fit = trainer_module.Trainer.fit

    def _capturing_fit(self):
        captured["start_epoch"] = self.start_epoch
        return original_fit(self)

    monkeypatch.setattr(trainer_module.Trainer, "fit", _capturing_fit)

    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    _make_raw_npz(train_path, n_samples=8, seq_length=seq_length)
    _make_raw_npz(val_path, n_samples=4, seq_length=seq_length)

    config_path = tmp_path / "config.yaml"
    config = _build_config(tmp_path, train_path, val_path)
    config["training"]["classifier_epochs"] = 2
    config_path.write_text(yaml.safe_dump(config))

    # First run creates a checkpoint at epoch 2.
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

    # Resume and capture the start_epoch passed to Trainer.
    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=True,
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

    assert captured["start_epoch"] == 2


def test_load_checkpoint_maps_optimizer_state_to_device(tmp_path):
    """Resumed optimizer state should live on the target device."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Run one step so the optimizer has state.
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    # Load onto CPU (tests in CI may not have CUDA).
    target_device = torch.device("cpu")
    fresh_model = torch.nn.Linear(4, 2)
    fresh_model.to(target_device)
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
    _load_checkpoint_if_requested(
        fresh_model, fresh_optimizer, checkpoint_path, target_device
    )

    # Verify optimizer state tensors are on the target device.
    for state in fresh_optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device == target_device
