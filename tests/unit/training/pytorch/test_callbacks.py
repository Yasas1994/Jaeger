import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import json

from jaeger.training.pytorch.callbacks import (
    EarlyStopping,
    JsonLogger,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)
from jaeger.training.pytorch.trainer import Trainer


class _MaskedTensorDataset(TensorDataset):
    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, y, torch.ones(x.shape[0], dtype=torch.bool)


class _DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )

    def forward(self, x, mask=None):
        return self.net(x)


def _make_data(n=16):
    xs = torch.randn(n, 10)
    ys = torch.randint(0, 2, (n,))
    return DataLoader(_MaskedTensorDataset(xs, ys), batch_size=4)


def _make_model():
    return _DummyClassifier()


def test_early_stopping():
    model = _make_model()
    train_loader = _make_data()
    val_loader = _make_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=1, restore_best_weights=True
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=10,
        device=torch.device("cpu"),
        callbacks=[early_stop],
    )
    history = trainer.fit()
    assert len(history) < 10


def test_model_checkpoint():
    model = _make_model()
    train_loader = _make_data()
    val_loader = _make_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "best.pt"
        checkpoint = ModelCheckpoint(filepath=ckpt_path, monitor="val_loss", mode="min")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=2,
            device=torch.device("cpu"),
            callbacks=[checkpoint],
        )
        trainer.fit()
        assert ckpt_path.exists()


def test_reduce_lr_on_plateau_lowers_lr():
    """ReduceLROnPlateau should lower LR after patience epochs of no improvement."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", mode="min", patience=1, factor=0.1, min_lr=1e-6
    )

    class _FakeTrainer:
        pass

    fake_trainer = _FakeTrainer()
    fake_trainer.optimizer = optimizer

    # Improve once to set baseline.
    reduce_lr.on_epoch_end(fake_trainer, 1, {"val_loss": 1.0})
    assert optimizer.param_groups[0]["lr"] == 0.1

    # One epoch of no improvement triggers reduction (patience=1).
    reduce_lr.on_epoch_end(fake_trainer, 2, {"val_loss": 1.0})
    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-10

    # Further plateaus reduce again until min_lr is reached.
    reduce_lr.on_epoch_end(fake_trainer, 3, {"val_loss": 1.0})
    assert abs(optimizer.param_groups[0]["lr"] - 0.001) < 1e-10


def test_terminate_on_nan_stops_training():
    """TerminateOnNaN should stop training when the monitored metric is NaN."""
    model = _make_model()
    train_loader = _make_data()
    val_loader = _make_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    terminate = TerminateOnNaN(monitor="train_loss")

    # Monkeypatch train_one_epoch to return NaN loss on epoch 2.
    from jaeger.training.pytorch import trainer as trainer_module
    from jaeger.training.pytorch.engine import train_one_epoch as original_train

    def _nan_train(*args, **kwargs):
        return {"loss": float("nan")}

    original_evaluate = trainer_module.evaluate

    def _nan_evaluate(*args, **kwargs):
        return {"loss": float("nan")}

    trainer_module.train_one_epoch = _nan_train
    trainer_module.evaluate = _nan_evaluate
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=5,
            device=torch.device("cpu"),
            callbacks=[terminate],
        )
        history = trainer.fit()
        assert len(history) == 1
    finally:
        trainer_module.train_one_epoch = original_train
        trainer_module.evaluate = original_evaluate


def test_json_logger_writes_history():
    """JsonLogger should write per-epoch metrics to a JSON file."""
    model = _make_model()
    train_loader = _make_data()
    val_loader = _make_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        json_logger = JsonLogger(filename=str(history_path))
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=2,
            device=torch.device("cpu"),
            callbacks=[json_logger],
        )
        trainer.fit()
        assert history_path.exists()
        with history_path.open() as fh:
            saved = json.load(fh)
        assert len(saved) == 2
        assert "train_loss" in saved[0]
        assert "val_loss" in saved[0]


def test_json_logger_appends_to_existing_history():
    """JsonLogger with append=True should extend an existing history file."""
    model = _make_model()
    train_loader = _make_data()
    val_loader = _make_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "history.json"
        history_path.write_text(json.dumps([{"epoch": 1, "train_loss": 0.9}]))

        json_logger = JsonLogger(filename=str(history_path), append=True)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=2,
            device=torch.device("cpu"),
            callbacks=[json_logger],
        )
        trainer.fit()

        with history_path.open() as fh:
            saved = json.load(fh)
        assert len(saved) == 3
        assert saved[0]["epoch"] == 1
