import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

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


def _make_dummy_data(n=16):
    xs = torch.randn(n, 10)
    ys = torch.randint(0, 2, (n,))
    return DataLoader(_MaskedTensorDataset(xs, ys), batch_size=4)


def _make_dummy_model():
    return _DummyClassifier()


def test_trainer_fit():
    model = _make_dummy_model()
    train_loader = _make_dummy_data()
    val_loader = _make_dummy_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=2,
            device=torch.device("cpu"),
            history_path=Path(tmpdir) / "history.json",
        )
        history = trainer.fit()
        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]
        assert (Path(tmpdir) / "history.json").exists()


def test_trainer_fit_with_progress_bar():
    model = _make_dummy_model()
    train_loader = _make_dummy_data()
    val_loader = _make_dummy_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=1,
            device=torch.device("cpu"),
            history_path=Path(tmpdir) / "history.json",
            progress_bar=True,
        )
        history = trainer.fit()
        assert len(history) == 1
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]


class _CounterModel(_DummyClassifier):
    """Model that counts forward calls."""

    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, x, mask=None):
        self.forward_count += 1
        return super().forward(x, mask)


def test_trainer_respects_train_and_validation_steps():
    """Trainer should pass train_steps/validation_steps to the engine loops."""
    model = _CounterModel()
    train_loader = _make_dummy_data(n=20)  # 5 batches
    val_loader = _make_dummy_data(n=12)  # 3 batches
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=2,
            device=torch.device("cpu"),
            history_path=Path(tmpdir) / "history.json",
            train_steps=2,
            validation_steps=1,
        )
        history = trainer.fit()
        assert len(history) == 2
        # Each epoch runs train_steps + validation_steps forward passes.
        assert model.forward_count == 2 * (2 + 1)


def test_trainer_resumes_from_start_epoch():
    """Trainer should start training from start_epoch + 1."""
    model = _CounterModel()
    train_loader = _make_dummy_data()
    val_loader = _make_dummy_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=5,
            device=torch.device("cpu"),
            history_path=Path(tmpdir) / "history.json",
            start_epoch=2,
        )
        history = trainer.fit()
        # start_epoch=2 means epochs 3, 4, 5 are trained.
        assert len(history) == 3
        assert [entry["epoch"] for entry in history] == [3, 4, 5]


def test_trainer_skips_training_when_start_epoch_reaches_epochs():
    """Trainer should not train if start_epoch >= epochs."""
    model = _CounterModel()
    train_loader = _make_dummy_data()
    val_loader = _make_dummy_data()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=3,
            device=torch.device("cpu"),
            history_path=Path(tmpdir) / "history.json",
            start_epoch=3,
        )
        history = trainer.fit()
        assert len(history) == 0
        assert model.forward_count == 0
