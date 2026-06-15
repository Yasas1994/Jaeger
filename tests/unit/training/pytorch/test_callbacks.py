import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from jaeger.training.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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
