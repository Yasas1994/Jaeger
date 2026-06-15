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
            checkpoint_dir=tmpdir,
            history_path=Path(tmpdir) / "history.json",
        )
        history = trainer.fit()
        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]
        assert (Path(tmpdir) / "history.json").exists()
        assert len(list(Path(tmpdir).glob("checkpoint_epoch_*.pt"))) == 2
