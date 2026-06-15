import torch

from jaeger.training.pytorch.engine import evaluate, train_one_epoch


class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
        )

    def forward(self, x, mask=None):
        return self.net(x)


def _make_dummy_classifier():
    return DummyClassifier()


def _make_dummy_loader(n=16):
    xs = torch.randn(n, 10)
    ys = torch.randint(0, 2, (n,))
    masks = torch.ones(n, dtype=torch.bool)
    dataset = list(zip(xs, ys, masks))
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def test_train_one_epoch():
    model = _make_dummy_classifier()
    loader = _make_dummy_loader()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = {}
    history = train_one_epoch(
        model, loader, loss_fn, optimizer, torch.device("cpu"), metrics=metrics
    )
    assert "loss" in history


def test_evaluate():
    model = _make_dummy_classifier()
    loader = _make_dummy_loader()
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {}
    history = evaluate(model, loader, loss_fn, torch.device("cpu"), metrics=metrics)
    assert "loss" in history
