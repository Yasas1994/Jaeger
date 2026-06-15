import torch
from rich.progress import Progress

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


def test_train_one_epoch_with_progress():
    model = _make_dummy_classifier()
    loader = _make_dummy_loader()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with Progress(transient=True) as progress:
        task_id = progress.add_task("train", total=len(loader))
        history = train_one_epoch(
            model,
            loader,
            loss_fn,
            optimizer,
            torch.device("cpu"),
            progress=progress,
            task_id=task_id,
        )
    assert "loss" in history


def test_evaluate_with_progress():
    model = _make_dummy_classifier()
    loader = _make_dummy_loader()
    loss_fn = torch.nn.CrossEntropyLoss()
    with Progress(transient=True) as progress:
        task_id = progress.add_task("val", total=len(loader))
        history = evaluate(
            model, loader, loss_fn, torch.device("cpu"), progress=progress, task_id=task_id
        )
    assert "loss" in history


def test_train_one_epoch_progress_shows_cumulative_mean():
    """The progress bar description should show the cumulative mean loss so far."""
    model = _make_dummy_classifier()
    loader = _make_dummy_loader(n=8)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with Progress(transient=True) as progress:
        task_id = progress.add_task("train", total=len(loader))
        train_one_epoch(
            model,
            loader,
            loss_fn,
            optimizer,
            torch.device("cpu"),
            progress=progress,
            task_id=task_id,
        )
        description = progress.tasks[task_id].description
    assert description.startswith("loss=")
    loss_value = float(description.split("=")[1])
    assert loss_value > 0


def test_train_one_epoch_profile_returns_timings():
    """Profiling should return per-section timing averages."""
    model = _make_dummy_classifier()
    loader = _make_dummy_loader()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    history = train_one_epoch(
        model, loader, loss_fn, optimizer, torch.device("cpu"), profile=True
    )
    assert "time_data_ms" in history
    assert "time_forward_ms" in history
    assert "time_backward_ms" in history
    assert "time_optim_ms" in history
    assert "time_metrics_ms" in history
    assert all(history[k] >= 0 for k in history if k.startswith("time_"))


def test_evaluate_profile_returns_timings():
    """Profiling should return per-section timing averages for evaluation."""
    model = _make_dummy_classifier()
    loader = _make_dummy_loader()
    loss_fn = torch.nn.CrossEntropyLoss()
    history = evaluate(model, loader, loss_fn, torch.device("cpu"), profile=True)
    assert "time_data_ms" in history
    assert "time_forward_ms" in history
    assert "time_metrics_ms" in history
    assert "time_backward_ms" not in history


class _CounterClassifier(DummyClassifier):
    """Classifier that counts forward calls."""

    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, x, mask=None):
        self.forward_count += 1
        return super().forward(x, mask)


def test_train_one_epoch_respects_train_steps():
    """train_one_epoch should stop after train_steps batches."""
    model = _CounterClassifier()
    loader = _make_dummy_loader(n=20)  # 5 batches of 4
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    history = train_one_epoch(
        model, loader, loss_fn, optimizer, torch.device("cpu"), train_steps=2
    )
    assert "loss" in history
    assert model.forward_count == 2


def test_train_one_epoch_negative_train_steps_runs_all():
    """Negative train_steps should run through the whole dataloader."""
    model = _CounterClassifier()
    loader = _make_dummy_loader(n=20)  # 5 batches
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    history = train_one_epoch(
        model, loader, loss_fn, optimizer, torch.device("cpu"), train_steps=-1
    )
    assert "loss" in history
    assert model.forward_count == 5


def test_evaluate_respects_validation_steps():
    """evaluate should stop after validation_steps batches."""
    model = _CounterClassifier()
    loader = _make_dummy_loader(n=20)  # 5 batches of 4
    loss_fn = torch.nn.CrossEntropyLoss()
    history = evaluate(
        model, loader, loss_fn, torch.device("cpu"), validation_steps=2
    )
    assert "loss" in history
    assert model.forward_count == 2
