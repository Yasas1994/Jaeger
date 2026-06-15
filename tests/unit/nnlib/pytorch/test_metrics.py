import torch
from jaeger.nnlib.pytorch.metrics import (
    BinaryAccuracy,
    CategoricalAccuracy,
    Precision,
    PrecisionForClass,
    Recall,
    RecallForClass,
    build_metrics,
)


def test_precision_for_class():
    metric = PrecisionForClass(class_id=1)
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    metric.update(preds, labels)
    result = metric.compute()
    assert 0.0 <= result <= 1.0


def test_recall_for_class():
    metric = RecallForClass(class_id=1)
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    metric.update(preds, labels)
    result = metric.compute()
    assert 0.0 <= result <= 1.0


def test_precision_for_class_zero_division():
    metric = PrecisionForClass(class_id=0)
    preds = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.float32)
    metric.update(preds, labels)
    result = metric.compute()
    assert result == 0.0


def test_metrics_multiple_updates():
    metric = PrecisionForClass(class_id=1)
    preds1 = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    labels1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    preds2 = torch.tensor([[0.3, 0.7], [0.9, 0.1]])
    labels2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    metric.update(preds1, labels1)
    metric.update(preds2, labels2)
    result = metric.compute()
    assert 0.0 <= result <= 1.0


def test_categorical_accuracy():
    metric = CategoricalAccuracy()
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    metric.update(preds, labels)
    result = metric.compute()
    # Predictions: class 1, class 0, class 1. Labels: class 1, class 0, class 1.
    assert result == 1.0


def test_binary_accuracy():
    metric = BinaryAccuracy()
    preds = torch.tensor([[2.0], [-2.0], [1.0], [-1.0]])
    labels = torch.tensor([[1], [0], [0], [1]], dtype=torch.float32)
    metric.update(preds, labels)
    result = metric.compute()
    # Predictions: 1, 0, 1, 0. Labels: 1, 0, 0, 1. 2/4 correct.
    assert result == 0.5


def test_overall_precision_and_recall():
    metric_precision = Precision(average="macro")
    metric_recall = Recall(average="macro")
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    labels = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.float32)
    metric_precision.update(preds, labels)
    metric_recall.update(preds, labels)
    assert 0.0 <= metric_precision.compute() <= 1.0
    assert 0.0 <= metric_recall.compute() <= 1.0


def test_build_metrics_from_config():
    metrics_cfg = [
        {"name": "categorical_accuracy", "params": None},
        {"name": "per_class_precision", "params": None},
    ]
    metrics = build_metrics(metrics_cfg, num_classes=3)
    assert "categorical_accuracy" in metrics
    assert "precision_class_0" in metrics
    assert "precision_class_1" in metrics
    assert "precision_class_2" in metrics


def test_build_metrics_unknown_metric_raises():
    metrics_cfg = [{"name": "not_a_metric", "params": None}]
    try:
        build_metrics(metrics_cfg, num_classes=2)
    except ValueError as exc:
        assert "Unsupported metric" in str(exc)
