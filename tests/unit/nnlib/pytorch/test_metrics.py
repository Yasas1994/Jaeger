import torch
from jaeger.nnlib.pytorch.metrics import PrecisionForClass, RecallForClass


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
