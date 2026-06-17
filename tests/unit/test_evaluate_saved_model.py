# tests/unit/test_evaluate_saved_model.py
import numpy as np

from scripts.evaluate_saved_model import compute_metrics


def test_compute_metrics_returns_confusion_matrix():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    metrics, cm = compute_metrics(y_true, y_pred, num_classes=3, return_cm=True)
    assert cm.shape == (3, 3)
    assert metrics["overall_accuracy"] == 0.5
