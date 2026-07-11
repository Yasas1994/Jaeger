from __future__ import annotations

import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402


# logits [-2, -0.3, 0.3, 2.0] with truth [0, 1, 1, 1]:
# threshold 0.0 -> pred [0, 0, 1, 1] -> tp=2, fp=0, fn=1 -> F1 = 4/5 = 0.8
LOGITS = tf.constant([[-2.0], [-0.3], [0.3], [2.0]])
Y_TRUE = tf.constant([[0.0], [1.0], [1.0], [1.0]])


def test_binary_f1_on_logits_threshold_zero():
    from jaeger.nnlib.metrics import BinaryF1Score

    m = BinaryF1Score()
    m.update_state(Y_TRUE, LOGITS)
    assert float(m.result()) == pytest.approx(0.8)


def test_binary_f1_accumulates_across_batches():
    from jaeger.nnlib.metrics import BinaryF1Score

    m = BinaryF1Score()
    m.update_state(Y_TRUE[:2], LOGITS[:2])  # tp=0, fp=0, fn=1
    m.update_state(Y_TRUE[2:], LOGITS[2:])  # tp=2, fp=0, fn=0
    assert float(m.result()) == pytest.approx(0.8)


def test_binary_f1_custom_threshold():
    from jaeger.nnlib.metrics import BinaryF1Score

    # threshold 0.5 (like Keras BinaryAccuracy default) -> pred [0,0,0,1]
    # -> tp=1, fp=0, fn=2 -> F1 = 2/4 = 0.5
    m = BinaryF1Score(threshold=0.5)
    m.update_state(Y_TRUE, LOGITS)
    assert float(m.result()) == pytest.approx(0.5)


def test_binary_f1_reset_state():
    from jaeger.nnlib.metrics import BinaryF1Score

    m = BinaryF1Score()
    m.update_state(Y_TRUE, LOGITS)
    m.reset_state()
    m.update_state(Y_TRUE[:1], LOGITS[:1])  # tn only -> tp=fp=fn=0
    assert float(m.result()) == pytest.approx(0.0)


def test_binary_f1_handles_flat_labels():
    from jaeger.nnlib.metrics import BinaryF1Score

    m = BinaryF1Score()
    m.update_state(tf.reshape(Y_TRUE, [-1]), LOGITS)
    assert float(m.result()) == pytest.approx(0.8)


def test_binary_f1_registered_in_metrics_factory():
    from jaeger.nnlib.builder import DynamicModelBuilder
    from jaeger.nnlib.metrics import BinaryF1Score

    metrics = DynamicModelBuilder.__new__(DynamicModelBuilder)._get_metrics(
        [{"name": "binary_f1", "params": None}]
    )
    assert len(metrics) == 1
    assert isinstance(metrics[0], BinaryF1Score)
    assert metrics[0].name == "binary_f1"
