"""Tests for jaeger.nnlib.metrics."""

from __future__ import annotations

import tensorflow as tf

from jaeger.nnlib import metrics


class TestPerClassMetrics:
    def test_precision(self):
        y_true = tf.constant([0, 1, 1, 0, 1])
        y_pred = tf.constant(
            [[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]]
        )
        m = metrics.PrecisionForClass(class_id=1)
        m.update_state(y_true, y_pred)
        result = m.result().numpy()
        assert 0.0 <= result <= 1.0

    def test_recall(self):
        y_true = tf.constant([0, 1, 1, 0, 1])
        y_pred = tf.constant(
            [[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]]
        )
        m = metrics.RecallForClass(class_id=1)
        m.update_state(y_true, y_pred)
        result = m.result().numpy()
        assert 0.0 <= result <= 1.0

    def test_specificity(self):
        y_true = tf.constant([0, 1, 1, 0, 1])
        y_pred = tf.constant(
            [[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]]
        )
        m = metrics.SpecificityForClass(class_id=0)
        m.update_state(y_true, y_pred)
        result = m.result().numpy()
        assert 0.0 <= result <= 1.0

    def test_reset_state(self):
        y_true = tf.constant([0, 1, 1, 0, 1])
        y_pred = tf.constant(
            [[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]]
        )
        m = metrics.PrecisionForClass(class_id=1)
        m.update_state(y_true, y_pred)
        before = m.result().numpy()
        m.reset_state()
        after = m.result().numpy()
        assert before != after or after == 0.0


class TestMacroF1Score:
    def test_multiclass_onehot_perfect(self):
        # All classes present and predicted correctly -> macro F1 of 1.0.
        num_classes = 6
        y_true = tf.one_hot(range(num_classes), num_classes)
        y_pred = tf.one_hot(range(num_classes), num_classes)
        m = metrics.MacroF1Score(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        assert abs(m.result().numpy() - 1.0) < 1e-6

    def test_integer_labels_perfect(self):
        num_classes = 3
        y_true = tf.constant([0, 1, 2, 0, 1, 2])
        y_pred = tf.one_hot(y_true, num_classes)
        m = metrics.MacroF1Score(num_classes=num_classes)
        m.update_state(y_true, y_pred)
        assert abs(m.result().numpy() - 1.0) < 1e-6

    def test_binary_rank2_label_does_not_crash(self):
        # Reliability labels are stored as ``(batch, 1)`` while the metric
        # may be configured with the classifier's ``num_classes``. This used
        # to crash ``assign_add`` with a ``(batch, num_classes)`` vs
        # ``(num_classes,)`` shape mismatch.
        y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.int32)
        y_pred = tf.constant([[0.2], [0.7], [0.1], [0.9]])
        m = metrics.MacroF1Score(num_classes=6)
        m.update_state(y_true, y_pred)
        result = m.result().numpy()
        assert 0.0 <= result <= 1.0

    def test_binary_rank2_label_num_classes_two(self):
        y_true = tf.constant([[0], [1], [0], [1]], dtype=tf.int32)
        y_pred = tf.constant([[0.9], [0.9], [0.1], [0.1]])
        m = metrics.MacroF1Score(num_classes=2)
        m.update_state(y_true, y_pred)
        result = m.result().numpy()
        assert 0.0 <= result <= 1.0
