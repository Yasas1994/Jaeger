"""Tests for jaeger.nnlib.metrics."""

from __future__ import annotations

import numpy as np
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
