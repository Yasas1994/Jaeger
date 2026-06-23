import numpy as np
import tensorflow as tf

from jaeger.nnlib.metrics import MacroF1Score


class TestMacroF1Score:
    def test_perfect_prediction(self):
        metric = MacroF1Score(num_classes=3)
        y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        y_pred = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()
        assert np.isclose(result.numpy(), 1.0)

    def test_zero_prediction(self):
        metric = MacroF1Score(num_classes=3)
        y_true = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 0], [1, 0, 0]], dtype=tf.float32)
        metric.update_state(y_true, y_pred)
        result = metric.result()
        assert result.numpy() >= 0.0

    def test_integer_labels(self):
        metric = MacroF1Score(num_classes=3)
        y_true = tf.constant([0, 1, 2, 0], dtype=tf.int32)
        y_pred = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=tf.float32
        )
        metric.update_state(y_true, y_pred)
        result = metric.result()
        assert np.isclose(result.numpy(), 1.0)

    def test_reset_state(self):
        metric = MacroF1Score(num_classes=3)
        y_true = tf.constant([[1, 0, 0]], dtype=tf.float32)
        y_pred = tf.constant([[0, 1, 0]], dtype=tf.float32)
        metric.update_state(y_true, y_pred)
        metric.reset_state()
        result = metric.result()
        assert np.isclose(result.numpy(), 0.0)

    def test_get_config(self):
        metric = MacroF1Score(num_classes=5, name="my_macro_f1")
        config = metric.get_config()
        assert config["num_classes"] == 5
        assert config["name"] == "my_macro_f1"
