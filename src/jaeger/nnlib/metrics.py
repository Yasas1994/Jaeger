import tensorflow as tf


class PrecisionForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id, name=None, **kwargs):
        if name is None:
            name = f"precision_class{class_id}"
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot → integers if needed
        if y_true.shape.rank > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_true_i = tf.equal(y_true, self.class_id)
        y_pred_i = tf.equal(y_pred, self.class_id)

        tp = tf.reduce_sum(tf.cast(y_true_i & y_pred_i, self.dtype))
        fp = tf.reduce_sum(tf.cast(~y_true_i & y_pred_i, self.dtype))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        return self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class RecallForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id, name=None, **kwargs):
        if name is None:
            name = f"recall_class{class_id}"
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_true.shape.rank > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_true_i = tf.equal(y_true, self.class_id)
        y_pred_i = tf.equal(y_pred, self.class_id)

        tp = tf.reduce_sum(tf.cast(y_true_i & y_pred_i, self.dtype))
        fn = tf.reduce_sum(tf.cast(y_true_i & ~y_pred_i, self.dtype))

        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)


class SpecificityForClass(tf.keras.metrics.Metric):
    def __init__(self, class_id, name=None, **kwargs):
        if name is None:
            name = f"specificity_class{class_id}"
        super().__init__(name=name, **kwargs)
        self.class_id = class_id
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot → integers if needed
        if y_true.shape.rank > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # "Not class_id" mask
        y_true_not_i = tf.not_equal(y_true, self.class_id)
        y_pred_not_i = tf.not_equal(y_pred, self.class_id)

        tn = tf.reduce_sum(tf.cast(y_true_not_i & y_pred_not_i, self.dtype))
        fp = tf.reduce_sum(tf.cast(y_true_not_i & ~y_pred_not_i, self.dtype))

        self.tn.assign_add(tn)
        self.fp.assign_add(fp)

    def result(self):
        return self.tn / (self.tn + self.fp + tf.keras.backend.epsilon())

    def reset_state(self):
        self.tn.assign(0.0)
        self.fp.assign(0.0)


class BinaryF1Score(tf.keras.metrics.Metric):
    """F1 score for a single-logit binary head (e.g. the reliability model).

    Thresholds the raw model output at ``threshold`` (default ``0.0``, i.e.
    ``sigmoid(logit) > 0.5``) and accumulates TP/FP/FN for the positive class.
    Unlike ``tf.keras.metrics.F1Score`` this accepts raw logits — the Keras
    metric requires ``0 < threshold <= 1`` (probabilities only).
    """

    def __init__(self, threshold: float = 0.0, name="binary_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) > self.threshold, tf.int32)

        tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), self.dtype))
        fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), self.dtype))
        fn = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), self.dtype))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        return (2 * self.tp) / (
            2 * self.tp + self.fp + self.fn + tf.keras.backend.epsilon()
        )

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


class MacroF1Score(tf.keras.metrics.Metric):
    """Macro-averaged F1 score for multi-class classification.

    Expects one-hot or integer labels and logits / probabilities.
    """

    def __init__(self, num_classes: int, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            name="tp", shape=(num_classes,), initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=(num_classes,), initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=(num_classes,), initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if y_true.shape.rank > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        elif y_true.shape.rank > 1 and y_true.shape[-1] == 1:
            # Binary labels are frequently stored as ``(batch, 1)``. Without
            # squeezing, ``tf.one_hot`` would emit a rank-3 tensor whose
            # broadcast against ``y_pred_one_hot`` produced a ``(batch, batch,
            # num_classes)`` update and crashed the ``assign_add`` on the
            # ``(num_classes,)`` state variable.
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_true_one_hot = tf.one_hot(y_true, self.num_classes, dtype=self.dtype)
        y_pred_one_hot = tf.one_hot(y_pred, self.num_classes, dtype=self.dtype)

        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        tp = self.true_positives
        fp = self.false_positives
        fn = self.false_negatives
        per_class_f1 = (2 * tp) / (2 * tp + fp + fn + tf.keras.backend.epsilon())
        return tf.reduce_mean(per_class_f1)

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config
