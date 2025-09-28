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