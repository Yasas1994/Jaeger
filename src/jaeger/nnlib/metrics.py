import tensorflow as tf

class Precision_per_class(tf.keras.metrics.Metric):

    def __init__(self, name="Precision_per_class", num_classes=4, **kwargs):
        super(Precision_per_class, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", shape=num_classes
        )
        self.pred_positives = self.add_weight(
            name="pp", initializer="zeros", shape=num_classes
        )
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.one_hot(y_pred, self.num_classes)

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)
        pp = tf.equal(y_pred, True)
        pp = tf.cast(pp, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, self.shape)
            tp = tf.multiply(tp, sample_weight)
            pp = tf.multiply(pp, sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(tp, axis=0))
        self.pred_positives.assign_add(tf.reduce_sum(pp, axis=0))

    def reset_state(self):
        self.true_positives.assign(tf.zeros(shape=self.num_classes))
        self.pred_positives.assign(tf.zeros(shape=self.num_classes))

    def result(self):
        result = tf.math.divide_no_nan(self.true_positives,
                                       self.pred_positives)
        return result


class Recall_per_class(tf.keras.metrics.Metric):

    def __init__(self, name="Recall_per_class", num_classes=4, **kwargs):
        super(Recall_per_class, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(
            name="tp", initializer="zeros", shape=num_classes
        )
        self.positives = self.add_weight(
            name="positives", initializer="zeros", shape=num_classes
        )
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred = tf.one_hot(y_pred, self.num_classes)

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.cast(tp, self.dtype)

        p = tf.equal(y_true, True)
        p = tf.cast(p, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, self.shape)
            tp = tf.multiply(tp, sample_weight)
            p = tf.multiply(p, sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(tp, axis=0))
        self.positives.assign_add(tf.reduce_sum(p, axis=0))

    def reset_state(self):
        self.true_positives.assign(tf.zeros(shape=self.num_classes))
        self.positives.assign(tf.zeros(shape=self.num_classes))

    def result(self):
        result = tf.math.divide_no_nan(self.true_positives, self.positives)

        return result