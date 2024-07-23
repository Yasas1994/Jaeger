"""

Copyright (c) 2024 Yasas Wijesekara

"""

import tensorflow as tf
from tensorflow.keras import mixed_precision


class JaegerModel(tf.keras.Model):
    """
    Custom model for Jaeger with training, testing, and prediction steps.

    Methods:
        compile: Compiles the model with loss function, optimizer, and metrics.
        train_step: Performs a training step with gradient calculation and
                    weight updates.
        test_step: Performs a testing step with loss calculation and metric
                   updates.
        predict_step: Performs a prediction step with inference mode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_classes = None
        self.step = None
        self.loss_fn = None
        self.optimizer = None
        self.metrics_ = None
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")

    def compile(self, loss_fn, optimizer, metrics, num_classes):
        super().compile()

        self.num_classes = num_classes
        self.step = 0
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics_ = metrics

    def train_step(self, data):

        if len(data) == 3:
            # sample weights is class weights when a dictionary of class
            #  weights is provided to .fit
            x, y, sample_weights = data
        else:
            sample_weights = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss_scaled = self.loss_fn(y, y_pred["output"], sample_weights)
            loss_scaled += sum(self.losses)
            self.loss_tracker.update_state(loss_scaled)
            if mixed_precision.global_policy().name == "mixed_float16":
                loss_scaled = self.optimizer.get_scaled_loss(loss_scaled)

        grad = tape.gradient(target=loss_scaled,
                             sources=self.trainable_variables)
        if mixed_precision.global_policy().name == "mixed_float16":
            grad = self.optimizer.get_unscaled_gradients(grad)

        self.optimizer.apply_gradients(
            zip(grad, self.trainable_variables)
        )
        self.step += 1
        if self.step % 100 == 0:
            self.loss_tracker.reset_state()

        self.reg_loss_tracker.update_state(sum(self.losses))

        return {
            "loss": self.loss_tracker.result(),
            "reg-loss": self.reg_loss_tracker.result(),
            "lr": self.optimizer.learning_rate,
        }

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        y = tf.cast(y, dtype=y_pred["output"].dtype)

        # Updates the metrics tracking the loss
        loss = self.loss_fn(y, y_pred["output"])
        self.loss_tracker.update_state(loss)

        for m in self.metrics:
            if "loss" not in m.name:
                m.update_state(y, y_pred["output"])

        return {
            "loss": self.loss_tracker.result(),
            **{m.name: m.result() for m in self.metrics},
        }

    def predict_step(self, data):
        # Unpack the data
        x, y = data[0], data[1:]
        # set model to inference mode
        y_logits = self(x, training=False)
        return {"y_hat": y_logits, "meta": y}

    @property
    def metrics(self):
        out = []
        if hasattr(self, "loss_tracker"):
            out.append(self.loss_tracker)
        if hasattr(self, "attr_name"):
            out.extend(*self.metrics)
        return out
