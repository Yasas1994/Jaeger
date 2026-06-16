"""Neural Mean Discrepancy layers."""

from __future__ import annotations

import tensorflow as tf


class NMDLayer(tf.keras.layers.Layer):
    """Compute a per-example channel mean-discrepancy vector.

    The output is the per-example channel mean (mask-aware) minus a reference
    mean. During training the reference mean is the current batch mean and the
    layer's moving mean is updated; during inference the moving mean is used.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        momentum: float = 0.9,
        dtype=None,
        **kwargs,
    ):
        if dtype is not None:
            kwargs["dtype"] = dtype
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.supports_masking = True

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        if channel_dim is None:
            raise ValueError("The last (channel) dimension must be defined.")
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(channel_dim,),
            initializer="zeros",
            trainable=False,
            dtype=self.variable_dtype,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        x = tf.cast(inputs, tf.float32)
        ndims = x.shape.rank
        if ndims is None:
            raise ValueError("Input rank must be statically known for NMDLayer.")

        example_axes = list(range(1, max(ndims - 1, 1)))

        if mask is not None:
            mask_f = tf.cast(mask, tf.float32)
            if mask_f.shape.rank is None or mask_f.shape.rank < ndims:
                mask_f = tf.expand_dims(mask_f, axis=-1)
            masked_inputs = x * mask_f
            per_ex_sum = tf.reduce_sum(masked_inputs, axis=example_axes)
            per_ex_count = tf.reduce_sum(mask_f, axis=example_axes) + self.epsilon
            mean_channel = per_ex_sum / per_ex_count
        else:
            mean_channel = tf.reduce_mean(x, axis=example_axes)

        mm = tf.cast(self.moving_mean, tf.float32)
        if training:
            mean_batch = tf.reduce_mean(mean_channel, axis=0)
            new_mm = self.momentum * mm + (1.0 - self.momentum) * mean_batch
            self.moving_mean.assign(tf.cast(new_mm, self.moving_mean.dtype))
            mean_to_use = mean_batch
        else:
            mean_to_use = mm

        nmd_f32 = mean_channel - mean_to_use
        return tf.cast(nmd_f32, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
            }
        )
        return config
