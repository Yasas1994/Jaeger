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

        reduce_axes = list(range(0, max(ndims - 1, 0)))
        example_axes = list(range(1, max(ndims - 1, 1)))

        if mask is not None:
            mask_f = tf.cast(mask, tf.float32)
            if mask_f.shape.rank is None or mask_f.shape.rank < ndims:
                mask_f = tf.expand_dims(mask_f, axis=-1)
            masked_inputs = x * mask_f

            valid_elements = tf.reduce_sum(mask_f, axis=reduce_axes) + self.epsilon
            mean_batch = tf.reduce_sum(masked_inputs, axis=reduce_axes) / valid_elements

            per_ex_sum = tf.reduce_sum(masked_inputs, axis=example_axes)
            per_ex_count = tf.reduce_sum(mask_f, axis=example_axes) + self.epsilon
            mean_channel = per_ex_sum / per_ex_count
        else:
            mean_batch = tf.reduce_mean(x, axis=reduce_axes)
            mean_channel = tf.reduce_mean(x, axis=example_axes)

        mm = tf.cast(self.moving_mean, tf.float32)
        if training:
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


class NMDMerge(tf.keras.layers.Layer):
    """Merge a list of NMD vectors into a single tensor."""

    def __init__(
        self,
        mode: str = "concat",
        axis: int = -1,
        target_dim: int | None = None,
        projection_kwargs: dict | None = None,
        dtype=None,
        **kwargs,
    ):
        if dtype is not None:
            kwargs["dtype"] = dtype
        super().__init__(**kwargs)
        if mode not in {"concat", "sum", "mean", "max", "weighted"}:
            raise ValueError(f"Unsupported NMD merge mode: {mode}")
        self.mode = mode
        self.axis = axis
        self.target_dim = target_dim
        self.projection_kwargs = dict(projection_kwargs or {})

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        self._num_inputs = len(input_shape)
        dims = [int(shape[-1]) for shape in input_shape]

        if self.mode == "concat":
            self._output_dim = sum(dims)
        else:
            if self.target_dim is None:
                if len(set(dims)) == 1:
                    self.target_dim = dims[0]
                else:
                    raise ValueError(
                        f"target_dim is required for merge mode '{self.mode}' "
                        "when NMD channel dimensions differ."
                    )
            self._output_dim = self.target_dim
            self.projections = [
                tf.keras.layers.Dense(
                    self.target_dim,
                    use_bias=False,
                    name=f"proj_{i}",
                    **self.projection_kwargs,
                )
                for i in range(self._num_inputs)
            ]
            if self.mode == "weighted":
                self.layer_weights = self.add_weight(
                    name="layer_weights",
                    shape=(self._num_inputs,),
                    initializer="ones",
                    trainable=True,
                )
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.mode == "concat":
            return tf.concat(inputs, axis=self.axis)

        projected = [proj(inp) for proj, inp in zip(self.projections, inputs)]

        if self.mode == "sum":
            return tf.add_n(projected)
        if self.mode == "mean":
            return tf.add_n(projected) / tf.cast(self._num_inputs, projected[0].dtype)
        if self.mode == "max":
            stacked = tf.stack(projected, axis=self.axis)
            return tf.reduce_max(stacked, axis=self.axis)
        if self.mode == "weighted":
            stacked = tf.stack(projected, axis=0)
            weights = tf.reshape(tf.nn.softmax(self.layer_weights), [-1, 1, 1])
            return tf.reduce_sum(stacked * weights, axis=0)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        batch = input_shape[0][0]
        if self.mode == "concat":
            channels = sum(int(shape[-1]) for shape in input_shape)
        else:
            channels = self.target_dim
        return (batch, channels)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "axis": self.axis,
                "target_dim": self.target_dim,
                "projection_kwargs": self.projection_kwargs,
            }
        )
        return config
