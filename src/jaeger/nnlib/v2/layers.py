import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
Author: yasas.wijesekara@uni-greifswald.de

This package provides layers for building deep learning models with biological sequences.
1D convolution is commonplacely used in modelling biological sequences (amino acid
and nucleotide sequences). However, current implementations of 1D convolutional layers
do not support masking which is important as some regions of genomes are either unknown 
or has very low complexity, and these regions can interfere with the training process.
Here, we provide layers that support masking and can we incooperated into your projects
by simply adding this file to your project. If you find any bugs please contact me or fix 
it and send me a pull request :)
"""


## Activations
class GeLU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        # Use the tanh approximation so the graph can be converted to TFLite.
        return tf.nn.gelu(inputs, approximate=True)

    def compute_output_shape(self, input_shape):
        return input_shape


class ReLU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True

    def call(self, inputs):
        return tf.nn.relu(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class SumStands(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.reduce_sum(inputs, axis=1, keepdims=False)

    def get_config(self):
        config = super().get_config()
        return config


class MaskedAdd(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedAdd, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        # inputs: list of tensors
        output = tf.math.add_n(inputs)
        return output

    def compute_mask(self, inputs, mask=None):
        # If all inputs share the same mask (e.g., coming from the same encoder), return that
        if mask is None:
            return None
        if isinstance(mask, list):
            # Combine masks (logical AND or OR, depending on use case)
            return mask[0]  # or tf.logical_and(mask[0], mask[1]), etc.
        return mask


class CustomPooling1D(tf.keras.layers.Layer):
    """Apply 1D pooling along a defined axis"""

    def __init__(self, pool_size, axis, **kwargs):
        super(CustomPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.axis = axis

    def call(self, inputs):
        return tf.nn.pool(
            inputs,
            window_shape=[1, self.pool_size, 1, 1],
            strides=[1, self.pool_size, 1, 1],
            pooling_type="MAX",
            padding="SAME",
            data_format="NHWC",
        )

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = (
            input_shape[self.axis] - self.pool_size
        ) // self.pool_size + 1
        return tuple(output_shape)


class GlobalMaxPoolingPerFeature(tf.keras.layers.Layer):
    """'Apply max_reduce along the last axis. Re-implementation of GlobalMaxPooling1D layer for
    Biological sequences"""

    def __init__(self, **kwargs):
        super(GlobalMaxPoolingPerFeature, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the feature axis
        return tf.reduce_max(
            inputs, axis=-1, keepdims=False, name="global_max_per_position"
        )

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0], input_shape[2])


class MaxReduce(tf.keras.layers.Layer):
    """Apply max_reduce along the frame axis"""

    def __init__(self, **kwargs):
        super(MaxReduce, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the frame axis
        return tf.reduce_max(inputs, axis=1, keepdims=False, name="max_reduce")

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0], input_shape[2], input_shape[3])


class MeanReduce(tf.keras.layers.Layer):
    """Apply mean_reduce along the frame axis"""

    def __init__(self, **kwargs):
        super(MeanReduce, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the frame axis
        return tf.reduce_mean(inputs, axis=1, keepdims=False, name="mean_reduce")

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0], input_shape[2], input_shape[3])


class MaskedNeuralMean(tf.keras.layers.Layer):
    """
    Masked Neural Mean layer returns the channelwise mean of in the input
    tensor [batch, strands, length, channels]
    This layer has no parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None, training=False):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # Ensure mask is the same type as inputs
            mask = tf.expand_dims(
                mask, axis=-1
            )  # Broadcast mask to have the same number of channels as inputs
            valid_elements = tf.reduce_sum(
                mask, axis=[0, 1, 2]
            )  # Count valid elements per feature map

            # Apply mask to inputs to ignore padded elements
            masked_inputs = inputs * mask

            # Calculate mean and variance only over valid elements
            mean = tf.reduce_sum(masked_inputs, axis=[0, 1, 2]) / valid_elements
            # variance = tf.reduce_sum(mask * tf.square(masked_inputs - mean), axis=[0, 1, 2]) / valid_elements

        else:
            # Standard batch normalization when no mask is provided
            mean, _ = tf.nn.moments(inputs, axes=[0, 1, 2])

        return mean


class SumReduce(tf.keras.layers.Layer):
    """Apply sum_reduce along the frame axis"""

    def __init__(self, **kwargs):
        super(SumReduce, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the frame axis
        return tf.reduce_sum(inputs, axis=1, keepdims=False, name="max_reduce")

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0], input_shape[2], input_shape[3])


class MaskedMaxPooling1D(tf.keras.layers.Layer):
    """
    MaxPooling1D implementation that accepts an incomming masking tensor
    and outputs a mask tensor corresponding to the output dimention of layer's
    output
    """

    def __init__(self, pool_size=2, strides=None, padding="valid", **kwargs):
        super(MaskedMaxPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.lower()

    def call(self, inputs, mask=None):
        # Reshape for MaskedMaxpooling1D
        input_shape = tf.shape(inputs)
        self._input_dtype = inputs.dtype
        # batch_size = input_shape[0]
        # dim1 = input_shape[1]
        dim2 = input_shape[2]
        dim3 = input_shape[3]

        if mask is not None:
            # Expand mask dimensions to match inputs
            mask = tf.cast(mask, dtype=inputs.dtype)
            inputs = inputs * tf.expand_dims(mask, axis=-1)  # Zero out masked positions

        reshaped_inputs = tf.reshape(inputs, (-1, dim2, dim3))
        outputs = tf.nn.max_pool1d(
            reshaped_inputs,
            ksize=self.pool_size,
            strides=self.strides,
            padding=self.padding.upper(),
        )

        # Reshape back to the original dimensions
        output_shape = self.compute_output_shape(input_shape)
        outputs = tf.reshape(outputs, output_shape)

        if mask is not None:
            reshaped_mask = tf.reshape(mask, (-1, dim2))
            mask = self.compute_output_mask(reshaped_mask)
            self._output_mask = tf.reshape(mask, output_shape[:-1])

        else:
            self._output_mask = None

        return outputs

    def compute_mask(self, inputs, mask=None):
        # Return the output mask computed in call()
        if hasattr(self, "_output_mask"):
            return self._output_mask
        else:
            return None

    def compute_output_mask(self, mask=None):
        if mask is None:
            return None
        else:
            mask = tf.cast(mask, dtype=self._input_dtype)
            mask = tf.expand_dims(mask, axis=-1)

            pooled_mask = tf.nn.max_pool1d(
                mask,
                ksize=self.pool_size,
                strides=self.strides,
                padding=self.padding.upper(),
            )
            pooled_mask = tf.cast(pooled_mask, dtype=tf.bool)

            # pooled_mask = tf.squeeze(pooled_mask, axis=-1)
            return pooled_mask

    def compute_output_shape(self, input_shape):
        # Compute the output shape along the pooling axis
        axis_length = input_shape[2]
        if axis_length is not None:
            if self.padding == "valid":
                output_length = (axis_length - self.pool_size) // self.strides + 1
            elif self.padding == "same":
                output_length = (axis_length + self.strides - 1) // self.strides

            # Create the new shape and replace the dimension for the pooled axis

            return input_shape[0], input_shape[1], output_length, input_shape[-1]
        else:
            return input_shape[0], input_shape[1], None, input_shape[-1]


class MaskedLayerNormalization(tf.keras.layers.Layer):
    """
    Masked layer normalization ignores the masked positions when calculating the summary
    statistics and normalization
    """

    def __init__(self, epsilon=1e-3, center=True, scale=True, **kwargs):
        super(MaskedLayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.supports_masking = True

    def build(self, input_shape):
        param_shape = input_shape[-1:]

        if self.scale:
            self.gamma = self.add_weight(
                name="gamma", shape=param_shape, initializer="ones", trainable=True
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta", shape=param_shape, initializer="zeros", trainable=True
            )
        else:
            self.beta = None

        super(MaskedLayerNormalization, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Layer-normalization statistics are prone to float16 overflow when the
        # channel values are large, so compute the moments in float32 and cast
        # the result back to the layer's compute dtype.
        compute_dtype = tf.float32
        if mask is not None:
            mask = tf.cast(mask, compute_dtype)
            mask = tf.expand_dims(mask, -1)
            mask = tf.stop_gradient(mask)
            # Zero masked positions so they do not affect per-position channel stats
            x = tf.cast(inputs, compute_dtype) * mask
        else:
            x = tf.cast(inputs, compute_dtype)

        # Normalize over the channel axis (last axis), as in standard layer norm.
        mean, variance = tf.nn.moments(x, axes=-1, keepdims=True)
        normalized = (x - mean) / tf.sqrt(
            variance + tf.cast(self.epsilon, compute_dtype)
        )

        # Apply scale and center
        if self.scale:
            normalized = normalized * tf.cast(self.gamma, compute_dtype)
        if self.center:
            normalized = normalized + tf.cast(self.beta, compute_dtype)

        # Re-apply the mask to keep masked positions at zero
        if mask is not None:
            normalized = normalized * mask

        return tf.cast(normalized, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MaskedLayerNormalization, self).get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
            }
        )
        return config


class MaskedDYT(tf.keras.layers.Layer):
    """Masked Dynamic Tanh (DyT) layer.

    A normalization-free drop-in replacement that applies a learnable scaled
    hyperbolic tangent followed by a per-channel affine transform. Masked
    positions are kept at zero.
    """

    def __init__(self, alpha_init: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha_init = alpha_init
        self.supports_masking = True

    def build(self, input_shape):
        channel_dim = int(input_shape[-1])
        if channel_dim is None:
            raise ValueError("The last (channel) dimension must be defined.")

        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            trainable=True,
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channel_dim,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channel_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        x = tf.cast(inputs, tf.float32)
        out = tf.math.tanh(self.alpha * x)
        out = out * self.gamma + self.beta
        out = tf.cast(out, self.compute_dtype)

        if mask is not None:
            mask_f = tf.cast(mask, out.dtype)
            if mask_f.shape.rank is None or mask_f.shape.rank < out.shape.rank:
                mask_f = tf.expand_dims(mask_f, axis=-1)
            out = out * mask_f

        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"alpha_init": self.alpha_init})
        return config


class MaskedGlobalAvgPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # Assuming inputs and mask have the same shape (batch_size, height, width, channels)
        if mask is not None:
            mask = tf.cast(
                mask, inputs.dtype
            )  # Convert mask to the same dtype as inputs
            mask = tf.expand_dims(mask, axis=-1)
            inputs = inputs * mask  # Zero out masked positions

            # Sum over the spatial dimensions (batch, frames, seq_length, projections)
            masked_sum = tf.reduce_sum(inputs, axis=[1, 2])
            mask_sum = tf.reduce_sum(mask, axis=[1, 2])

            # Guard kept for clarity; divide_no_nan also handles all-zero masks.
            mask_sum = tf.maximum(mask_sum, tf.keras.backend.epsilon())

            # Compute the masked average
            return tf.math.divide_no_nan(masked_sum, mask_sum)
        else:
            # Default to regular global average pooling if no mask is provided
            return tf.reduce_mean(inputs, axis=[1, 2])

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[-1],
        )  # Output shape is (batch_size, num_channels)

    def compute_mask(self, inputs, mask=None):
        # Output is (B, C); there is no sequence dimension left to mask.
        return None

    def get_config(self):
        return super().get_config()


class GatedFrameGlobalMaxPooling(tf.keras.layers.Layer):
    """
    Frame-aware global max pooling.
    Input:  (B, F, L, D)
    Output: (B, D)  [and optionally (B, F) if return_gate=True]
    """

    def __init__(
        self,
        return_gate: bool = False,
        kernel_initializer="orthogonal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.return_gate = return_gate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        # child layer will be created in build()
        self.score_dense = None

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(f"Expected rank-4 input (B,F,L,D), got {input_shape}")
        d = int(input_shape[-1])
        self.score_dense = tf.keras.layers.Dense(
            units=1,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name=f"{self.name}_gate",
        )
        self.score_dense.build((input_shape[0], input_shape[1], d))
        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        # (B,F,L,D) -> max over L
        per_frame = tf.reduce_max(x, axis=2)  # (B,F,D)

        logits = self.score_dense(per_frame)  # (B,F,1)

        gates = tf.sigmoid(logits)
        gates = gates / (
            tf.reduce_sum(gates, axis=1, keepdims=True) + tf.keras.backend.epsilon()
        )

        pooled = tf.reduce_sum(per_frame * gates, axis=1)  # (B,D)

        if self.return_gate:
            return pooled, tf.squeeze(gates, axis=-1)  # (B,D), (B,F)
        return pooled

    def compute_output_shape(self, input_shape):
        b, f, seq_len, d = input_shape
        if self.return_gate:
            return (b, d), (b, f)
        return (b, d)

    # --- New Keras 3 APIs ---

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "return_gate": self.return_gate,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

    def get_build_config(self):
        # record anything needed to rebuild variables
        return {"score_dense": self.score_dense.get_config()}

    def build_from_config(self, config):
        # re-create child layer from saved config
        self.score_dense = tf.keras.layers.Dense.from_config(config["score_dense"])


# class MaskedBatchNorm(tf.keras.layers.Layer):
#     """
#     Masked Batch Normalization that supports arbitrary input rank and optional return
#     of normalized mean difference vectors. Masked positions are excluded from statistics.
#     """

#     def __init__(self, epsilon=1e-5, momentum=0.9, return_nmd=False,dtype=None, **kwargs):
#         super().__init__(**kwargs)
#         self.epsilon = epsilon
#         self.momentum = momentum
#         self.supports_masking = True
#         self.return_nmd = return_nmd

#     def build(self, input_shape):
#         channel_dim = input_shape[-1]
#         if channel_dim is None:
#             raise ValueError("The last (channel) dimension must be defined.")
#         self.gamma = self.add_weight(
#             shape=(channel_dim,), initializer="ones", trainable=True, name="gamma"
#         )
#         self.beta = self.add_weight(
#             shape=(channel_dim,), initializer="zeros", trainable=True, name="beta"
#         )
#         self.moving_mean = self.add_weight(
#             shape=(channel_dim,), initializer="zeros", trainable=False, name="moving_mean"
#         )
#         self.moving_variance = self.add_weight(
#             shape=(channel_dim,), initializer="ones", trainable=False, name="moving_variance"
#         )
#         super().build(input_shape)

#     def _vec_broadcast_shape(self, ndims_int, vec):
#         """Return [1, 1, ..., C] to reshape a (C,) vector for broadcasting."""
#         c = tf.shape(vec)[0]
#         ones = tf.ones([ndims_int - 1], dtype=tf.int32)  # [1]*(ndims-1) as a Tensor
#         return tf.concat([ones, [c]], axis=0)

#     def call(self, inputs, mask=None, training=False):
#         # Use STATIC rank for axes so Keras can infer shapes
#         ndims = inputs.shape.rank
#         if ndims is None:
#             # Fallback: require known rank for this layer
#             raise ValueError("Input rank must be statically known for MaskedBatchNorm.")

#         # Axes as Python lists (not Tensors!)
#         reduce_axes = list(range(0, max(ndims - 1, 0)))   # batch stats: all except channel
#         example_axes = list(range(1, max(ndims - 1, 1)))  # per-example: skip batch & channel

#         # Prepare mask (once) and masked inputs (once)
#         use_mask = mask is not None
#         if use_mask:
#             mask = tf.cast(mask, inputs.dtype)
#             if mask.shape.rank is None or mask.shape.rank < ndims:
#                 mask = tf.expand_dims(mask, axis=-1)  # broadcast over channels
#             masked_inputs = inputs * mask

#             valid_elements = tf.reduce_sum(mask, axis=reduce_axes) + self.epsilon
#             mean_batch = tf.reduce_sum(masked_inputs, axis=reduce_axes) / valid_elements

#             mean_broadcast_batch = tf.reshape(
#                 mean_batch, self._vec_broadcast_shape(ndims, mean_batch)
#             )
#             variance_batch = (
#                 tf.reduce_sum(mask * tf.square(inputs - mean_broadcast_batch), axis=reduce_axes)
#                 / valid_elements
#             )
#         else:
#             # axes are Python lists → safe for Keras shape inference
#             mean_batch, variance_batch = tf.nn.moments(inputs, axes=reduce_axes)

#         # Pick stats (update EMA during training)
#         if training:
#             self.moving_mean.assign(
#                 self.momentum * self.moving_mean + (1.0 - self.momentum) * mean_batch
#             )
#             self.moving_variance.assign(
#                 self.momentum * self.moving_variance + (1.0 - self.momentum) * variance_batch
#             )
#             mean_to_use = mean_batch
#             var_to_use = variance_batch
#         else:
#             mean_to_use = self.moving_mean
#             var_to_use = self.moving_variance

#         # Normalize (build broadcast shapes once)
#         mean_broadcast = tf.reshape(mean_to_use, self._vec_broadcast_shape(ndims, mean_to_use))
#         var_broadcast  = tf.reshape(var_to_use,  self._vec_broadcast_shape(ndims, var_to_use))
#         inv_std = tf.math.rsqrt(var_broadcast + self.epsilon)

#         normalized = (inputs - mean_broadcast) * inv_std
#         output = self.gamma * normalized + self.beta

#         if not self.return_nmd:
#             return output

#         # NMD: per-example channel mean (mask-aware) minus reference mean
#         if use_mask:
#             per_ex_sum   = tf.reduce_sum(masked_inputs, axis=example_axes)             # (B, C)
#             per_ex_count = tf.reduce_sum(mask,         axis=example_axes) + self.epsilon  # (B, 1)
#             mean_channel = per_ex_sum / per_ex_count                                   # (B, C)
#         else:
#             # If ndims==2, example_axes == [], reduce_mean returns inputs (OK)
#             mean_channel = tf.reduce_mean(inputs, axis=example_axes)                   # (B, C)

#         nmd = mean_channel - mean_to_use  # (B, C) - (C,) via broadcasting
#         return output, nmd

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "epsilon": self.epsilon,
#             "momentum": self.momentum,
#             "return_nmd": self.return_nmd,
#         })
#         return config


class MaskedBatchNorm(tf.keras.layers.Layer):
    """
    Masked Batch Normalization that supports arbitrary input rank and optional return
    of normalized mean difference vectors. Masked positions are excluded from statistics.
    """

    def __init__(
        self,
        epsilon=1e-5,
        momentum=0.9,
        return_nmd=False,
        use_masking=True,
        dtype=None,
        **kwargs,
    ):
        # Let Keras / mixed_precision policy control dtype if provided
        if dtype is not None:
            kwargs["dtype"] = dtype
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.momentum = momentum
        self.use_masking = use_masking
        self.supports_masking = use_masking
        self.return_nmd = return_nmd

    def build(self, input_shape):
        channel_dim = input_shape[-1]
        if channel_dim is None:
            raise ValueError("The last (channel) dimension must be defined.")

        # Use variable_dtype so these stay float32 under mixed_float16
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channel_dim,),
            initializer="ones",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channel_dim,),
            initializer="zeros",
            trainable=True,
            dtype=self.variable_dtype,
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(channel_dim,),
            initializer="zeros",
            trainable=False,
            dtype=self.variable_dtype,
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(channel_dim,),
            initializer="ones",
            trainable=False,
            dtype=self.variable_dtype,
        )
        super().build(input_shape)

    def _vec_broadcast_shape(self, ndims_int, vec):
        """Return [1, 1, ..., C] to reshape a (C,) vector for broadcasting."""
        c = tf.shape(vec)[0]
        ones = tf.ones([ndims_int - 1], dtype=tf.int32)
        return tf.concat([ones, [c]], axis=0)

    def call(self, inputs, mask=None, training=None):
        # Force stats math into float32
        x = tf.cast(inputs, tf.float32)
        ndims = x.shape.rank
        if ndims is None:
            raise ValueError("Input rank must be statically known for MaskedBatchNorm.")

        reduce_axes = list(
            range(0, max(ndims - 1, 0))
        )  # batch stats: all except channel
        example_axes = list(
            range(1, max(ndims - 1, 1))
        )  # per-example: skip batch & channel

        use_mask = self.use_masking and mask is not None
        if use_mask:
            mask_f = tf.cast(mask, tf.float32)
            if mask_f.shape.rank is None or mask_f.shape.rank < ndims:
                mask_f = tf.expand_dims(mask_f, axis=-1)  # broadcast over channels

            masked_inputs = x * mask_f

            valid_elements = tf.reduce_sum(mask_f, axis=reduce_axes) + self.epsilon
            mean_batch = tf.reduce_sum(masked_inputs, axis=reduce_axes) / valid_elements

            mean_broadcast_batch = tf.reshape(
                mean_batch, self._vec_broadcast_shape(ndims, mean_batch)
            )
            variance_batch = (
                tf.reduce_sum(
                    mask_f * tf.square(x - mean_broadcast_batch), axis=reduce_axes
                )
                / valid_elements
            )
        else:
            mean_batch, variance_batch = tf.nn.moments(x, axes=reduce_axes)

        # Ensure stats are float32
        mean_batch = tf.cast(mean_batch, tf.float32)
        variance_batch = tf.cast(variance_batch, tf.float32)

        # --- update moving stats in float32, then cast to var dtype on assign ---
        mm = tf.cast(self.moving_mean, tf.float32)
        mv = tf.cast(self.moving_variance, tf.float32)

        if training:
            new_mm = self.momentum * mm + (1.0 - self.momentum) * mean_batch
            new_mv = self.momentum * mv + (1.0 - self.momentum) * variance_batch

            self.moving_mean.assign(tf.cast(new_mm, self.moving_mean.dtype))
            self.moving_variance.assign(tf.cast(new_mv, self.moving_variance.dtype))

            mean_to_use = mean_batch
            var_to_use = variance_batch
        else:
            mean_to_use = mm
            var_to_use = mv

        mean_broadcast = tf.reshape(
            mean_to_use, self._vec_broadcast_shape(ndims, mean_to_use)
        )
        var_broadcast = tf.reshape(
            var_to_use, self._vec_broadcast_shape(ndims, var_to_use)
        )
        inv_std = tf.math.rsqrt(var_broadcast + tf.cast(self.epsilon, tf.float32))

        # Normalize + affine in float32
        gamma = tf.cast(self.gamma, tf.float32)
        beta = tf.cast(self.beta, tf.float32)

        normalized = (x - mean_broadcast) * inv_std
        output_f32 = gamma * normalized + beta

        # Cast back to the layer's compute dtype (float16 under mixed precision)
        output = tf.cast(output_f32, self.compute_dtype)

        if not self.return_nmd:
            return output

        # NMD in float32, then cast
        if use_mask:
            per_ex_sum = tf.reduce_sum(masked_inputs, axis=example_axes)  # (B, C)
            per_ex_count = tf.reduce_sum(mask_f, axis=example_axes) + self.epsilon
            mean_channel = per_ex_sum / per_ex_count  # (B, C)
        else:
            mean_channel = tf.reduce_mean(x, axis=example_axes)  # (B, C)

        nmd_f32 = mean_channel - mean_to_use  # (B, C) - (C,)
        nmd = tf.cast(nmd_f32, self.compute_dtype)

        return output, nmd

    def compute_output_shape(self, input_shape):
        if self.return_nmd:
            batch = input_shape[0]
            channels = input_shape[-1]
            return (input_shape, (batch, channels))
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "epsilon": self.epsilon,
                "momentum": self.momentum,
                "return_nmd": self.return_nmd,
                "use_masking": self.use_masking,
            }
        )
        return config


# class MaskedConv1D(tf.keras.layers.Layer):
#     """
#     Masked 1D convolution that accepts an optional mask tensor and sets
#     mask positions to zero before applying convolution. Also, propagates the mask
#     to the next layer. Accepts inputs like (batch, frames, length, channels) where
#     batch and length dimention can be unknown.
#     """

#     def __init__(
#         self,
#         filters,
#         kernel_size,
#         strides=1,
#         axis=-1,
#         padding="valid",
#         dilation_rate=1,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         dtype=None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.axis = axis
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding.upper()
#         self.dilation_rate = dilation_rate
#         self.activation = tf.keras.activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
#         self.bias_initializer = tf.keras.initializers.get(bias_initializer)
#         self.kernel_regularizer = kernel_regularizer

#     def build(self, input_shape):
#         channel_axis = self.axis
#         input_dim = input_shape[channel_axis]
#         kernel_shape = (self.kernel_size, input_dim, self.filters)

#         self.kernel = self.add_weight(
#             shape=kernel_shape,
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             name="kernel",
#             trainable=True,
#         )
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 shape=(self.filters,),
#                 initializer=self.bias_initializer,
#                 name="bias",
#                 trainable=True,
#             )
#         else:
#             self.bias = None
#         super().build(input_shape)


#     def call(self, inputs, mask=None):
#         input_shape = tf.shape(inputs)
#         output_shape = self.compute_output_shape(input_shape)

#         output_mask = None
#         if mask is not None:
#             mask = tf.cast(mask, dtype=inputs.dtype)
#             inputs = inputs * tf.expand_dims(mask, axis=-1)

#             # compute output mask here (part of the graph, fine during training/inference)
#             reshaped_mask = tf.reshape(mask, (-1, input_shape[2]))
#             mask = tf.expand_dims(reshaped_mask, axis=-1)
#             mask_kernel = tf.ones((self.kernel_size, 1, 1), dtype=mask.dtype)
#             output_mask = tf.nn.conv1d(
#                 input=mask,
#                 filters=mask_kernel,
#                 stride=self.strides,
#                 padding=self.padding,
#                 dilations=self.dilation_rate,
#                 data_format="NWC",
#             )
#             output_mask = tf.equal(output_mask, self.kernel_size)
#             output_mask = tf.squeeze(output_mask, axis=-1)
#             output_mask = tf.reshape(output_mask, shape=output_shape[:-1])

#         # conv1d on actual data
#         reshaped_inputs = tf.reshape(inputs, (-1, input_shape[2], input_shape[3]))
#         outputs = tf.nn.conv1d(
#             input=reshaped_inputs,
#             filters=self.kernel,
#             stride=self.strides,
#             padding=self.padding,
#             dilations=self.dilation_rate,
#             data_format="NWC",
#         )
#         if self.use_bias:
#             outputs = tf.nn.bias_add(outputs, self.bias, data_format="NWC")
#         if self.activation is not None:
#             outputs = self.activation(outputs)

#         outputs = tf.reshape(outputs, output_shape)

#         # stash mask so compute_mask() can forward it
#         self._output_mask = output_mask
#         return outputs

#     def compute_mask(self, inputs, mask=None):
#         # never recompute, just return the cached mask
#         return getattr(self, "_output_mask", None)

#     def get_config(self):
#         config = super(MaskedConv1D, self).get_config()
#         config.update(
#             {
#                 "filters": self.filters,
#                 "kernel_size": self.kernel_size,
#                 "strides": self.strides,
#                 "padding": self.padding.lower(),
#                 "dilation_rate": self.dilation_rate,
#                 "activation": tf.keras.activations.serialize(self.activation),
#                 "use_bias": self.use_bias,
#                 "kernel_initializer": tf.keras.initializers.serialize(
#                     self.kernel_initializer
#                 ),
#                 "bias_initializer": tf.keras.initializers.serialize(
#                     self.bias_initializer
#                 ),
#             }
#         )
#         return config

#     def compute_output_shape(self, input_shape):
#         """
#         compute the output shape only if the length dimention is not None.
#         """
#         length = input_shape[2]
#         if length is not None:
#             if self.padding == "SAME":
#                 out_length = (length + self.strides - 1) // self.strides
#             elif self.padding == "VALID":
#                 out_length = (
#                     length - self.dilation_rate * (self.kernel_size - 1) - 1
#                 ) // self.strides + 1
#             else:
#                 raise ValueError("Invalid padding type.")
#             out_length = out_length
#             return (input_shape[0], input_shape[1], out_length, self.filters)
#         else:
#             return (input_shape[0], input_shape[1], input_shape[2], self.filters)


class MaskedConv1D(tf.keras.layers.Layer):
    """
    Masked 1D convolution that accepts an optional mask tensor and sets
    mask positions to zero before applying convolution. Also propagates the mask
    to the next layer. Accepts inputs like (batch, frames, length, channels) where
    batch and length dimension can be unknown.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        axis=-1,
        padding="valid",
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        use_masking=True,
        dtype=None,  # keep for compatibility, but usually leave None under MP
        **kwargs,
    ):
        if dtype is not None:
            kwargs["dtype"] = dtype  # let Keras handle policy + dtype
        super().__init__(**kwargs)

        self.axis = axis
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_masking = use_masking

        self.supports_masking = use_masking

    def build(self, input_shape):
        channel_axis = self.axis
        input_dim = int(input_shape[channel_axis])
        kernel_shape = (self.kernel_size, input_dim, self.filters)

        # IMPORTANT: use variable_dtype so vars stay float32 under mixed_float16
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.variable_dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.variable_dtype,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs, mask=None):
        # Inputs arrive in compute_dtype under policy (e.g. float16)
        # Be explicit:
        inputs = tf.cast(inputs, self.compute_dtype)

        input_shape = tf.shape(inputs)
        output_shape = self.compute_output_shape(input_shape)

        output_mask = None
        if self.use_masking and mask is not None:
            # Broadcast mask, but do mask math in float32 for stability
            mask = tf.cast(mask, tf.float32)
            # apply mask on inputs (cast back to compute_dtype)
            inputs = inputs * tf.cast(tf.expand_dims(mask, axis=-1), self.compute_dtype)

            # compute output mask (pure mask math in float32)
            reshaped_mask = tf.reshape(mask, (-1, input_shape[2]))
            mask_1d = tf.expand_dims(reshaped_mask, axis=-1)
            mask_kernel = tf.ones((self.kernel_size, 1, 1), dtype=tf.float32)

            mask_conv = tf.nn.conv1d(
                input=mask_1d,
                filters=mask_kernel,
                stride=self.strides,
                padding=self.padding,
                dilations=self.dilation_rate,
                data_format="NWC",
            )
            output_mask = tf.equal(mask_conv, float(self.kernel_size))
            output_mask = tf.squeeze(output_mask, axis=-1)
            output_mask = tf.reshape(output_mask, shape=output_shape[:-1])

        # conv1d on actual data
        reshaped_inputs = tf.reshape(inputs, (-1, input_shape[2], input_shape[3]))

        # Cast kernel to compute dtype when mixing with activations
        kernel = tf.cast(self.kernel, self.compute_dtype)

        outputs = tf.nn.conv1d(
            input=reshaped_inputs,
            filters=kernel,
            stride=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            data_format="NWC",
        )
        if self.use_bias:
            bias = tf.cast(self.bias, self.compute_dtype)
            outputs = tf.nn.bias_add(outputs, bias, data_format="NWC")
        if self.activation is not None:
            outputs = self.activation(outputs)

        outputs = tf.reshape(outputs, output_shape)

        # stash mask so compute_mask() can forward it
        self._output_mask = output_mask
        return outputs

    def compute_mask(self, inputs, mask=None):
        # never recompute, just return the cached mask
        return getattr(self, "_output_mask", None)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding.lower(),
                "dilation_rate": self.dilation_rate,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "axis": self.axis,
                "use_masking": self.use_masking,
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                )
                if self.kernel_regularizer is not None
                else None,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        """
        compute the output shape only if the length dimension is not None.
        This version is written to work with either TensorShape or tf.Tensor.
        """
        length = input_shape[2]
        if length is not None:
            if self.padding == "SAME":
                out_length = (length + self.strides - 1) // self.strides
            elif self.padding == "VALID":
                out_length = (
                    length - self.dilation_rate * (self.kernel_size - 1) - 1
                ) // self.strides + 1
            else:
                raise ValueError("Invalid padding type.")
            return (input_shape[0], input_shape[1], out_length, self.filters)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.filters)


class MaskedBiLSTM(tf.keras.layers.Layer):
    """Bidirectional LSTM for 4-D fragment inputs with optional masking.

    Input shape: ``(batch, frames, length, channels)``
    Output shape: ``(batch, frames, length, 2 * units)`` when
    ``return_sequences=True`` (the default).

    The layer reshapes the input to ``(batch * frames, length, channels)``,
    applies a standard Keras ``Bidirectional(LSTM(...))``, then reshapes the
    result back to the 4-D fragment layout.

    If ``ignore_mask=False`` (the default) any supplied mask is flattened and
    passed to the LSTM so padded positions are ignored. This requires
    ``use_cudnn=False`` because cuDNN only supports right-padded masks. If you
    want to use cuDNN for speed, set ``use_cudnn=True`` and ``ignore_mask=True``;
    this tells the layer not to pass a mask to the LSTM and is safe when all
    input positions are valid (e.g. fixed-length crops).
    """

    def __init__(
        self,
        units: int,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        return_sequences: bool = True,
        use_cudnn: bool = False,
        ignore_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.use_cudnn = use_cudnn
        self.ignore_mask = ignore_mask

        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                use_cudnn=use_cudnn,
            ),
            merge_mode="concat",
        )

    def call(self, inputs, mask=None):
        b = tf.shape(inputs)[0]
        f = tf.shape(inputs)[1]
        t = tf.shape(inputs)[2]
        c = tf.shape(inputs)[3]

        # Collapse frames into the batch dimension so the LSTM sees 3-D data.
        x = tf.reshape(inputs, (b * f, t, c))

        lstm_mask = None
        if mask is not None and not self.ignore_mask:
            mask = tf.cast(mask, tf.bool)
            lstm_mask = tf.reshape(mask, (b * f, t))

        x = self.bilstm(x, mask=lstm_mask)

        # Restore the original (batch, frames, ...) layout.
        out_c = tf.shape(x)[-1]
        if self.return_sequences:
            return tf.reshape(x, (b, f, t, out_c))
        return tf.reshape(x, (b, f, out_c))

    def compute_mask(self, inputs, mask=None):
        # If the LSTM ignores the mask, downstream layers should too.
        if self.ignore_mask:
            return None
        # Mask shape is unchanged by the LSTM.
        return mask

    def compute_output_shape(self, input_shape):
        batch, frames, length, _ = input_shape
        if self.return_sequences:
            return (batch, frames, length, self.units * 2)
        return (batch, frames, self.units * 2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "return_sequences": self.return_sequences,
                "use_cudnn": self.use_cudnn,
                "ignore_mask": self.ignore_mask,
            }
        )
        return config


class MultiScaleConv1D(tf.keras.layers.Layer):
    """Parallel masked 1D convolutions at multiple scales.

    Input shape: (batch, frames, length, channels)
    Output shape: (batch, frames, length, total_filters) for merge="concat", or
                  (batch, frames, length, branch_filters) for merge="add".

    Each branch is configured by a dict passed to `MaskedConv1D`. Branch
    sequence lengths must align, which is enforced by using ``padding="same"``
    and ``strides=1`` by default.
    """

    def __init__(
        self,
        branches: list[dict],
        merge: str = "concat",
        kernel_initializer: str | tf.keras.initializers.Initializer = "glorot_uniform",
        kernel_regularizer: str | tf.keras.regularizers.Regularizer | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(branches, list) or len(branches) == 0:
            raise ValueError("branches must be a non-empty list of dicts")
        if not all(isinstance(b, dict) for b in branches):
            raise ValueError("branches must be a non-empty list of dicts")
        if merge not in {"concat", "add"}:
            raise ValueError(f"merge must be 'concat' or 'add', got {merge!r}")

        self.branches = list(branches)
        self.merge = merge.lower()
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.supports_masking = True
        self._convs: list[MaskedConv1D] = []

    def _resolve(self, value, kind: str):
        """Convert string regularizer/initializer names to objects."""
        if kind == "regularizer" and isinstance(value, str):
            return tf.keras.regularizers.get(value)
        if kind == "initializer" and isinstance(value, str):
            return tf.keras.initializers.get(value)
        return value

    def _serialize_value(self, value):
        """Serialize Keras objects that may appear in branch configs."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, tf.keras.initializers.Initializer):
            return tf.keras.initializers.serialize(value)
        if isinstance(value, tf.keras.regularizers.Regularizer):
            return tf.keras.regularizers.serialize(value)
        if callable(value):
            # covers activation functions and custom callables
            return tf.keras.utils.serialize_keras_object(value)
        return value

    def build(self, input_shape):
        self._convs = []

        if self.merge == "add":
            filters = [b.get("filters") for b in self.branches]
            if len(set(filters)) != 1:
                raise ValueError(
                    "All branches must have the same filters when merge='add'"
                )

        for i, cfg in enumerate(self.branches):
            branch_cfg = dict(cfg)
            branch_cfg.setdefault("padding", "same")
            branch_cfg.setdefault("strides", 1)
            branch_cfg.setdefault("name", f"{self.name}_branch_{i}")
            branch_cfg.setdefault(
                "kernel_initializer",
                self._resolve(self.kernel_initializer, "initializer"),
            )
            branch_cfg.setdefault(
                "kernel_regularizer",
                self._resolve(self.kernel_regularizer, "regularizer"),
            )
            branch_cfg.setdefault("use_bias", self.use_bias)

            if branch_cfg.get("padding", "same").lower() != "same":
                raise ValueError(
                    f"Branch {i}: padding must be 'same' to keep sequence length "
                    f"aligned across branches, got {branch_cfg['padding']!r}"
                )
            if branch_cfg.get("strides", 1) != 1:
                raise ValueError(
                    f"Branch {i}: strides must be 1 to keep sequence length "
                    f"aligned across branches, got {branch_cfg['strides']!r}"
                )

            self._convs.append(MaskedConv1D(**branch_cfg))
        super().build(input_shape)

    def call(self, inputs, mask=None):
        outputs = []
        masks = []
        for conv in self._convs:
            out = conv(inputs, mask=mask)
            outputs.append(out)
            masks.append(conv.compute_mask(inputs, mask=mask))

        if self.merge == "concat":
            x = tf.concat(outputs, axis=-1)
        else:
            x = tf.add_n(outputs)

        combined_mask = None
        if masks and masks[0] is not None:
            combined_mask = masks[0]
            for m in masks[1:]:
                combined_mask = tf.logical_and(combined_mask, m)
        self._output_mask = combined_mask
        return x

    def compute_mask(self, inputs, mask=None):
        return getattr(self, "_output_mask", mask)

    def compute_output_shape(self, input_shape):
        if self.merge == "concat":
            total = sum(b.get("filters", 0) for b in self.branches)
        else:
            total = self.branches[0].get("filters", 0)
        return (input_shape[0], input_shape[1], input_shape[2], total)

    def get_config(self):
        config = super().get_config()

        def _serialize(value, kind):
            if value is None:
                return None
            if isinstance(value, str):
                return value
            if kind == "initializer":
                return tf.keras.initializers.serialize(value)
            if kind == "regularizer":
                return tf.keras.regularizers.serialize(value)
            return value

        serialized_branches = []
        for branch in self.branches:
            serialized_branches.append(
                {k: self._serialize_value(v) for k, v in branch.items()}
            )

        config.update(
            {
                "branches": serialized_branches,
                "merge": self.merge,
                "kernel_initializer": _serialize(
                    self._resolve(self.kernel_initializer, "initializer"), "initializer"
                ),
                "kernel_regularizer": _serialize(
                    self._resolve(self.kernel_regularizer, "regularizer"), "regularizer"
                ),
                "use_bias": self.use_bias,
            }
        )
        return config


class OODSignalLayer(tf.keras.layers.Layer):
    """Compute scalar out-of-distribution signals from classifier logits and an optional NMD vector.

    Inputs can be a dict ``{"logits": ..., "nmd": ...}`` or a sequence ``[logits, nmd]``.
    Output shape is ``(batch, num_signals)``.
    """

    _SUPPORTED_SIGNALS = {
        "max_prob",
        "entropy",
        "energy",
        "margin",
        "nmd_norm",
    }

    def __init__(
        self,
        signals: list[str] | None = None,
        epsilon: float = 1e-10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if signals is None:
            signals = ["max_prob"]
        self.signals = list(signals)
        self.epsilon = epsilon

        unsupported = set(self.signals) - self._SUPPORTED_SIGNALS
        if unsupported:
            raise ValueError(
                f"Unsupported signal(s): {sorted(unsupported)}. "
                f"Supported: {sorted(self._SUPPORTED_SIGNALS)}"
            )

    def call(self, inputs, training=None):
        if isinstance(inputs, dict):
            logits = inputs["logits"]
            nmd = inputs.get("nmd")
        elif isinstance(inputs, (list, tuple)):
            logits, nmd = inputs
        else:
            logits = inputs
            nmd = None

        logits = tf.cast(logits, tf.float32)
        probs = tf.nn.softmax(logits, axis=-1)

        computed = []
        for signal in self.signals:
            if signal == "max_prob":
                computed.append(tf.reduce_max(probs, axis=-1, keepdims=True))
            elif signal == "entropy":
                safe_probs = tf.maximum(probs, self.epsilon)
                entropy = -tf.reduce_sum(
                    safe_probs * tf.math.log(safe_probs), axis=-1, keepdims=True
                )
                computed.append(entropy)
            elif signal == "energy":
                computed.append(tf.reduce_logsumexp(logits, axis=-1, keepdims=True))
            elif signal == "margin":
                top_2 = tf.nn.top_k(probs, k=2).values  # (batch, 2)
                margin = top_2[..., 0:1] - top_2[..., 1:2]
                computed.append(margin)
            elif signal == "nmd_norm":
                if nmd is None:
                    raise ValueError("signal 'nmd_norm' requires an NMD vector input")
                nmd = tf.cast(nmd, tf.float32)
                computed.append(tf.norm(nmd, ord="euclidean", axis=-1, keepdims=True))

        return tf.concat(computed, axis=-1)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            logits_shape = input_shape["logits"]
        elif isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            logits_shape = input_shape[0]
        else:
            logits_shape = input_shape
        return (logits_shape[0], len(self.signals))

    def get_config(self):
        config = super().get_config()
        config.update({"signals": self.signals, "epsilon": self.epsilon})
        return config


# class ResidualBlock(tf.keras.layers.Layer):
#     """The Residual block of ResNet models."""

#     def __init__(
#         self,
#         use_1x1conv=False,
#         block_number=1,
#         activation="gelu",
#         return_nmd=False,
#         **kwargs,
#     ):

#         super().__init__()
#         self.supports_masking = True
#         self.filters = kwargs.get('filters')
#         self.padding = kwargs.get('padding', 'same').upper()
#         self.strides = kwargs.get('strides', 1)
#         self.block_number = block_number
#         self.return_nmd = return_nmd
#         self.conv1 = MaskedConv1D(
#             padding=self.padding,
#             name=f"masked_conv1d_blk{self.block_number}_1",
#             **{k:v for k,v in kwargs.items() if k not in ["name", "padding"]}
#         )
#         self.conv2 = MaskedConv1D(
#             padding=self.padding,
#             name=f"masked_conv1d_blk{self.block_number}_2",
#             **{k:v for k,v in kwargs.items() if k not in ["name", "padding", "strides"]}
#         )
#         self.conv3 = None
#         if use_1x1conv or kwargs.get('strides') > 1:
#             self.conv3 = MaskedConv1D(
#                  padding=self.padding,
#                  kernel_size=1,
#                  name=f"masked_conv1d_blk{self.block_number}_bypass",
#                  **{k:v for k,v in kwargs.items() if k not in ["kernel_size", "name", "padding"]}
#             )
#             self.bn3 = MaskedBatchNorm(name=f"masked_batchnorm_blk{self.block_number}_bypass",)
#         self.bn1 = MaskedBatchNorm(name=f"masked_batchnorm_blk{self.block_number}_1",)
#         self.bn2 = MaskedBatchNorm(name=f"masked_batchnorm_blk{self.block_number}_2", return_nmd=return_nmd)
#         self.add = MaskedAdd(name=f"resblock_add_blk{self.block_number}")
#         self.activation = tf.keras.layers.Activation(activation, name=f"resblock_activation_blk{self.block_number}")

#     def call(self, inputs, mask=None):
#         x = self.conv1(inputs, mask=mask)
#         x = self.bn1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         if self.return_nmd:
#             x, x_nmd = self.bn2(x)
#         else:
#             x = self.bn2(x)

#         if self.conv3 is not None:
#             x = self.add([x, self.bn3(self.conv3(inputs))])
#         else:
#             x = self.add([x, inputs])
#         if self.return_nmd:
#             return self.activation(x), x_nmd
#         return self.activation(x)

#     def compute_output_shape(self, input_shape):
#         """
#         compute the output shape only if the length dimention is not None.
#         """
#         length = input_shape[2]
#         if length is not None:
#             if self.padding == "SAME":
#                 out_length = (length + self.strides - 1) // self.strides
#             elif self.padding == "VALID":
#                 out_length = (
#                     length - self.dilation_rate * (self.kernel_size - 1) - 1
#                 ) // self.strides + 1
#             else:
#                 raise ValueError("Invalid padding type.")
#             out_length = out_length
#             if self.return_nmd:
#                 return (input_shape[0], input_shape[1], out_length, self.filters), (input_shape[0], self.filters)
#             return (input_shape[0], input_shape[1], out_length, self.filters)
#         else:
#             if self.return_nmd:
#                 return (input_shape[0], input_shape[1], input_shape[2], self.filters), (input_shape[0], self.filters)
#             return (input_shape[0], input_shape[1], input_shape[2], self.filters)


#     # Todo: implement output mask computation -> return the mask computed by the last layer?


class ResidualBlock(tf.keras.layers.Layer):
    """The Residual block of ResNet models."""

    def __init__(
        self,
        use_1x1conv=False,
        block_number=1,
        activation="gelu",
        return_nmd=False,
        **kwargs,
    ):
        # --- 1. Pull out args meant for the convs, so they don't go to super().__init__ ---
        self.filters = kwargs.pop("filters", None)
        self.kernel_size = kwargs.pop("kernel_size", 3)
        self.strides = kwargs.pop("strides", 1)
        self.padding = kwargs.pop("padding", "same").upper()
        self.dilation_rate = kwargs.pop("dilation_rate", 1)
        self.use_bias = kwargs.pop("use_bias", True)
        self.kernel_regularizer = kwargs.pop("kernel_regularizer", None)
        self.kernel_initializer = kwargs.pop("kernel_initializer", "glorot_uniform")
        self.bias_initializer = kwargs.pop("bias_initializer", "zeros")
        self.norm_type = kwargs.pop("norm_type", "masked_batchnorm").lower()
        self.alpha_init = kwargs.pop("alpha_init", 0.5)
        self.use_masking = kwargs.pop("use_masking", True)

        if return_nmd and self.norm_type != "masked_batchnorm":
            raise ValueError(
                "return_nmd=True is only supported with norm_type='masked_batchnorm'. "
                f"Got norm_type={self.norm_type!r}."
            )

        # now kwargs only contains things Layer.__init__ understands (name, dtype, trainable, etc.)
        super().__init__(**kwargs)

        self.supports_masking = self.use_masking
        self.block_number = block_number
        self.return_nmd = return_nmd

        # --- 2. Build the internal conv/bn/add layers using the extracted values ---

        conv_common = dict(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_masking=self.use_masking,
        )

        def _make_norm(name, return_nmd=False):
            if self.norm_type == "masked_batchnorm":
                return MaskedBatchNorm(
                    name=name, return_nmd=return_nmd, use_masking=self.use_masking
                )
            if self.norm_type == "masked_layernorm":
                return MaskedLayerNormalization(name=name)
            if self.norm_type == "masked_dyt":
                return MaskedDYT(name=name, alpha_init=self.alpha_init)
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        # first conv
        self.conv1 = MaskedConv1D(
            name=f"masked_conv1d_blk{self.block_number}_1",
            **conv_common,
        )

        # second conv has stride = 1
        conv2_common = conv_common.copy()
        conv2_common["strides"] = 1
        self.conv2 = MaskedConv1D(
            name=f"masked_conv1d_blk{self.block_number}_2",
            **conv2_common,
        )

        # optional 1x1 conv for the shortcut
        self.conv3 = None
        self.bn3 = None
        if use_1x1conv or self.strides > 1:
            bypass_common = conv_common.copy()
            bypass_common["kernel_size"] = 1
            self.conv3 = MaskedConv1D(
                name=f"masked_conv1d_blk{self.block_number}_bypass",
                **bypass_common,
            )
            self.bn3 = _make_norm(
                name=f"{self.norm_type}_blk{self.block_number}_bypass",
            )

        self.bn1 = _make_norm(
            name=f"{self.norm_type}_blk{self.block_number}_1",
        )
        self.bn2 = _make_norm(
            name=f"{self.norm_type}_blk{self.block_number}_2",
            return_nmd=return_nmd,
        )

        if self.use_masking:
            self.add = MaskedAdd(name=f"resblock_add_blk{self.block_number}")
        else:
            self.add = tf.keras.layers.Add(name=f"resblock_add_blk{self.block_number}")
        self.activation_layer = tf.keras.layers.Activation(
            activation, name=f"resblock_activation_blk{self.block_number}"
        )

    def call(self, inputs, mask=None, training=None):
        x = self.conv1(inputs, mask=mask)
        if self.norm_type == "masked_batchnorm":
            x = self.bn1(x, training=training)
        else:
            x = self.bn1(x)
        x = self.activation_layer(x)

        x = self.conv2(x)
        if self.return_nmd:
            x, x_nmd = self.bn2(x, training=training)
        else:
            x = self.bn2(x)

        if self.conv3 is not None:
            shortcut = self.conv3(inputs, mask=mask)
            if self.norm_type == "masked_batchnorm":
                shortcut = self.bn3(shortcut, training=training)
            else:
                shortcut = self.bn3(shortcut)
        else:
            shortcut = inputs

        x = self.add([x, shortcut])
        x = self.activation_layer(x)

        if self.return_nmd:
            return x, x_nmd
        return x

    def compute_output_shape(self, input_shape):
        length = input_shape[2]
        if length is not None:
            if self.padding == "SAME":
                out_length = (length + self.strides - 1) // self.strides
            elif self.padding == "VALID":
                out_length = (
                    length - self.dilation_rate * (self.kernel_size - 1) - 1
                ) // self.strides + 1
            else:
                raise ValueError("Invalid padding type.")
            if self.return_nmd:
                return (
                    (input_shape[0], input_shape[1], out_length, self.filters),
                    (input_shape[0], self.filters),
                )
            return (input_shape[0], input_shape[1], out_length, self.filters)
        else:
            if self.return_nmd:
                return (
                    (input_shape[0], input_shape[1], input_shape[2], self.filters),
                    (input_shape[0], self.filters),
                )
            return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "use_1x1conv": hasattr(self, "conv3") and self.conv3 is not None,
                "block_number": self.block_number,
                "activation": tf.keras.activations.serialize(
                    self.activation_layer.activation
                ),
                "return_nmd": self.return_nmd,
                "norm_type": self.norm_type,
                "alpha_init": self.alpha_init,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding.lower(),
                "dilation_rate": self.dilation_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                )
                if self.kernel_regularizer is not None
                else None,
            }
        )
        return config


# To do: implement a method to set epsilon considering the data type of the tensors passed to the layers
# if not implemented correctly, this can lead to overflow/underflow issues.


class MetricModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.gradient_accumulation_steps = 1
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regularization_loss_tracker = tf.keras.metrics.Mean(name="reg-loss")
        self.gradient_tracker = tf.keras.metrics.Mean(name="gradient")

    def compile(self, optimizer, loss_fn, **kwargs):
        # Keras 3 optimizers support gradient accumulation natively. Hand the
        # setting off to the optimizer so accumulation happens inside the
        # replicated train_step; this avoids applying gradients from a callback
        # (e.g. at epoch end) where no replica context exists under
        # tf.distribute.Strategy such as MirroredStrategy.
        if self.gradient_accumulation_steps > 1:
            self._configure_optimizer_for_gradient_accumulation(optimizer)
        super(MetricModel, self).compile(optimizer=optimizer, **kwargs)
        self.loss_fn = loss_fn

    def _configure_optimizer_for_gradient_accumulation(self, optimizer):
        """Propagate ``gradient_accumulation_steps`` to the core optimizer.

        ``LossScaleOptimizer`` wraps the real optimizer; the accumulation
        setting must live on the inner optimizer because the wrapper delegates
        gradient application to it.
        """
        target = optimizer
        if isinstance(target, tf.keras.optimizers.LossScaleOptimizer):
            target = target.inner_optimizer
        if hasattr(target, "gradient_accumulation_steps"):
            target.gradient_accumulation_steps = self.gradient_accumulation_steps

    def _update_gradient_metric(
        self, total_norm: tf.Tensor, total_params: tf.Tensor
    ) -> None:
        """Update the gradient tracker with log-average gradient norm."""
        avg_grad_norm = total_norm / tf.maximum(total_params, 1.0)
        log_grad = tf.math.log(avg_grad_norm + 1e-12)
        self.gradient_tracker.update_state(log_grad)

    def train_step(self, data):
        if len(data) == 3:
            x, y, _ = data
        else:
            # sample _weighning has to be implemented
            _ = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            report_loss = self.loss_fn([y, y_pred])
            report_loss += sum(self.losses)
            loss = report_loss
            # If using mixed precision
            if hasattr(self.optimizer, "get_scaled_loss"):
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        # If using mixed precision
        if hasattr(self.optimizer, "get_unscaled_gradients"):
            grads = self.optimizer.get_unscaled_gradients(grads)

        # Let the optimizer apply (and optionally accumulate) the gradients.
        # Keras 3 optimizers support ``gradient_accumulation_steps`` natively;
        # this keeps all gradient-related work inside the replicated train_step
        # and avoids the need for a callback-time flush.
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Compute average gradient norm for monitoring.
        total_norm = 0.0
        total_params = 0.0
        for grad, weight in zip(grads, self.trainable_weights):
            if grad is not None:
                norm = tf.norm(grad)
                total_norm += norm
                total_params += tf.cast(tf.math.reduce_prod(weight.shape), tf.float32)
        self._update_gradient_metric(total_norm, total_params)

        # Update step and metrics
        self.step += 1
        if self.step % 100 == 0:
            # Optional: reset metrics every 100 steps
            self.loss_tracker.reset_state()
            self.gradient_tracker.reset_state()

        self.loss_tracker.update_state(report_loss)
        self.regularization_loss_tracker.update_state(sum(self.losses))

        return {
            "loss": self.loss_tracker.result(),
            "reg-loss": self.regularization_loss_tracker.result(),
            "grad": self.gradient_tracker.result(),
            "lr": self.optimizer.learning_rate,
        }

    def test_step(self, data):
        # No scaling needed in eval
        if len(data) == 3:
            x, y, _ = data
        else:
            x, y = data

        y_pred = self(x, training=False)
        base_loss = self.loss_fn([y, y_pred])
        reg_loss = tf.add_n(self.losses) if self.losses else 0.0
        total_loss = base_loss + reg_loss

        self.loss_tracker.update_state(total_loss)
        # (Usually people don’t track reg-loss in test, but you can if you want)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # These are reset automatically at the start of each epoch
        return [
            self.loss_tracker,
            self.regularization_loss_tracker,
            self.gradient_tracker,
        ]


class SinusoidalPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.supports_masking = True

    def call(self, inputs, start_index=0):
        # inputs: tensor with shape [..., seq_length, hidden_size]
        # Compute dynamic shapes
        input_shape = tf.shape(inputs)
        seq_length = input_shape[-2]
        hidden_size = input_shape[-1]

        # Positions [0, 1, ..., seq_length-1] offset by start_index
        positions = tf.cast(
            tf.range(seq_length) + start_index, dtype=self.compute_dtype
        )

        # Minimum frequency as inverse of max_wavelength
        min_freq = tf.cast(1.0 / self.max_wavelength, dtype=self.compute_dtype)

        # Compute timescales for each dimension: min_freq^(2i/hidden_size)
        dim_indices = tf.cast(tf.range(hidden_size), dtype=self.compute_dtype)
        # floor(dim_indices/2)*2 for pairing sin/cos
        even_dims = tf.floor(dim_indices / 2) * 2
        timescales = tf.pow(
            min_freq, even_dims / tf.cast(hidden_size, self.compute_dtype)
        )

        # Compute angles: outer product of positions and timescales
        angles = tf.expand_dims(positions, -1) * tf.expand_dims(timescales, 0)

        # Build masks: even dims use sine, odd use cosine
        sin_mask = tf.cast(tf.equal(dim_indices % 2, 0), self.compute_dtype)
        cos_mask = 1.0 - sin_mask

        # Compute positional encodings: sin for even, cos for odd dims
        pos_encoding = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        # pos_encoding shape [seq_length, hidden_size]

        # Broadcast to match input shape
        broadcast_shape = tf.concat(
            [input_shape[:-2], [seq_length, hidden_size]], axis=0
        )
        pos_encoding = tf.broadcast_to(pos_encoding, broadcast_shape)

        return pos_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"max_wavelength": self.max_wavelength})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        feed_forward_dim,
        dropout_rate=0.1,
        attention_axes=2,  # For (batch, strand, length, feature), axis=2 is "length"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.attention_axes = attention_axes  # Make sure axis is correct for your input

        # 1) Self-attention sublayer
        self.attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="attn_norm"
        )
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            attention_axes=[self.attention_axes],
            name="mha",
        )
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate, name="attn_dropout")

        # 2) Feed-forward sublayer
        self.ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="ffn_norm"
        )
        self.ffn_dense1 = tf.keras.layers.Dense(
            feed_forward_dim, activation="gelu", name="ffn_dense1"
        )
        self.ffn_dropout1 = tf.keras.layers.Dropout(dropout_rate, name="ffn_dropout1")
        self.ffn_dense2 = tf.keras.layers.Dense(embed_dim, name="ffn_dense2")
        self.ffn_dropout2 = tf.keras.layers.Dropout(dropout_rate, name="ffn_dropout2")

    def call(self, inputs, mask=None, training=False, return_attention=False):
        # --- Multi-Head Self-Attention + Residual
        x_norm = self.attn_norm(inputs)
        # Reshape/transpose if needed to ensure attention is over length
        # For (batch, strand, length, feature), attention_axes=[2] is correct
        attn_out, attn_weights = self.mha(
            x_norm, x_norm, training=training, return_attention_scores=True
        )
        attn_out = self.attn_dropout(attn_out, training=training)
        x = inputs + attn_out

        # --- Feed-Forward Network + Residual
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn_dense1(x_norm)
        ffn_out = self.ffn_dropout1(ffn_out, training=training)
        ffn_out = self.ffn_dense2(ffn_out)
        ffn_out = self.ffn_dropout2(ffn_out, training=training)
        output = x + ffn_out

        if return_attention:
            return output, attn_weights
        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "feed_forward_dim": self.feed_forward_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return cfg


class CrossFrameAttention(tf.keras.layers.Layer):
    """
    Multi-Head Self-Attention across reading frames.

    For input shape (batch, frames, length, channels), this layer reshapes to
    (batch * length, frames, channels) and applies self-attention across the
    frame dimension. This allows each position in the sequence to attend to all
    6 reading frames simultaneously, enabling the model to learn relationships
    like: "frame 1 and frame 4 are reverse complements", "frame 2 has a shift",
    or "the correct ORF is frame 3".

    After attention, the tensor is reshaped back to (batch, frames, length, channels).

    Args:
        embed_dim: Dimension of the input/output features (must be divisible by num_heads).
        num_heads: Number of attention heads.
        feed_forward_dim: Hidden dimension of the feed-forward network.
        dropout_rate: Dropout rate for attention and FFN.
        use_ffn: Whether to include the feed-forward sublayer (default True).
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        feed_forward_dim,
        dropout_rate=0.1,
        use_ffn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.use_ffn = use_ffn

        # Self-attention sublayer — attention over frames (axis=1 after reshape)
        self.attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="attn_norm"
        )
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            attention_axes=[1],  # attend over frames after reshape
            name="mha",
        )
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate, name="attn_dropout")

        # Optional feed-forward sublayer
        if use_ffn:
            self.ffn_norm = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, name="ffn_norm"
            )
            self.ffn_dense1 = tf.keras.layers.Dense(
                feed_forward_dim, activation="gelu", name="ffn_dense1"
            )
            self.ffn_dropout1 = tf.keras.layers.Dropout(
                dropout_rate, name="ffn_dropout1"
            )
            self.ffn_dense2 = tf.keras.layers.Dense(embed_dim, name="ffn_dense2")
            self.ffn_dropout2 = tf.keras.layers.Dropout(
                dropout_rate, name="ffn_dropout2"
            )

    def call(self, inputs, mask=None, training=False, return_attention=False):
        # inputs: (batch, frames, length, channels)
        batch_size = tf.shape(inputs)[0]
        num_frames = tf.shape(inputs)[1]
        seq_len = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]

        # Reshape to (batch * length, frames, channels) for frame-wise attention
        # Each sequence position attends across all 6 frames
        x = tf.transpose(inputs, [0, 2, 1, 3])  # (B, L, F, C)
        x = tf.reshape(x, [-1, num_frames, channels])  # (B*L, F, C)

        # --- Multi-Head Self-Attention + Residual
        x_norm = self.attn_norm(x)
        attn_out, attn_weights = self.mha(
            x_norm, x_norm, training=training, return_attention_scores=True
        )
        attn_out = self.attn_dropout(attn_out, training=training)
        x = x + attn_out  # residual over frames

        # --- Optional Feed-Forward Network + Residual
        if self.use_ffn:
            x_norm = self.ffn_norm(x)
            ffn_out = self.ffn_dense1(x_norm)
            ffn_out = self.ffn_dropout1(ffn_out, training=training)
            ffn_out = self.ffn_dense2(ffn_out)
            ffn_out = self.ffn_dropout2(ffn_out, training=training)
            x = x + ffn_out

        # Reshape back to (batch, frames, length, channels)
        x = tf.reshape(x, [batch_size, seq_len, num_frames, channels])
        x = tf.transpose(x, [0, 2, 1, 3])  # (B, F, L, C)

        if return_attention:
            return x, attn_weights
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "feed_forward_dim": self.feed_forward_dim,
                "dropout_rate": self.dropout_rate,
                "use_ffn": self.use_ffn,
            }
        )
        return cfg


class AxialAttention(tf.keras.layers.Layer):
    """
    Axial attention for 4D sequence tensors (batch, frames, length, channels).

    Applies attention alternately along the length axis (intra-frame) and the
    frame axis (cross-frame). This captures both local sequence patterns within
    each frame and global relationships across frames.

    Args:
        embed_dim: Feature dimension.
        num_heads: Number of attention heads.
        feed_forward_dim: FFN hidden dimension.
        dropout_rate: Dropout rate.
        num_blocks: Number of (length-attn + frame-attn) blocks to stack.
        epsilon: Small constant for normalization layers.
        norm_type: Normalization layer to use after each (length + frame) block.
            One of ``layernorm``, ``masked_layernorm``, ``masked_dyt``,
            ``masked_batchnorm``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        feed_forward_dim,
        dropout_rate=0.1,
        num_blocks=1,
        epsilon=1e-6,
        norm_type="layernorm",
        alpha_init=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.epsilon = epsilon
        self.norm_type = norm_type.lower()
        self.alpha_init = alpha_init

        self.supports_masking = True

        self.length_attns = []
        self.frame_attns = []
        self.norms = []

        def _make_norm(name):
            if self.norm_type in ("layernorm", "layer_normalization"):
                return tf.keras.layers.LayerNormalization(epsilon=epsilon, name=name)
            if self.norm_type == "masked_layernorm":
                return MaskedLayerNormalization(epsilon=epsilon, name=name)
            if self.norm_type == "masked_dyt":
                return MaskedDYT(name=name, alpha_init=self.alpha_init)
            if self.norm_type == "masked_batchnorm":
                return MaskedBatchNorm(epsilon=epsilon, name=name)
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        for i in range(num_blocks):
            # Attention over length (intra-frame) — uses existing TransformerEncoder logic
            self.length_attns.append(
                TransformerEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout_rate=dropout_rate,
                    attention_axes=2,  # length axis in (B, F, L, C)
                    name=f"length_attn_{i}",
                )
            )
            self.norms.append(_make_norm(name=f"{self.norm_type}_post_{i}"))

            # Attention over frames (cross-frame)
            self.frame_attns.append(
                CrossFrameAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    dropout_rate=dropout_rate,
                    use_ffn=True,
                    name=f"frame_attn_{i}",
                )
            )

    def call(self, inputs, mask=None, training=False):
        x = inputs
        for length_attn, frame_attn, norm in zip(
            self.length_attns, self.frame_attns, self.norms
        ):
            residual = x
            x = length_attn(x, training=training)
            x = frame_attn(x, training=training)
            if self.norm_type == "masked_batchnorm":
                x = norm(x, training=training)
            elif self.norm_type in ("masked_layernorm", "masked_dyt"):
                x = norm(x, mask=mask)
            else:
                x = norm(x, training=training)
            x += residual

        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "feed_forward_dim": self.feed_forward_dim,
                "dropout_rate": self.dropout_rate,
                "num_blocks": self.num_blocks,
                "epsilon": self.epsilon,
                "norm_type": self.norm_type,
                "alpha_init": self.alpha_init,
            }
        )
        return cfg


class LocalAttention(tf.keras.layers.Layer):
    """Windowed self-attention along the sequence-length axis.

    Input shape: (batch, frames, length, channels)
    Output shape: (batch, frames, length, channels)

    Each position attends only to neighbors within ``window_size // 2`` on
    either side. This is cheaper and more appropriate than full self-attention
    for short sequences where long-range dependencies are noisy.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        window_size: int,
        dropout_rate: float = 0.1,
        num_blocks: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.window_size = window_size
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.supports_masking = True

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                {
                    "ln1": tf.keras.layers.LayerNormalization(
                        epsilon=1e-6, name=f"{self.name}_ln1_{i}"
                    ),
                    "mha": tf.keras.layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=embed_dim // num_heads,
                        dropout=dropout_rate,
                        name=f"{self.name}_mha_{i}",
                    ),
                    "ln2": tf.keras.layers.LayerNormalization(
                        epsilon=1e-6, name=f"{self.name}_ln2_{i}"
                    ),
                    "ffn1": tf.keras.layers.Dense(
                        feed_forward_dim,
                        activation="gelu",
                        name=f"{self.name}_ffn1_{i}",
                    ),
                    "ffn2": tf.keras.layers.Dense(
                        embed_dim, name=f"{self.name}_ffn2_{i}"
                    ),
                }
            )

    def _local_attention_mask(self, length: tf.Tensor):
        """Return a boolean (1, length, length) mask for the local window."""
        half_window = self.window_size // 2
        row = tf.range(length)[:, None]
        col = tf.range(length)[None, :]
        mask = tf.abs(row - col) <= half_window
        return mask[None, ...]

    def call(self, inputs, mask=None, training=None):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )

        shape = tf.shape(inputs)
        batch = shape[0]
        frames = shape[1]
        length = shape[2]
        channels = shape[3]

        # Reshape to (batch*frames, length, channels) for length-wise attention.
        x = tf.reshape(inputs, [batch * frames, length, channels])

        attn_mask = self._local_attention_mask(length)

        if mask is not None:
            # mask: (batch, frames, length) -> (batch*frames, length)
            seq_mask = tf.reshape(mask, [batch * frames, length])
            # Combine local window mask with sequence validity mask.
            key_mask = seq_mask[:, None, :]  # (B*F, 1, L)
            attn_mask = attn_mask & key_mask

        for block in self.blocks:
            x_norm = block["ln1"](x)
            attn = block["mha"](
                x_norm,
                x_norm,
                attention_mask=attn_mask,
                training=training,
            )
            x = x + attn

            x_norm = block["ln2"](x)
            ffn = block["ffn2"](block["ffn1"](x_norm))
            x = x + ffn

        return tf.reshape(x, [batch, frames, length, channels])

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "feed_forward_dim": self.feed_forward_dim,
                "window_size": self.window_size,
                "dropout_rate": self.dropout_rate,
                "num_blocks": self.num_blocks,
            }
        )
        return config


class ResidualBlockStack(tf.keras.layers.Layer):
    """A deterministic, reusable stack of ResidualBlock layers.

    The previous implementation built a Keras Functional submodel on the fly.
    Because Keras assigns incrementing internal IDs to Functional submodels,
    the nested weight names changed from build to build and legacy checkpoints
    could not be loaded reliably. This custom layer keeps stable sublayer names
    and supports the same ``block_size``/``return_nmd`` interface.
    """

    def __init__(self, block_size: int, in_shape: tuple | None = None, **kwargs):
        name = kwargs.pop("name", "resblock")
        return_nmd = kwargs.pop("return_nmd", False)
        use_masking = kwargs.pop("use_masking", True)

        # Only pass Layer-recognized kwargs to super().__init__
        layer_kwargs = {}
        for key in ("trainable", "dtype"):
            if key in kwargs:
                layer_kwargs[key] = kwargs.pop(key)
        super().__init__(name=name, **layer_kwargs)

        self.block_size = block_size
        self.return_nmd = return_nmd
        self.use_masking = use_masking
        self.supports_masking = use_masking
        self.blocks = []

        for i in range(block_size):
            if i != 0 and "use_1x1conv" in kwargs:
                kwargs.pop("use_1x1conv")

            block_name = f"{name}_{i}"
            block_number = f"{name.split('_')[-1]}{i}"
            block_kwargs = dict(
                kwargs,
                return_nmd=return_nmd if (i == block_size - 1) else False,
                use_masking=use_masking,
            )

            self.blocks.append(
                ResidualBlock(
                    block_number=block_number, name=block_name, **block_kwargs
                )
            )

    def call(self, inputs, training=None):
        x = inputs
        nmd = None
        for i, block in enumerate(self.blocks):
            out = block(x, training=training)
            if self.return_nmd and i == len(self.blocks) - 1:
                x, nmd = out
            else:
                x = out
        return (x, nmd) if self.return_nmd else x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_size": self.block_size,
                "return_nmd": self.return_nmd,
            }
        )
        return config


def ResidualBlock_wrapper(block_size: int, in_shape: tuple, **kwargs):
    """
    Build a sequential stack of ResidualBlock layers as a custom Layer.
    If return_nmd=True, the final block outputs both (x, nmd).
    """
    return ResidualBlockStack(block_size=block_size, in_shape=in_shape, **kwargs)


def causal_fft_convolve(u: tf.Tensor, h: tf.Tensor) -> tf.Tensor:
    """Depthwise causal convolution via FFT.

    Args:
        u: (batch, dim, L)
        h: (dim, L) — causal filter, one per channel. Must already be causal.

    Returns:
        y: (batch, dim, L) with the same dtype as ``u``.
    """
    orig_dtype = u.dtype
    u = tf.cast(u, tf.float32)
    h = tf.cast(h, tf.float32)

    tf.debugging.assert_rank(u, 3, message="causal_fft_convolve: u must be rank 3")
    tf.debugging.assert_rank(h, 2, message="causal_fft_convolve: h must be rank 2")
    u_dim = tf.shape(u)[1]
    h_dim = tf.shape(h)[0]
    tf.debugging.assert_equal(
        u_dim, h_dim, message="causal_fft_convolve: u and h must have matching dim"
    )
    u_L = tf.shape(u)[2]
    h_L = tf.shape(h)[1]
    tf.debugging.assert_equal(
        u_L, h_L, message="causal_fft_convolve: u and h must have matching length"
    )

    L = tf.shape(u)[-1]
    n = 2 * L - 1

    h_pad = tf.pad(h, [[0, 0], [0, n - L]])
    u_pad = tf.pad(u, [[0, 0], [0, 0], [0, n - L]])

    H = tf.signal.rfft(h_pad, fft_length=[n])
    U = tf.signal.rfft(u_pad, fft_length=[n])
    Y = U * tf.expand_dims(H, 0)

    y = tf.signal.irfft(Y, fft_length=[n])
    y = y[..., :L]
    return tf.cast(y, orig_dtype)


class HyenaFilter(tf.keras.layers.Layer):
    """Generate implicit long convolution filters h_t = Window(t) * FFN(PE(t)).

    When `seq_len` is fixed at construction, the positional encoding is
    pre-allocated; calling with a larger `seq_len` will raise an error. Pass
    `seq_len=None` and provide the length at call time for variable-length
    inputs.
    """

    def __init__(
        self,
        seq_len: int | None,
        dim: int,
        order: int = 2,
        pe_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.dim = dim
        self.order = order
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

    def build(self, input_shape):
        max_len = self.seq_len or 1
        pe = self._make_positional_encoding(max_len, self.pe_dim)
        self.pos_encoding = self.add_weight(
            shape=pe.shape,
            initializer=tf.keras.initializers.Constant(pe),
            trainable=False,
            dtype=tf.float32,
            name="pos_encoding",
        )

        self.ffns = []
        for i in range(self.order):
            ffn = tf.keras.Sequential(name=f"ffn_{i}")
            for j in range(self.num_layers):
                is_last = j == self.num_layers - 1
                units = self.dim if is_last else self.hidden_dim
                act = self.activation if not is_last else None
                ffn.add(tf.keras.layers.Dense(units, activation=act))
            setattr(self, f"ffn_{i}", ffn)
            ffn.build((max_len, self.pe_dim))
            self.ffns.append(ffn)

        self.alphas = self.add_weight(
            shape=(self.order, self.dim),
            initializer="ones",
            trainable=True,
            name="alphas",
        )
        self.biases = self.add_weight(
            shape=(self.order, self.dim),
            initializer="zeros",
            trainable=True,
            name="biases",
        )
        super().build(input_shape)

    def _make_positional_encoding(self, length: int, dim: int) -> np.ndarray:
        pos = np.arange(length, dtype=np.float32)[:, None]
        div = np.exp(np.arange(0, dim, 2, dtype=np.float32) * -(np.log(10000.0) / dim))
        pe_sin = np.sin(pos * div)
        pe_cos = np.cos(pos * div)
        pe = np.reshape(np.stack([pe_sin, pe_cos], axis=-1), (length, -1))
        return pe[:, :dim]

    def call(self, seq_len: tf.Tensor | None = None):
        L = seq_len if seq_len is not None else self.seq_len
        if L is None:
            raise ValueError("HyenaFilter requires seq_len at build or call time")
        L = tf.cast(L, tf.int32)

        if self.seq_len is not None:
            pe = self.pos_encoding[:L]
        else:
            pe = self._make_positional_encoding(L, self.pe_dim)
        t = tf.range(L, dtype=tf.float32)

        filters = []
        for i in range(self.order):
            x = self.ffns[i](pe)
            window = (
                tf.exp(-self.alphas[i][None, :] * t[:, None]) + self.biases[i][None, :]
            )
            h = window * x
            filters.append(h)

        return tf.transpose(tf.stack(filters, axis=0), [0, 2, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "seq_len": self.seq_len,
                "dim": self.dim,
                "order": self.order,
                "pe_dim": self.pe_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
            }
        )
        return cfg


class HyenaOperator(tf.keras.layers.Layer):
    """Core Hyena recurrence: z^1 = v; z^{n+1} = x^n \u2299 (h^n * z^n); y = z^{N+1}."""

    def __init__(
        self,
        dim: int,
        seq_len: int | None,
        order: int = 2,
        filter_hidden: int = 32,
        filter_layers: int = 2,
        filter_activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.seq_len = seq_len
        self.order = order
        self.filter_hidden = filter_hidden
        self.filter_layers = filter_layers
        self.filter_activation = filter_activation

    def build(self, input_shape):
        self.projs = [
            tf.keras.layers.Dense(self.dim, use_bias=False, name=f"proj_{i}")
            for i in range(self.order + 1)
        ]
        for i, proj in enumerate(self.projs):
            setattr(self, f"proj_{i}", proj)
            proj.build((None, self.dim))

        self.filter_gen = HyenaFilter(
            seq_len=self.seq_len,
            dim=self.dim,
            order=self.order,
            hidden_dim=self.filter_hidden,
            num_layers=self.filter_layers,
            activation=self.filter_activation,
        )
        self.filter_gen.build((None, self.seq_len or 1, self.dim))
        super().build(input_shape)

    def call(self, x):
        # x: (batch, L, dim)
        proj = [p(x) for p in self.projs]
        z = tf.transpose(proj[0], [0, 2, 1])  # (batch, dim, L)
        seq_len = tf.shape(z)[-1]
        filters = self.filter_gen(seq_len)

        for i in range(self.order):
            gate = tf.transpose(proj[i + 1], [0, 2, 1])
            h = filters[i]
            conv = causal_fft_convolve(z, h)
            z = conv * gate

        return tf.transpose(z, [0, 2, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "dim": self.dim,
                "seq_len": self.seq_len,
                "order": self.order,
                "filter_hidden": self.filter_hidden,
                "filter_layers": self.filter_layers,
                "filter_activation": self.filter_activation,
            }
        )
        return cfg


class HyenaBlock(tf.keras.layers.Layer):
    """Apply the Hyena block.

    Masking is supported by zeroing padded positions before and after the block.
    Note that the internal LayerNormalization is not masked, so padded positions
    still affect the normalization statistics.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int | None = None,
        order: int = 2,
        filter_hidden: int = 32,
        filter_layers: int = 2,
        filter_activation: str = "gelu",
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.seq_len = seq_len
        self.order = order
        self.filter_hidden = filter_hidden
        self.filter_layers = filter_layers
        self.filter_activation = filter_activation
        self.dropout = dropout
        self.supports_masking = True

    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm.build(input_shape)
        self.hyena = HyenaOperator(
            dim=self.dim,
            seq_len=self.seq_len,
            order=self.order,
            filter_hidden=self.filter_hidden,
            filter_layers=self.filter_layers,
            filter_activation=self.filter_activation,
        )
        self.hyena.build((None, input_shape[2], self.dim))
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.dropout_layer.build((None, input_shape[2], self.dim))
        super().build(input_shape)

    def call(self, x, mask=None, training=False):
        # x: (batch, strands, length, dim)
        batch = tf.shape(x)[0]
        strands = tf.shape(x)[1]
        length = tf.shape(x)[2]

        mask_float = None
        if mask is not None:
            # mask shape is typically (batch, strands, length); expand to match x
            mask_float = tf.cast(mask, x.dtype)
            while mask_float.shape.rank < x.shape.rank:
                mask_float = tf.expand_dims(mask_float, axis=-1)
            x = x * mask_float

        residual = x
        x = self.norm(x)
        x = tf.reshape(x, [batch * strands, length, self.dim])
        x = self.hyena(x)
        x = self.dropout_layer(x, training=training)
        x = tf.reshape(x, [batch, strands, length, self.dim])
        out = x + residual

        if mask_float is not None:
            out = out * mask_float
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "dim": self.dim,
                "seq_len": self.seq_len,
                "order": self.order,
                "filter_hidden": self.filter_hidden,
                "filter_layers": self.filter_layers,
                "filter_activation": self.filter_activation,
                "dropout": self.dropout,
            }
        )
        return cfg
