import tensorflow as tf
import tensorflow.keras as keras
from keras.src.layers.convolutional.base_conv import BaseConv

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
        return tf.nn.gelu(inputs)
    
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
    def __init__(self,**kwargs):
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
    '''Apply 1D pooling along a defined axis'''
    def __init__(self, pool_size, axis, **kwargs):
        super(CustomPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.axis = axis

    def call(self, inputs):
        return tf.nn.pool(inputs, window_shape=[1, self.pool_size, 1, 1], strides=[1, self.pool_size, 1, 1],
                          pooling_type='MAX', padding='SAME', data_format='NHWC')

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = (input_shape[self.axis] - self.pool_size) // self.pool_size + 1
        return tuple(output_shape)

class GlobalMaxPoolingPerFeature(tf.keras.layers.Layer):
    ''''Apply max_reduce along the last axis. Re-implementation of GlobalMaxPooling1D layer for 
    Biological sequences'''
    def __init__(self, **kwargs):
        super(GlobalMaxPoolingPerFeature, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the feature axis
        return tf.reduce_max(inputs, axis=-1, keepdims=False, name='global_max_per_position')

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0], input_shape[2])
    
class MaxReduce(tf.keras.layers.Layer):
    '''Apply max_reduce along the frame axis'''
    def __init__(self, **kwargs):
        super(MaxReduce, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the frame axis
        return tf.reduce_max(inputs, axis=1, keepdims=False, name='max_reduce')

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0],input_shape[2], input_shape[3])
    
class MeanReduce(tf.keras.layers.Layer):
    '''Apply mean_reduce along the frame axis'''
    def __init__(self, **kwargs):
        super(MeanReduce, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the frame axis
        return tf.reduce_mean(inputs, axis=1, keepdims=False, name='mean_reduce')

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0],input_shape[2], input_shape[3])

class MaskedNeuralMean(tf.keras.layers.Layer):
    """
    Masked Neural Mean layer returns the channelwise mean of in the input
    tensor [batch, strands, length, channels]
    This layer has no parameters.
    """
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.supports_masking = True


    def call(self, inputs, mask=None, training=False):
        
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # Ensure mask is the same type as inputs
            mask = tf.expand_dims(mask, axis=-1)  # Broadcast mask to have the same number of channels as inputs
            valid_elements = tf.reduce_sum(mask, axis=[0, 1, 2])  # Count valid elements per feature map

            # Apply mask to inputs to ignore padded elements
            masked_inputs = inputs * mask

            # Calculate mean and variance only over valid elements
            mean = tf.reduce_sum(masked_inputs, axis=[0, 1, 2]) / valid_elements
            #variance = tf.reduce_sum(mask * tf.square(masked_inputs - mean), axis=[0, 1, 2]) / valid_elements

        else:
            # Standard batch normalization when no mask is provided
            mean, _ = tf.nn.moments(inputs, axes=[0, 1, 2])

        return mean
     
class SumReduce(tf.keras.layers.Layer):
    '''Apply sum_reduce along the frame axis'''
    def __init__(self, **kwargs):
        super(SumReduce, self).__init__(**kwargs)

    def call(self, inputs):
        # Take the maximum value along the frame axis
        return tf.reduce_sum(inputs, axis=1, keepdims=False, name='max_reduce')

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0],input_shape[2], input_shape[3])
    
class MaskedMaxPooling1D(tf.keras.layers.Layer):
    """
    MaxPooling1D implementation that accepts an incomming masking tensor
    and outputs a mask tensor corresponding to the output dimention of layer's
    output
    """
    
    def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
        super(MaskedMaxPooling1D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding.lower()

    def call(self, inputs, mask=None):
        
         # Reshape for MaskedMaxpooling1D
        input_shape = tf.shape(inputs)
        self.input_dtype = inputs.dtype
        batch_size = input_shape[0]
        dim1 = input_shape[1]
        dim2 = input_shape[2]
        dim3 = input_shape[3]
        
        if mask is not None:
            # Expand mask dimensions to match inputs
            mask = tf.cast(mask, dtype=inputs.dtype)
            inputs = inputs *  tf.expand_dims(mask, axis=-1)  # Zero out masked positions
            
        reshaped_inputs = tf.reshape(inputs, (-1, dim2, dim3))
        outputs = tf.nn.max_pool1d(
            reshaped_inputs,
            ksize=self.pool_size,
            strides=self.strides,
            padding=self.padding.upper()
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
        if hasattr(self, '_output_mask'):
            return self._output_mask
        else:
            return None
    def compute_output_mask(self, mask=None):
        if mask is None:
            return None
        else:
            mask = tf.cast(mask, dtype=self.input_dtype)
            mask = tf.expand_dims(mask, axis=-1)
   
            pooled_mask = tf.nn.max_pool1d(
                mask,
                ksize=self.pool_size,
                strides=self.strides,
                padding=self.padding.upper()
            )
            pooled_mask = tf.cast(pooled_mask, dtype=tf.bool)
            
            #pooled_mask = tf.squeeze(pooled_mask, axis=-1)
            return pooled_mask
        
    def compute_output_shape(self, input_shape):
        
        # Compute the output shape along the pooling axis
        axis_length = input_shape[2]
        if axis_length is not None:
            if self.padding == 'valid':
                output_length = (axis_length - self.pool_size) // self.strides + 1
            elif self.padding == 'same':
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
                name='gamma',
                shape=param_shape,
                initializer='ones',
                trainable=True
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer='zeros',
                trainable=True
            )
        else:
            self.beta = None

        super(MaskedLayerNormalization, self).build(input_shape)

    def call(self, inputs, mask=None):
        
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            mask = tf.expand_dims(mask, -1)
            mask = tf.stop_gradient(mask)

            # Compute the mean over unmasked positions
            masked_inputs = inputs * mask
            mask_sum = tf.reduce_sum(mask, axis=[1,2], keepdims=True)
            mean = tf.reduce_sum(masked_inputs, axis=-1, keepdims=True) / (mask_sum + self.epsilon)
            
            # Compute the variance over unmasked positions
            squared_diff = tf.square((inputs - mean) * mask)
            variance = tf.reduce_sum(squared_diff, axis=-1, keepdims=True) / (mask_sum + self.epsilon)
        else:
            mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)

        # Normalize the inputs
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        # Apply scale and center
        if self.scale:
            normalized = normalized * self.gamma
        if self.center:
            normalized = normalized + self.beta

        # Re-apply the mask to keep masked positions at zero
        if mask is not None:
            normalized = normalized * mask

        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MaskedLayerNormalization, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
        })
        return config
    
class MaskedGlobalAvgPooling(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def call(self, inputs, mask=None):
            # Assuming inputs and mask have the same shape (batch_size, height, width, channels)
            if mask is not None:
                mask = tf.cast(mask, inputs.dtype)  # Convert mask to the same dtype as inputs
                mask = tf.expand_dims(mask, axis=-1)
                inputs = inputs * mask  # Zero out masked positions

                # Sum over the spatial dimensions (batch, frames, seq_length, projections)
                masked_sum = tf.reduce_sum(inputs, axis=[1, 2])
                mask_sum = tf.reduce_sum(mask, axis=[1, 2])

                # Avoid division by zero
                mask_sum = tf.maximum(mask_sum, tf.keras.backend.epsilon())

                # Compute the masked average
                return tf.math.divide_no_nan(masked_sum, mask_sum)
            else:
                # Default to regular global average pooling if no mask is provided
                return tf.reduce_mean(inputs, axis=[1, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])  # Output shape is (batch_size, num_channels)


class MaskedBatchNorm(tf.keras.layers.Layer):
    """
    Masked batch normalization ignores the masked positions when calculating the summary 
    statistics and normalization. This custom layer has an extra kwarg that returns nmd
    vectors u[exmple] - u[train]
    """
    def __init__(self, epsilon=1e-5, momentum=0.99, return_nmd=False, **kwargs):
        super().__init__( **kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.supports_masking = True
        self.return_nmd = return_nmd

    def build(self, input_shape):
        # Trainable parameters: gamma (scale) and beta (shift)
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer='ones', trainable=True, name='gamma')
        self.beta = self.add_weight(shape=input_shape[-1:], initializer='zeros', trainable=True, name='beta')

        # Running mean and variance (used during inference)
        self.moving_mean = self.add_weight(shape=input_shape[-1:], initializer='zeros', trainable=False, name='moving_mean')
        self.moving_variance = self.add_weight(shape=input_shape[-1:], initializer='ones', trainable=False, name='moving_variance')

    def call(self, inputs, mask=None, training=False):

        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)  # Ensure mask is the same type as inputs
            mask = tf.expand_dims(mask, axis=-1)  # Broadcast mask to have the same number of channels as inputs
            valid_elements = tf.reduce_sum(mask, axis=[0, 1, 2])  # Count valid elements per feature map

            # Apply mask to inputs to ignore padded elements
            masked_inputs = inputs * mask

            # Calculate mean and variance only over valid elements
            mean_batch = tf.reduce_sum(masked_inputs, axis=[0, 1, 2]) / valid_elements
            variance_batch = tf.reduce_sum(mask * tf.square(masked_inputs - mean_batch), axis=[0, 1, 2]) / valid_elements

        else:
            # Standard batch normalization when no mask is provided
            mean_batch, variance_batch = tf.nn.moments(inputs, axes=[0, 1, 2])
    
        if training:
            # Update moving averages of mean and variance during training
            new_moving_mean = self.moving_mean * self.momentum + mean_batch * (1.0 - self.momentum)
            new_moving_variance = self.moving_variance * self.momentum + variance_batch * (1.0 - self.momentum)
            self.moving_mean.assign(new_moving_mean)
            self.moving_variance.assign(new_moving_variance)


        # Normalize the inputs
        normalized_inputs = (inputs - self.moving_mean) / tf.sqrt(self.moving_variance + self.epsilon)
        # Scale and shift (gamma and beta)
        output = self.gamma * normalized_inputs + self.beta

        # Return the result, applying the mask again to keep padded areas unchanged (optional)
        if self.return_nmd:
            mean_channel, _ = tf.nn.moments(inputs, axes=[1, 2])
            return output, mean_channel - self.moving_mean
        return output


class MaskedConv1D(tf.keras.layers.Layer):
    """
    Masked 1D convolution that accepts an optional mask tensor and sets 
    mask positions to zero before applying convolution. Also, propagates the mask
    to the next layer. Accepts inputs like (batch, frames, length, channels) where 
    batch and length dimention can be unknown.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 axis=-1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
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

    def build(self, input_shape):
        channel_axis = self.axis
        input_dim = input_shape[channel_axis]
        kernel_shape = (self.kernel_size, input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='kernel',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        trainable=True)
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs, mask=None):

        input_shape = tf.shape(inputs)
        # for reshaping after conv
        output_shape = self.compute_output_shape(input_shape)
        batch_size = input_shape[0]
        dim1 = input_shape[1]
        dim2 = input_shape[2]
        dim3 = input_shape[3]

        if mask is not None:
            # Expand mask dimensions to match inputs
            mask_dtype = mask.dtype
            mask = tf.cast(mask, dtype=inputs.dtype)
            inputs = inputs *  tf.expand_dims(mask, axis=-1)  # Zero out masked positions
            # Compute reshape mask after convolution
            reshaped_mask = tf.reshape(mask, (-1, dim2))
            output_mask = self.compute_output_mask(reshaped_mask)
            
            # Set the mask for subsequent layers
            output_mask = tf.reshape(output_mask, shape=output_shape[:-1]) # bactch, 6, len
            self._output_mask = output_mask
        else:
            output_mask = None   

        # Reshape for Conv1D

        
        reshaped_inputs = tf.reshape(inputs, (-1, dim2, dim3)) # batch*6, len, features
        # Perform convolution
        outputs = tf.nn.conv1d(
            input=reshaped_inputs,
            filters=self.kernel,
            stride=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            data_format='NWC'  # (batch, width, channels)
        )
        
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format='NWC')

        if self.activation is not None:
            outputs = self.activation(outputs)
        
        
       

        outputs = tf.reshape(outputs, output_shape) # bactch, 6, len, features
        
        return outputs 

    def compute_output_mask(self, input_mask):
        """
        masked positions   == 0
        unmasked positions == 1
        """
        # Compute the mask for the outputs
        kernel_size = self.kernel_size
        strides = self.strides

        # Convolve the mask with a kernel of ones
        mask = tf.expand_dims(input_mask, axis=-1)

        mask_kernel = tf.ones((kernel_size, 1, 1), dtype=mask.dtype)
        output_mask = tf.nn.conv1d(
            input=mask,
            filters=mask_kernel,
            stride=strides,
            padding=self.padding,
            dilations=self.dilation_rate,
            data_format='NWC'
        )

        # If all positions covered by the kernel are valid (sum equals kernel_size), output is valid
        output_mask = tf.equal(output_mask, kernel_size)
        output_mask = tf.squeeze(output_mask, axis=-1)
        return output_mask

    def compute_mask(self, inputs, mask=None):
        # Return the output mask computed in call()
        if hasattr(self, '_output_mask'):
            return self._output_mask
        else:
            return None

    def get_config(self):
        config = super(MaskedConv1D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'dilation_rate': self.dilation_rate,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def compute_output_shape(self, input_shape):
        '''
        compute the output shape only if the length dimention is not None.
        '''
        length = input_shape[2]
        if length is not None:
            if self.padding == 'SAME':
                out_length = (length + self.strides - 1) // self.strides
            elif self.padding == 'VALID':
                out_length = (length - self.dilation_rate * (self.kernel_size - 1) - 1) // self.strides + 1
            else:
                raise ValueError('Invalid padding type.')
            out_length = out_length
            return (input_shape[0], input_shape[1], out_length, self.filters)
        else:
            return (input_shape[0], input_shape[1], input_shape[2], self.filters)


class ResidualBlock(tf.keras.layers.Layer): 
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, kernel_size=5, dilation_rate=1, use_1x1conv=False, strides=1, block_number=1 ,**kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.block_number = block_number
        self.conv1 = MaskedConv1D(num_channels,
                                  padding='same',
                                  use_bias=False,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  strides=strides,
                                  name=f'masked_conv1d_blk_{self.block_number}_1',
                                  kernel_initializer=tf.keras.initializers.HeUniform()
                                 )
        self.conv2 = MaskedConv1D(num_channels,
                                  kernel_size=kernel_size,
                                  dilation_rate=dilation_rate,
                                  use_bias=False,
                                  padding='same',
                                  name=f'masked_conv1d_blk{self.block_number}_2',
                                  kernel_initializer=tf.keras.initializers.HeUniform()
                                 )
        self.conv3 = None
        if use_1x1conv or strides > 1:
            self.conv3 = MaskedConv1D(num_channels,
                                      use_bias=False,
                                      kernel_size=1,
                                      strides=strides)
        self.bn1 = MaskedBatchNorm(name=f'masked_batchnorm_blk{self.block_number}_1')
        self.bn2 = MaskedBatchNorm(name=f'masked_batchnorm_blk{self.block_number}_2')
        self.add = MaskedAdd(name=f'resblock_batchnorm_blk{self.block_number}')

        match kwargs.get("activation", "gelu"):
            case "gelu":
                self.activation = GeLU
            case "relu":
                self.activation = ReLU
        
    def call(self, inputs, mask=None):
        x = self.activation(name=f'resblock_activation_blk{self.block_number}_1')(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        if self.conv3 is not None:
            inputs = self.conv3(inputs)
        x = self.add([x, inputs])
        return self.activation(name=f'resblock_batchnorm_blk{self.block_number}_2')(x)
    

    def compute_output_shape(self, input_shape):
        '''
        compute the output shape only if the length dimention is not None.
        '''
        return input_shape 
    # Todo: implement output mask computation -> return the mask computed by the last layer?

# To do: implement a method to set epsilon considering the data type of the tensors passed to the layers
# if not implemented correctly, this can lead to overflow/underflow issues. 
