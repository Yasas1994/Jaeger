"""

Copyright (c) 2024 Yasas Wijesekara

"""

import tensorflow as tf
import numpy as np


def rc_cnn(
    x, name="",
    filters=16,
    stride=1,
    kernel_size=5,
    dilation_rate=1,
    padding="same"
):

    f = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        strides=stride,
        kernel_initializer=tf.keras.initializers.HeUniform(),
        padding=padding,
        name=name,
    )
    outputs = [f(i) for i in x]

    return outputs


class SplitLayer(tf.keras.layers.Layer):
    def __init__(self, num_splits, axis=1, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)
        self.num_splits = num_splits
        self.axis = axis

    def build(self, input_shape):
        # No trainable weights needed
        super(SplitLayer, self).build(input_shape)

    def call(self, inputs):
        splits = tf.split(inputs, self.num_splits, axis=self.axis)
        return splits

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = input_shape[self.axis] // self.num_splits
        return [tuple(output_shape) for _ in range(self.num_splits)]


def rc_batchnorm(x, name):

    f = tf.keras.layers.BatchNormalization(name=f"bn_{name}")

    outputs = [f(i) for i in x]

    return outputs


def rc_batchnorm2(x, name):

    splits = len(x)
    x = tf.keras.layers.Concatenate(axis=0)(x)
    x = tf.keras.layers.BatchNormalization(name=f"bn_{name}")(x)
    x = SplitLayer(axis=0, num_splits=splits)(x)

    return x


def rc_maxpool(x, pool_size=2):
    f = tf.keras.layers.MaxPooling1D(pool_size=pool_size)
    outputs = [f(i) for i in x]

    return outputs


def rc_gelu(x):
    f = tf.nn.gelu
    outputs = []
    outputs = [f(i) for i in x]

    return outputs


def rc_resnet_block(
    x,
    name,
    kernel_size=[3, 3],
    dilation_rate=[1, 1],
    filters=[16, 16],
    add_residual=True,
):  # simple resnet for viruses#
    """x: input tensor
    name:name for the block
    kernel_size: a list specifying the kernel size of each conv layer
    dilation_rate: a list specifying the dilation rate of each conv layer
    filter:  a list specifying the number of filters of each conv layer
    shared_weights: whether to use reverse conplement parameter sharing.(True)
    add_residual: whether to add residual connections.(True)
    """

    xx = rc_cnn(
        x,
        name=f"{name}{1}",
        filters=filters[0],
        kernel_size=kernel_size[0],
        padding="same",
        dilation_rate=dilation_rate[0],
    )
    xx = rc_gelu(xx)
    xx = rc_batchnorm(xx, name=f"{name}{1}")

    # Create layers
    for n, (k, d, f) in enumerate(zip(kernel_size[1:],
                                      dilation_rate[1:],
                                      filters[1:])):
        xx = rc_cnn(
            xx,
            name=f"{name}{n+2}",
            filters=f,
            kernel_size=k,
            padding="same",
            dilation_rate=d,
        )
        xx = rc_gelu(xx)
        xx = rc_batchnorm(xx, name=f"{name}{n+2}")

    # scale up the skip connection output if the filter sizes are different

    if (filters[-1] != filters[0] or x[-1].shape[-1] != filters[-1])\
       and add_residual:
        x = rc_cnn(
            x,
            name=f"{name}_skip",
            filters=f,
            kernel_size=1,
            padding="same",
            dilation_rate=1,
        )
        x = rc_gelu(x)
        x = rc_batchnorm(x, name=f"{name}_skip")

    # Add Residue
    add = tf.keras.layers.Add()
    if add_residual:
        outputs = [add(i) for i in zip(x, xx)]
        return rc_gelu(outputs)
    else:
        return rc_gelu(xx)


def ConvolutionalTower(inputs, num_res_blocks=5, add_residual=True):
    """
    Covolutional tower to increase the receptive filed size based on dilated
    convolutions.

    order of operations

    original batch norm paper suggested that BN should be applied
    before the non-linear transformation. However, in practice, BN
    is applied after the non-linearity as it is shown perform better.

    linear transformation -> non-linearity -> batch norm
    """

    x = rc_cnn(
        inputs,
        name="block1_0",
        filters=128,
        stride=1,
        kernel_size=9,
        dilation_rate=1,
        padding="same",
    )
    x = rc_gelu(x)
    x = rc_batchnorm(x, name="block1_1")
    x = rc_maxpool(x, pool_size=2)

    x = rc_cnn(
        x,
        name="block1_1",
        filters=128,
        stride=1,
        kernel_size=5,
        dilation_rate=2,
        padding="same",
    )
    x = rc_gelu(x)
    x = rc_batchnorm(x, name="block1_2")
    x = rc_maxpool(x, pool_size=2)

    if num_res_blocks:
        for i, n in enumerate(range(num_res_blocks)):
            x = (
                lambda x, n: rc_resnet_block(
                    x,
                    name=f"block2_{n}",
                    kernel_size=[5, 5],
                    dilation_rate=[3 + i, 3 + i],
                    filters=[128, 128],
                    add_residual=add_residual,
                )
            )(x, n)

    return tf.keras.layers.Add()(x)


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)

        position_embedding_matrix = self.get_position_encoding(
            sequence_length, output_dim
        )

        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False,
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def call(self, inputs):
        sequence_length = tf.shape(inputs)[-1]
        batch_size = tf.shape(inputs)[0]
        position_indices = [
            [c for c in range(sequence_length)] for i in range(batch_size)
        ]  # tf.range(tf.shape(inputs)[-2])
        position_indices = tf.Variable(position_indices)
        # embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_indices  # embedded_words +


class Patches(tf.keras.layers.Layer):

    def __init__(self, num_patches, patch_size, name="split"):
        super(Patches, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.layer_name = name

    def call(self, data):
        batch_size = tf.shape(data)[0]
        splitted_seq = tf.split(
            data,
            num_or_size_splits=self.num_patches,
            axis=1,
            num=self.patch_size,
            name=self.layer_name,
        )
        patches = tf.stack(splitted_seq, axis=1, name="stack")
        patches = tf.reshape(
            patches, [batch_size, self.num_patches, self.patch_size * 4]
        )

        return patches


class PatchEncoder(tf.keras.layers.Layer):
    # Parch encoding + Position encoding
    def __init__(
        self,
        num_patches,
        projection_dim=None,
        embed_input=False,
        use_sine=True
    ):  # num_patches == sequence length when input comes from a conv block
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches

        if embed_input:
            self.projection = tf.keras.layers.Dense(units=projection_dim)
        else:
            self.projection = None

        if use_sine:
            self.position_embedding_layer = tf.keras.layers.Embedding(
                input_dim=num_patches, output_dim=projection_dim
            )
        else:
            position_embedding_matrix = self.get_position_encoding(
                num_patches, projection_dim
            )

            self.position_embedding_layer = tf.keras.layers.Embedding(
                input_dim=num_patches,
                output_dim=projection_dim,
                weights=[position_embedding_matrix],
                trainable=False,
            )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        if self.projection is not None:
            input_projection = self.projection(patches)
        else:
            input_projection = patches

        encoded = input_projection + self.position_embedding_layer(positions)

        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def Baseline_model(input_shape=None):  # archeae model 1
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22,
                                                4,
                                                name="aa",
                                                mask_zero=True)
    embeddings = [embedding_layer(i) for i in [f1input,
                                               f2input,
                                               f3input,
                                               r1input,
                                               r2input,
                                               r3input]]
    # A block
    x = ConvolutionalTower(embeddings, num_res_blocks=None)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    # C block
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-1")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-2")(x)
    out = tf.keras.layers.Dense(4, name="outdense")(x)
    return [f1input, f2input, f3input, r1input, r2input, r3input], out


def Res_model(input_shape=None):  # archeae model 1
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22,
                                                4,
                                                name="aa",
                                                mask_zero=True)
    embeddings = [embedding_layer(i) for i in [f1input,
                                               f2input,
                                               f3input,
                                               r1input,
                                               r2input,
                                               r3input]]
    # A block
    x = ConvolutionalTower(embeddings, num_res_blocks=5)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    # C block
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-1")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-2")(x)
    out = tf.keras.layers.Dense(4, name="outdense")(x)
    return [f1input, f2input, f3input, r1input, r2input, r3input], out


def WRes_model(input_shape=None):  # archeae model 1
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22,
                                                4,
                                                name="aa",
                                                mask_zero=True)
    embeddings = [embedding_layer(i) for i in [f1input,
                                               f2input,
                                               f3input,
                                               r1input,
                                               r2input,
                                               r3input]]
    # B block
    x = ConvolutionalTower(embeddings,
                           num_res_blocks=5,
                           add_residual=False)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    # C block
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-1")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-2")(x)
    out = tf.keras.layers.Dense(4, name="outdense")(x)
    return ([f1input, f2input, f3input, r1input, r2input, r3input],
            {"output": out})


def WRes_model_embeddings(input_shape=None, dropout_active=True):
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22,
                                                4,
                                                name="aa",
                                                mask_zero=True)
    embeddings = [embedding_layer(i) for i in [f1input,
                                               f2input,
                                               f3input,
                                               r1input,
                                               r2input,
                                               r3input]]
    # B block
    x = ConvolutionalTower(embeddings, num_res_blocks=5, add_residual=False)
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    # C block
    x = tf.keras.layers.Dropout(0.5)(x,
                                     training=dropout_active)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-1")(x)
    x = tf.keras.layers.Dropout(0.5)(x,
                                     training=dropout_active)
    gmp = tf.keras.layers.Dense(128,
                                activation=tf.nn.gelu,
                                name="augdense-2")(x)
    out = tf.keras.layers.Dense(4,
                                name="outdense")(gmp)
    return [f1input, f2input, f3input, r1input, r2input, r3input], {
        "output": out,
        "embedding": gmp,
    }


def LSTM_model(input_shape=None):
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22,
                                                4,
                                                name="aa",
                                                mask_zero=True)
    embeddings = [embedding_layer(i) for i in [f1input,
                                               f2input,
                                               f3input,
                                               r1input,
                                               r2input,
                                               r3input]]

    x = ConvolutionalTower(embeddings, num_res_blocks=5)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128,
                             name="lstm"),
        name="bidirlstm"
    )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-1")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128,
                              activation=tf.nn.gelu,
                              name="augdense-2")(x)
    out = tf.keras.layers.Dense(4,
                                name="outdense")(x)
    return [f1input, f2input, f3input, r1input, r2input, r3input], out


def Vitra(
    input_shape=(None,),
    num_patches=512,
    transformer_layers=4,
    num_heads=4,
    att_dropout=0.1,
    projection_dim=128,
    att_hidden_units=[128, 128],
    mlp_hidden_units=[128, 128],
    mlp_dropout=0.1,
    use_global=True,
    global_type="max",
):

    f2input = tf.keras.Input(shape=input_shape, name="forward_2")
    f3input = tf.keras.Input(shape=input_shape, name="forward_3")
    r1input = tf.keras.Input(shape=input_shape, name="reverse_1")
    r2input = tf.keras.Input(shape=input_shape, name="reverse_2")
    f1input = tf.keras.Input(shape=input_shape, name="forward_1")
    r3input = tf.keras.Input(shape=input_shape, name="reverse_3")
    embedding_layer = tf.keras.layers.Embedding(22,
                                                4,
                                                name="aa",
                                                mask_zero=True)
    embeddings = [embedding_layer(i) for i in [f1input,
                                               f2input,
                                               f3input,
                                               r1input,
                                               r2input,
                                               r3input]]
    # Create patches.
    patches = ConvolutionalTower(embeddings, num_res_blocks=5)
    # patches = Patches(num_patches=num_patches,patch_size=patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(
        num_patches=num_patches, projection_dim=projection_dim
    )(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=att_dropout
        )(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=att_hidden_units, dropout_rate=mlp_dropout)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    rep = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    if use_global:
        if global_type == "average":
            rep = tf.keras.layers.GlobalAveragePooling1D()(rep)
        elif global_type == "max":
            rep = tf.keras.layers.GlobalMaxPooling1D()(rep)
    else:
        rep = tf.keras.layers.Flatten()(rep)

    rep = tf.keras.layers.Dropout(0.1)(rep)
    # Add MLP.
    features = mlp(rep,
                   hidden_units=mlp_hidden_units,
                   dropout_rate=0.5)
    # Classify outputs.
    logits = tf.keras.layers.Dense(4)(features)
    # Create the Keras model.
    return [f1input, f2input, f3input, r1input, r2input, r3input], logits


# layers for the second generation models
class GlobalMaxPoolingPerFeature(tf.keras.layers.Layer):
    """
    Apply max_reduce along the last axis. Re-implementation of
    GlobalMaxPooling1D layer for Biological sequences
    """

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
        return tf.reduce_mean(inputs,
                              axis=1,
                              keepdims=False,
                              name="max_reduce")

    def compute_output_shape(self, input_shape):
        # Output shape will have the same batch size and the number of features
        return (input_shape[0], input_shape[2], input_shape[3])


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


def resnet_block_g2(
    x,
    name,
    kernel_size=[3, 3],
    dilation_rate=[1, 1],
    filters=[16, 16],
    add_residual=True,
):
    """
    Args:
    ----
        x: input tensor
        name:name for the block
        kernel_size: a list specifying the kernel size of each conv layer
        dilation_rate: a list specifying the dilation rate of each conv layer
        filter:  a list specifying the number of filters of each conv layer
        shared_weights: whether to use reverse conplement parameter sharing.
                        defaault (True)
        add_residual: whether to add residual connections.(True)
    """

    xx = tf.keras.layers.Conv1D(
        filters[0],
        kernel_size[0],
        strides=1,
        dilation_rate=dilation_rate[0],
        padding="same",
        name=f"{name}_{1}",
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=2),
    )(x)

    xx = tf.keras.layers.BatchNormalization(axis=-1,
                                            name=f"{name}_{1}_norm")(xx)
    xx = tf.nn.relu(xx)
    # Create layers
    for n, (k, d, f) in enumerate(
        zip(kernel_size[1:], dilation_rate[1:], filters[1:]),
        1
    ):

        xx = tf.keras.layers.Conv1D(
            f,
            k,
            strides=1,
            dilation_rate=d,
            padding="same",
            name=f"{name}_{n+2}",
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=2),
        )(xx)

        xx = tf.keras.layers.BatchNormalization(axis=-1,
                                                name=f"{name}_{n+2}_norm")(xx)
        xx = tf.nn.leaky_relu(xx, alpha=0.1)

    # scale up the skip connection output if the filter sizes are different

    if (x.shape[-1] != filters[-1]) and add_residual:

        x = tf.keras.layers.Conv1D(
            filters[-1],
            1,
            strides=1,
            dilation_rate=1,
            name=f"{name}_skip",
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=2),
        )(x)

        x = tf.keras.layers.BatchNormalization(axis=-1,
                                               name=f"{name}_skip_norm")(x)
        x = tf.nn.leaky_relu(x, alpha=0.1)

    # Add Residue
    if add_residual:
        return tf.keras.layers.Add()([x, xx])
    else:
        return xx


def ConvolutionalTower_g2(x,
                          num_res_blocks=5,
                          add_residual=True):
    """
    Covolutional tower to increase the receptive filed size with dilated
    convolutions
    """

    x = tf.keras.layers.Conv1D(
        128,
        9,
        strides=1,
        dilation_rate=1,
        padding="same",
        name="conv1",
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=2),
    )(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=-1, name="block1_1")(x)
    x = tf.nn.leaky_relu(x, alpha=0.1)

    x = tf.keras.layers.Conv1D(
        128,
        3,
        strides=1,
        dilation_rate=2,
        padding="same",
        name="conv2",
        kernel_initializer=tf.keras.initializers.Orthogonal(gain=2),
    )(x)

    x = tf.keras.layers.BatchNormalization(axis=-1, name="block1_2")(x)
    x = tf.nn.leaky_relu(x, alpha=0.1)

    if num_res_blocks:
        for i, n in enumerate(range(num_res_blocks)):
            x = (
                lambda x, n: resnet_block_g2(
                    x,
                    name=f"block2_{n}",
                    kernel_size=[3, 3],
                    dilation_rate=[3, 3],
                    filters=[256, 256],
                    add_residual=add_residual,
                )
            )(x, n)

    return x


def create_jaeger_model(
    input_shape, vocab_size=22, embedding_size=4, out_shape=6, bias_init=None
):

    inputs = tf.keras.Input(shape=input_shape, name="translated")
    #     embedding_layer = tf.keras.layers.Embedding(vocab_size,
    #                                                 embedding_size,
    #                                                 name="aa-embedding",
    #                                                 mask_zero=True)

    #     x = embedding_layer(inputs)
    x = ConvolutionalTower_g2(inputs, num_res_blocks=10, add_residual=True)
    # A block
    x = SumReduce()(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           name="sum_reduce_norm")(x)
    # create amino acid feature vec
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x batch-norm and dropout do not play nicely together
    # https://doi.org/10.48550/arXiv.1801.05134
    x = tf.keras.layers.Dense(
        32,
        activation=tf.nn.relu,
        name="augdense-1",
        kernel_initializer=tf.keras.initializers.HeNormal(),
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,
                                           name="dense1")(x)
    x = tf.keras.layers.Dense(
        32,
        activation=tf.nn.relu,
        name="augdense-2",
        kernel_initializer=tf.keras.initializers.HeNormal(),
        kernel_regularizer=tf.keras.regularizers.L2(1e-4),
    )(x)
    x1 = tf.keras.layers.BatchNormalization(axis=-1,
                                            name="dense2")(x)
    # x = tf.keras.layers.Dense(dense_c2_nodes,
    # activation=tf.nn.gelu,
    # name='augdense-2',
    # kernel_regularizer=tf.keras.regularizers.L2(1e-4))(x)
    if bias_init is not None:
        bias_init = tf.keras.initializers.Constant(bias_init)
    out = tf.keras.layers.Dense(
        out_shape,
        name="outdense",
        dtype="float32",
        kernel_initializer=tf.keras.initializers.HeNormal(),
        use_bias=True,
        bias_initializer=bias_init,
    )(
        x1
    )  # validation loss jumps when bias is removed

    return inputs, {"output": out, "embedding": x1}
