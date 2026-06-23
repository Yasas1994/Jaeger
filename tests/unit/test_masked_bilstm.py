import numpy as np
import tensorflow as tf

from jaeger.nnlib.v2.layers import MaskedBiLSTM


class TestMaskedBiLSTM:
    def test_output_shape_return_sequences(self):
        layer = MaskedBiLSTM(units=32)
        inputs = tf.keras.Input(shape=(6, 100, 16))
        outputs = layer(inputs)
        model = tf.keras.Model(inputs, outputs)
        x = np.random.randn(2, 6, 100, 16).astype(np.float32)
        y = model.predict(x)
        assert y.shape == (2, 6, 100, 64)

    def test_output_shape_no_return_sequences(self):
        layer = MaskedBiLSTM(units=32, return_sequences=False)
        inputs = tf.keras.Input(shape=(6, 100, 16))
        outputs = layer(inputs)
        model = tf.keras.Model(inputs, outputs)
        x = np.random.randn(2, 6, 100, 16).astype(np.float32)
        y = model.predict(x)
        assert y.shape == (2, 6, 64)

    def test_mask_propagation(self):
        layer = MaskedBiLSTM(units=16)
        x = tf.constant(np.random.randn(2, 6, 10, 8).astype(np.float32))
        mask = tf.sequence_mask([8, 5], maxlen=10)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, 6, 1])  # (2, 6, 10)
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 10, 32]

    def test_ignore_mask_allows_cudnn(self):
        # With ignore_mask=True the layer should not pass a mask to the LSTM,
        # so cuDNN can be used even with non-right-padded masks upstream.
        layer = MaskedBiLSTM(units=16, use_cudnn=True, ignore_mask=True)
        x = tf.constant(np.random.randn(2, 6, 10, 8).astype(np.float32))
        mask = tf.sequence_mask([8, 5], maxlen=10)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, 6, 1])
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 10, 32]
        assert layer.compute_mask(x, mask=mask) is None

    def test_get_config(self):
        layer = MaskedBiLSTM(
            units=32,
            dropout=0.1,
            recurrent_dropout=0.2,
            use_cudnn=True,
            ignore_mask=True,
        )
        config = layer.get_config()
        assert config["units"] == 32
        assert config["dropout"] == 0.1
        assert config["recurrent_dropout"] == 0.2
        assert config["return_sequences"] is True
        assert config["use_cudnn"] is True
        assert config["ignore_mask"] is True
