"""Tests for jaeger.nnlib.v2.layers."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2 import layers


class TestActivations:
    def test_gelu_shape(self):
        x = tf.constant([[1.0, -1.0, 0.0]])
        out = layers.GeLU()(x)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out.numpy()))

    def test_relu_shape(self):
        x = tf.constant([[1.0, -1.0, 0.0]])
        out = layers.ReLU()(x)
        assert out.shape == x.shape
        assert np.all(out.numpy() >= 0)


class TestMaskedConv1D:
    def test_output_shape(self):
        # Input shape: (batch, frames, length, channels)
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MaskedConv1D(filters=8, kernel_size=3, padding="same")
        out = layer(x, mask=mask)
        assert out.shape.as_list()[:3] == [2, 6, 32]
        assert out.shape.as_list()[-1] == 8


class TestResidualBlock:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.ResidualBlock(filters=4, kernel_size=3, block_number=0)
        out = layer(x, mask=mask)
        assert out.shape == x.shape

    def test_residual_block_masked_dyt_builds(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.ResidualBlock(
            filters=4, kernel_size=3, block_number=0, norm_type="masked_dyt"
        )
        out = layer(x, mask=mask)
        assert out.shape == x.shape

    def test_residual_block_return_nmd_requires_masked_batchnorm(self):
        with pytest.raises(ValueError, match="return_nmd=True is only supported"):
            layers.ResidualBlock(
                filters=4,
                kernel_size=3,
                block_number=0,
                norm_type="masked_dyt",
                return_nmd=True,
            )

    @pytest.mark.parametrize(
        "norm_type", ["masked_batchnorm", "masked_layernorm", "masked_dyt"]
    )
    def test_residual_block_various_norm_types(self, norm_type):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.ResidualBlock(
            filters=4, kernel_size=3, block_number=0, norm_type=norm_type
        )
        out = layer(x, mask=mask)
        assert out.shape == x.shape


class TestMaskedBatchNorm:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MaskedBatchNorm()
        out = layer(x, mask=mask)
        assert out.shape == x.shape


class TestMaskedLayerNormalization:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MaskedLayerNormalization()
        out = layer(x, mask=mask)
        assert out.shape == x.shape

    def test_masked_positions_zero_and_unmasked_normalized(self):
        x = tf.random.normal((2, 4, 8, 16), stddev=5.0)
        mask = tf.ones((2, 4, 8), dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(mask, [[0, 0, 0], [1, 2, 3]], [False, False])

        layer = layers.MaskedLayerNormalization(epsilon=1e-3)
        out = layer(x, mask=mask)

        masked_values = tf.boolean_mask(out, ~mask)
        unmasked_values = tf.boolean_mask(out, mask)

        assert tf.reduce_max(tf.abs(masked_values)) < 1e-5
        assert abs(float(tf.reduce_mean(unmasked_values))) < 0.05
        assert abs(float(tf.math.reduce_std(unmasked_values)) - 1.0) < 0.05

    def test_large_values_mixed_precision(self):
        old_policy = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            x = tf.random.normal((2, 6, 100, 128), stddev=50.0, dtype=tf.float16)
            mask = tf.ones((2, 6, 100), dtype=tf.bool)
            layer = layers.MaskedLayerNormalization()
            with tf.GradientTape() as tape:
                tape.watch(x)
                out = layer(x, mask=mask)
            grads = tape.gradient(out, x)
            assert out.dtype == tf.float16
            assert np.all(np.isfinite(out.numpy()))
            assert grads is not None
            assert np.all(np.isfinite(grads.numpy()))
        finally:
            tf.keras.mixed_precision.set_global_policy(old_policy)


class TestMaskedGlobalAvgPooling:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MaskedGlobalAvgPooling()
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 4]


class TestGatedFrameGlobalMaxPooling:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.GatedFrameGlobalMaxPooling(return_gate=False)
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 4]

    def test_return_gate(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.GatedFrameGlobalMaxPooling(return_gate=True)
        out, gate = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 4]
        assert gate.shape.as_list() == [2, 6]


class TestTransformerEncoder:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.TransformerEncoder(embed_dim=8, num_heads=2, feed_forward_dim=16)
        out = layer(x, mask=mask)
        assert out.shape == x.shape


class TestAxialAttention:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 8))
        layer = layers.AxialAttention(embed_dim=8, num_heads=2, feed_forward_dim=16)
        out = layer(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize(
        "norm_type",
        ["layernorm", "masked_layernorm", "masked_dyt", "masked_batchnorm"],
    )
    def test_output_shape_with_norm_type(self, norm_type):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.AxialAttention(
            embed_dim=8,
            num_heads=2,
            feed_forward_dim=16,
            norm_type=norm_type,
        )
        out = layer(x, mask=mask)
        assert out.shape == x.shape


class TestPositionEmbedding:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 8))
        layer = layers.SinusoidalPositionEmbedding(max_wavelength=1000)
        out = layer(x)
        assert out.shape == x.shape
