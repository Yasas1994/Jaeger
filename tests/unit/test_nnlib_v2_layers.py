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


class TestMaskedBatchNorm:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MaskedBatchNorm()
        out = layer(x, mask=mask)
        assert out.shape == x.shape


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


class TestPositionEmbedding:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 32, 8))
        layer = layers.SinusoidalPositionEmbedding(max_wavelength=1000)
        out = layer(x)
        assert out.shape == x.shape
