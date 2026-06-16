"""Tests for short-fragment layers."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2 import layers


class TestMultiScaleConv1D:
    def test_concat_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                {"filters": 16, "kernel_size": 5, "dilation_rate": 1},
                {"filters": 8, "kernel_size": 3, "dilation_rate": 3},
            ],
            merge="concat",
        )
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 32, 32]
        assert np.all(np.isfinite(out.numpy()))

    def test_add_output_shape(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                {"filters": 8, "kernel_size": 5, "dilation_rate": 1},
            ],
            merge="add",
        )
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 32, 8]

    def test_add_mismatched_filters_raises(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3},
                {"filters": 16, "kernel_size": 3},
            ],
            merge="add",
        )
        with pytest.raises(ValueError):
            layer(x, mask=mask)

    def test_mask_propagation(self):
        x = tf.random.normal((2, 6, 32, 4))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(
            mask, [[0, 0, 15], [0, 0, 16]], [False, False]
        )
        layer = layers.MultiScaleConv1D(
            branches=[{"filters": 8, "kernel_size": 3, "padding": "same"}],
            merge="concat",
        )
        _ = layer(x, mask=mask)
        assert layer.compute_mask(x, mask=mask) is not None

    def test_no_mask_works(self):
        x = tf.random.normal((2, 6, 32, 4))
        layer = layers.MultiScaleConv1D(
            branches=[{"filters": 8, "kernel_size": 3, "padding": "same"}],
            merge="concat",
        )
        out = layer(x)
        assert out.shape.as_list() == [2, 6, 32, 8]

    def test_invalid_merge_raises(self):
        with pytest.raises(ValueError):
            layers.MultiScaleConv1D(
                branches=[{"filters": 8, "kernel_size": 3}],
                merge="invalid",
            )

    def test_misaligned_branch_padding_raises(self):
        x = tf.random.normal((2, 6, 32, 4))
        layer = layers.MultiScaleConv1D(
            branches=[{"filters": 8, "kernel_size": 3, "padding": "valid"}],
            merge="concat",
        )
        with pytest.raises(ValueError):
            layer(x)

    def test_get_config_round_trip(self):
        x = tf.random.normal((2, 6, 32, 4))
        layer = layers.MultiScaleConv1D(
            branches=[
                {"filters": 8, "kernel_size": 3, "dilation_rate": 1},
                {"filters": 16, "kernel_size": 5, "dilation_rate": 1},
            ],
            merge="concat",
        )
        original_shape = layer(x).shape.as_list()
        config = layer.get_config()
        restored = layers.MultiScaleConv1D.from_config(config)
        restored_out = restored(x)
        assert restored_out.shape.as_list() == original_shape
        assert restored.merge == layer.merge
        assert restored.branches == layer.branches


class TestMaskedGlobalAvgPooling:
    def test_masked_average_ignores_padding(self):
        x = tf.constant(
            [
                [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]],
                [[[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]]],
            ]
        )  # (2, 1, 3, 2)
        mask = tf.constant(
            [
                [[True, True, False]],
                [[True, False, False]],
            ]
        )  # (2, 1, 3)
        layer = layers.MaskedGlobalAvgPooling()
        out = layer(x, mask=mask)
        expected = tf.constant([[2.0, 3.0], [5.0, 6.0]])
        np.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-5)

    def test_pooler_registered_in_builder(self):
        from jaeger.nnlib.builder import DynamicModelBuilder

        builder = DynamicModelBuilder(
            {"model": {}, "training": {"callbacks": {"directories": []}}}
        )
        pooler = builder._get_pooler("masked_average")
        assert pooler is layers.MaskedGlobalAvgPooling


class TestLocalAttention:
    def test_output_shape(self):
        x = tf.random.normal((2, 6, 64, 16))
        mask = tf.ones((2, 6, 64), dtype=tf.bool)
        layer = layers.LocalAttention(
            embed_dim=16, num_heads=4, feed_forward_dim=32, window_size=16, num_blocks=2
        )
        out = layer(x, mask=mask)
        assert out.shape.as_list() == [2, 6, 64, 16]

    def test_gradient_flow(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.LocalAttention(
            embed_dim=8, num_heads=2, feed_forward_dim=16, window_size=8, num_blocks=1
        )
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = layer(x, mask=mask)
            loss = tf.reduce_mean(out ** 2)
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert np.all(np.isfinite(grads.numpy()))

    def test_invalid_config_raises(self):
        x = tf.random.normal((2, 6, 32, 8))
        mask = tf.ones((2, 6, 32), dtype=tf.bool)
        layer = layers.LocalAttention(
            embed_dim=8, num_heads=3, feed_forward_dim=16, window_size=8
        )
        with pytest.raises(ValueError):
            layer(x, mask=mask)
