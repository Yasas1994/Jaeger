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
