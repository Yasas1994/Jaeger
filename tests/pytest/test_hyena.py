from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2.layers import causal_fft_convolve


def test_causal_fft_convolve_shape():
    u = tf.random.normal((2, 4, 16))  # (batch, dim, L)
    h = tf.random.normal((4, 16))  # (dim, L)
    y = causal_fft_convolve(u, h)
    assert y.shape.as_list() == [2, 4, 16]


def test_causal_fft_convolve_preserves_dtype():
    for dtype in (tf.float16, tf.bfloat16, tf.float32, tf.float64):
        u = tf.cast(tf.random.normal((2, 4, 16)), dtype)
        h = tf.random.normal((4, 16))
        y = causal_fft_convolve(u, h)
        assert y.dtype == dtype


def test_causal_fft_convolve_causality():
    """A causal filter with all mass at t=0 should not shift the input."""
    L = 8
    u = tf.random.normal((1, 1, L))
    h = tf.one_hot(0, L, dtype=tf.float32)[None, :]  # (1, L)
    y = causal_fft_convolve(u, h)
    np.testing.assert_allclose(y.numpy(), u.numpy(), atol=1e-5)


def test_causal_fft_convolve_validates_shape():
    u = tf.random.normal((2, 4, 16))
    h = tf.random.normal((3, 16))  # mismatched dim
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        _ = causal_fft_convolve(u, h)
