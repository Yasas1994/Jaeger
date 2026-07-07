from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2.layers import HyenaFilter, causal_fft_convolve


def test_hyena_filter_shape():
    layer = HyenaFilter(seq_len=32, dim=8, order=2)
    filters = layer()
    assert filters.shape.as_list() == [2, 8, 32]


def test_hyena_filter_dynamic_length():
    layer = HyenaFilter(seq_len=None, dim=8, order=2)
    filters = layer(seq_len=16)
    assert filters.shape.as_list() == [2, 8, 16]


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


def test_hyena_filter_tracks_weights():
    layer = HyenaFilter(seq_len=16, dim=8, order=2)
    layer.build((None, 8, 16))
    weight_names = [getattr(w, "path", w.name) for w in layer.weights]
    assert any("alphas" in n for n in weight_names)
    assert any("biases" in n for n in weight_names)
    assert any("ffn_0" in n for n in weight_names)
    assert any("pos_encoding" in n for n in weight_names)


def test_hyena_filter_serialization_roundtrip():
    layer = HyenaFilter(
        seq_len=16, dim=8, order=2, pe_dim=8, hidden_dim=16, num_layers=2
    )
    layer.build((None, 8, 16))
    weights = layer.get_weights()

    config = layer.get_config()
    restored = HyenaFilter.from_config(config)
    restored.build((None, 8, 16))
    restored.set_weights(weights)

    out1 = layer()
    out2 = restored()
    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)
