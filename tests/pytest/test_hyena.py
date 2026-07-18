from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.nnlib.v2.layers import (
    HyenaBlock,
    HyenaFilter,
    HyenaOperator,
    causal_fft_convolve,
)
from jaeger.utils.misc import load_model_config


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


def test_hyena_operator_shape():
    layer = HyenaOperator(dim=8, seq_len=32, order=2)
    x = tf.random.normal((2, 32, 8))  # (batch, length, dim)
    y = layer(x)
    assert y.shape.as_list() == [2, 32, 8]


def test_hyena_block_4d_shape():
    layer = HyenaBlock(dim=8, seq_len=32, order=2)
    x = tf.random.normal((2, 6, 32, 8))  # (batch, strands, length, dim)
    y = layer(x)
    assert y.shape.as_list() == [2, 6, 32, 8]


def test_hyena_block_mixed_bfloat16():
    old_policy = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        layer = HyenaBlock(dim=32, seq_len=64, order=2)
        x = tf.cast(tf.random.normal((2, 6, 64, 32)), tf.bfloat16)
        y = layer(x)
        assert y.shape.as_list() == [2, 6, 64, 32]
        assert y.dtype == tf.bfloat16
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


def test_hyena_operator_dynamic_length():
    layer = HyenaOperator(dim=8, seq_len=None, order=2)
    x = tf.random.normal((2, 24, 8))
    y = layer(x)
    assert y.shape.as_list() == [2, 24, 8]


def test_hyena_block_dynamic_length():
    layer = HyenaBlock(dim=8, seq_len=None, order=2)
    x = tf.random.normal((2, 6, 24, 8))
    y = layer(x)
    assert y.shape.as_list() == [2, 6, 24, 8]


def test_hyena_block_symbolic_model():
    inputs = tf.keras.Input(shape=(6, 32, 8))
    x = HyenaBlock(dim=8, seq_len=32, order=2)(inputs)
    model = tf.keras.Model(inputs, x)
    out = model(tf.random.normal((2, 6, 32, 8)))
    assert out.shape.as_list() == [2, 6, 32, 8]


def test_hyena_block_per_strand_independence():
    layer = HyenaBlock(dim=8, seq_len=16, order=2)
    single = tf.random.normal((1, 1, 16, 8))
    layer(single)

    stacked = tf.concat([single, single], axis=1)  # (1, 2, 16, 8)
    stacked_out = layer(stacked)

    np.testing.assert_allclose(
        stacked_out[0, 0].numpy(), stacked_out[0, 1].numpy(), atol=1e-5
    )


def test_hyena_block_respects_mask():
    layer = HyenaBlock(dim=8, seq_len=16, order=2)
    x = tf.random.normal((1, 1, 16, 8))
    # mask out the second half
    mask = tf.concat([tf.ones((1, 1, 8)), tf.zeros((1, 1, 8))], axis=-1)
    out = layer(x, mask=mask)
    np.testing.assert_allclose(out[0, 0, 8:].numpy(), np.zeros((8, 8)), atol=1e-5)


def test_hyena_block_masked_equals_truncated():
    """Valid-position outputs must be identical to running on the unpadded
    prefix (causal conv + re-zeroed padded positions)."""
    layer = HyenaBlock(dim=8, seq_len=None, order=2)
    x = tf.random.normal((2, 6, 32, 8))
    mask = tf.concat([tf.ones((2, 6, 20)), tf.zeros((2, 6, 12))], axis=-1)
    y_masked = layer(x, mask=tf.cast(mask, tf.bool), training=False)
    y_trunc = layer(x[:, :, :20], training=False)
    np.testing.assert_allclose(y_masked[:, :, :20].numpy(), y_trunc.numpy(), atol=1e-4)


def test_builder_creates_hyena_model():
    cfg_path = Path("train_config/hyena_test.yaml")
    cfg = load_model_config(cfg_path)
    shutil.rmtree(cfg["training"]["data_dir"], ignore_errors=True)
    builder = DynamicModelBuilder(cfg)
    models = builder.build_fragment_classifier()
    assert "rep_model" in models
    assert "jaeger_classifier" in models


def test_hyena_filter_alphas_loguniform_init():
    layer = HyenaFilter(seq_len=32, dim=64, order=2)
    layer.build((None, 64, 32))
    alphas = layer.alphas.numpy()
    assert alphas.min() >= 1e-3
    assert alphas.max() <= 1.0
    # log-uniform over 3 decades should span short and long filters
    assert alphas.min() < 1e-2
    assert alphas.max() > 1e-1


def test_hyena_filter_negative_alphas_stay_finite():
    """A negative drift of decay rates must not explode (exp(+|a|t) NaN)."""
    layer = HyenaFilter(seq_len=256, dim=8, order=2)
    layer.build((None, 8, 256))
    layer.alphas.assign(tf.fill((2, 8), -5.0))
    filters = layer()
    assert np.all(np.isfinite(filters.numpy()))


def test_hyena_filter_siren_activation():
    layer = HyenaFilter(seq_len=16, dim=8, order=2, activation="sin")
    layer.build((None, 8, 16))
    hidden_dense = layer.ffns[0].layers[0]
    assert hidden_dense.activation is tf.sin
    filters = layer()
    assert np.all(np.isfinite(filters.numpy()))


def test_hyena_block_output_projection_opt_in():
    block = HyenaBlock(dim=8, seq_len=16, order=2, output_projection=True)
    x = tf.random.normal((2, 6, 16, 8))
    y = block(x)
    assert y.shape.as_list() == [2, 6, 16, 8]
    assert block.out_proj is not None
    weight_names = [getattr(w, "path", w.name) for w in block.weights]
    assert any("out_proj" in n for n in weight_names)

    default_block = HyenaBlock(dim=8, seq_len=16, order=2)
    default_block(x)
    assert default_block.out_proj is None


def test_hyena_block_output_projection_serialization_roundtrip():
    block = HyenaBlock(dim=8, seq_len=16, order=2, output_projection=True)
    x = tf.random.normal((2, 6, 16, 8))
    block(x)
    weights = block.get_weights()

    restored = HyenaBlock.from_config(block.get_config())
    restored(x)
    restored.set_weights(weights)
    np.testing.assert_allclose(
        block(x, training=False).numpy(),
        restored(x, training=False).numpy(),
        atol=1e-5,
    )


def test_hyena_filter_normalize_unit_l2():
    layer = HyenaFilter(seq_len=64, dim=8, order=2, normalize=True)
    layer.build((None, 8, 64))
    filters = layer()  # (order, dim, L)
    norms = tf.linalg.norm(filters, axis=-1).numpy()  # (order, dim)
    np.testing.assert_allclose(norms, np.ones((2, 8)), atol=1e-5)


def test_hyena_block_filter_normalize_preserves_magnitude():
    """Normalized filters keep the block output at input scale; unnormalized
    long/flat filters amplify it (loss explosion seen at init)."""
    tf.random.set_seed(0)
    x = tf.random.normal((2, 6, 666, 64))
    normed = HyenaBlock(dim=64, order=2, filter_normalize=True)
    plain = HyenaBlock(dim=64, order=2, filter_normalize=False)
    y_normed = normed(x, training=False)
    y_plain = plain(x, training=False)
    in_std = float(tf.math.reduce_std(x))
    out_std_normed = float(tf.math.reduce_std(y_normed))
    out_std_plain = float(tf.math.reduce_std(y_plain))
    # normed: op output ~ input scale, so block (op + residual) stays bounded
    assert out_std_normed < 2.5 * in_std
    # unnormalized long/flat filters always amplify more than normalized ones
    assert out_std_plain > out_std_normed


def test_hyena_block_filter_normalize_tail_does_not_leak():
    """With a fixed (padded) length, tail content must not change valid
    outputs: the causal conv only looks backward. This is the invariant that
    matters for the padded-batch path — per-tap filter scale is normalized
    over the runtime length, so padded==truncated does NOT hold exactly."""
    layer = HyenaBlock(dim=8, seq_len=None, order=2, filter_normalize=True)
    x = tf.random.normal((2, 6, 32, 8))
    mask = tf.concat([tf.ones((2, 6, 20)), tf.zeros((2, 6, 12))], axis=-1)
    mask = tf.cast(mask, tf.bool)
    y1 = layer(x, mask=mask, training=False)
    # same valid prefix, garbage tail
    x2 = tf.concat([x[:, :, :20], tf.random.normal((2, 6, 12, 8)) * 100], axis=2)
    y2 = layer(x2, mask=mask, training=False)
    np.testing.assert_allclose(y1[:, :, :20].numpy(), y2[:, :, :20].numpy(), atol=1e-4)
    # padded positions are zeroed in both cases
    np.testing.assert_allclose(
        y1[:, :, 20:].numpy(), np.zeros((2, 6, 12, 8)), atol=1e-5
    )


def test_hyena_block_kernel_regularizer_adds_losses():
    block = HyenaBlock(
        dim=8,
        seq_len=16,
        order=2,
        output_projection=True,
        kernel_regularizer=tf.keras.regularizers.L2(1e-5),
    )
    x = tf.random.normal((2, 6, 16, 8))
    _ = block(x)
    assert len(block.losses) > 0
    # L2(1e-5) on fresh weights is tiny
    total = float(tf.add_n(block.losses))
    assert 0 < total < 1.0

    plain = HyenaBlock(dim=8, seq_len=16, order=2)
    _ = plain(x)
    assert len(plain.losses) == 0


def test_hyena_block_kernel_regularizer_serialization_roundtrip():
    block = HyenaBlock(dim=8, seq_len=16, order=2, kernel_regularizer="l2")
    restored = HyenaBlock.from_config(block.get_config())
    assert isinstance(restored.kernel_regularizer, tf.keras.regularizers.L2)


def test_builder_hyena_block_regularizer_from_yaml(tmp_path):
    """Config keys kernel_regularizer/kernel_regularizer_w on hyena_block must
    be converted to a regularizer instance by the builder."""
    cfg = load_model_config(Path("train_config/hyena_test.yaml"))
    for layer in cfg["model"]["representation_learner"]["hidden_layers"]:
        if layer["name"] == "hyena_block":
            layer["config"]["kernel_regularizer"] = "l2"
            layer["config"]["kernel_regularizer_w"] = 1e-5
    import shutil

    shutil.rmtree(cfg["training"]["data_dir"], ignore_errors=True)
    builder = DynamicModelBuilder(cfg)
    models = builder.build_fragment_classifier()
    assert "rep_model" in models
    rep = models["rep_model"]
    assert any(isinstance(layer, HyenaBlock) for layer in rep.layers)
    hyena = next(layer for layer in rep.layers if isinstance(layer, HyenaBlock))
    assert isinstance(hyena.kernel_regularizer, tf.keras.regularizers.L2)
    _ = rep(
        tf.constant(np.random.randint(1, 33, size=(2, 6, 666)), dtype=tf.int32),
        training=False,
    )
    assert len(hyena.losses) > 0
