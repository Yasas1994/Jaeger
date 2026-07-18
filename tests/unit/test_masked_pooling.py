from __future__ import annotations

import numpy as np
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.nnlib.v2.layers import (
    MaskedBatchNorm,
    MaskedGlobalAvgPooling,
    MaskedGlobalMaxPooling,
)


def _padded_pair(batch=2, strands=6, length=32, dim=8, valid=20, seed=0):
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.normal(size=(batch, strands, length, dim)), tf.float32)
    mask = tf.concat(
        [
            tf.ones((batch, strands, valid)),
            tf.zeros((batch, strands, length - valid)),
        ],
        axis=-1,
    )
    return x, tf.cast(mask, tf.bool), valid


# --- MaskedGlobalMaxPooling ----------------------------------------------------


def test_masked_max_pooling_matches_truncated_max():
    x, mask, valid = _padded_pair()
    pooled = MaskedGlobalMaxPooling()(x, mask=mask)
    expected = tf.reduce_max(x[:, :, :valid], axis=(1, 2))
    np.testing.assert_allclose(pooled.numpy(), expected.numpy(), atol=1e-6)


def test_masked_max_pooling_excludes_padded_constants():
    """Padded positions with large values must not win the max."""
    x, mask, valid = _padded_pair()
    x = x * 0.01  # small valid values
    x = tf.where(
        tf.cast(mask[..., None], tf.float32) > 0, x, tf.constant(1e6, tf.float32)
    )
    pooled = MaskedGlobalMaxPooling()(x, mask=mask)
    expected = tf.reduce_max(x[:, :, :valid], axis=(1, 2))
    np.testing.assert_allclose(pooled.numpy(), expected.numpy(), atol=1e-6)
    # the stock pooler gets this wrong
    stock = tf.keras.layers.GlobalMaxPooling2D()(x)
    assert float(tf.reduce_max(tf.abs(stock - expected))) > 1e3


def test_fully_masked_sample_pools_to_zero_not_sentinel():
    """A sample whose mask is entirely zero must produce a zero vector, not
    the -1e9 sentinel (which explodes downstream logits/loss)."""
    x = tf.random.normal((2, 6, 16, 8))
    mask = tf.concat(
        [tf.ones((1, 6, 16), tf.bool), tf.zeros((1, 6, 16), tf.bool)], axis=0
    )
    pooled = MaskedGlobalMaxPooling()(x, mask=mask)
    assert pooled.shape == (2, 8)
    # valid sample: plain max over everything
    np.testing.assert_allclose(
        pooled[0].numpy(), tf.reduce_max(x[0], axis=(0, 1)).numpy(), atol=1e-6
    )
    # fully-masked sample: zeros, not -1e9
    np.testing.assert_allclose(pooled[1].numpy(), np.zeros(8), atol=0.0)


def test_masked_max_pooling_no_mask_matches_stock():
    x, _, _ = _padded_pair()
    masked = MaskedGlobalMaxPooling()(x)
    stock = tf.keras.layers.GlobalMaxPooling2D()(x)
    np.testing.assert_allclose(masked.numpy(), stock.numpy(), atol=1e-6)


def test_supports_masking_flag():
    assert MaskedGlobalMaxPooling().supports_masking is True
    assert MaskedGlobalAvgPooling().supports_masking is True


# --- MaskedGlobalAvgPooling ----------------------------------------------------


def test_masked_avg_pooling_matches_masked_mean():
    x, mask, valid = _padded_pair()
    pooled = MaskedGlobalAvgPooling()(x, mask=mask)
    expected = tf.reduce_mean(x[:, :, :valid], axis=(1, 2))
    np.testing.assert_allclose(pooled.numpy(), expected.numpy(), atol=1e-6)


def test_masked_avg_pooling_no_mask_matches_stock():
    x, _, _ = _padded_pair()
    masked = MaskedGlobalAvgPooling()(x)
    stock = tf.keras.layers.GlobalAveragePooling2D()(x)
    np.testing.assert_allclose(masked.numpy(), stock.numpy(), atol=1e-6)


# --- builder mapping -----------------------------------------------------------


def test_builder_default_poolers_are_masked():
    builder = DynamicModelBuilder({"model": {"name": "jaeger"}, "training": {}})
    assert builder._get_pooler("max") is MaskedGlobalMaxPooling
    assert builder._get_pooler("average") is MaskedGlobalAvgPooling
    assert builder._get_pooler("masked_max") is MaskedGlobalMaxPooling


# --- end-to-end: embedding -> MaskedBatchNorm -> max pool ----------------------


def test_padded_batch_pooling_matches_truncated():
    """Embedding(mask_zero) + inference-mode BN + masked max pool must give the
    same representation for a padded batch as for the truncated input — the
    exact situation in the two-pass short-contig inference path."""
    dim, vocab, valid, length = 8, 33, 20, 32
    ids = tf.keras.Input(shape=(6, None), dtype=tf.int32)
    emb = tf.keras.layers.Embedding(vocab, dim, mask_zero=True)(ids)
    bn = MaskedBatchNorm()
    x = bn(emb)
    pooled = MaskedGlobalMaxPooling()(x)
    model = tf.keras.Model(ids, pooled)

    rng = np.random.default_rng(1)
    full = tf.constant(rng.integers(1, vocab, size=(2, 6, length)), dtype=tf.int32)
    padded = tf.concat(
        [full[:, :, :valid], tf.zeros((2, 6, length - valid), tf.int32)], axis=-1
    )
    # make BN's moving stats non-trivial so padded zeros become constants
    bn.moving_mean.assign(tf.fill((dim,), 3.0))
    bn.moving_variance.assign(tf.fill((dim,), 2.0))

    out_padded = model(padded, training=False)
    out_trunc = model(full[:, :, :valid], training=False)
    np.testing.assert_allclose(out_padded.numpy(), out_trunc.numpy(), atol=1e-5)
