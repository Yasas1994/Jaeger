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


# --- MaskedLastPooling ---------------------------------------------------------


def test_last_pooling_no_mask_takes_last_position():
    from jaeger.nnlib.v2.layers import MaskedLastPooling

    x = tf.reshape(tf.range(2 * 6 * 8, dtype=tf.float32), (2, 6, 8, 1))
    pooled = MaskedLastPooling()(x)
    # last position per frame = indices 7, 15, ..., mean over 6 frames
    last = x[:, :, -1, 0]  # (2, 6)
    expected = tf.reduce_mean(last, axis=1)
    np.testing.assert_allclose(pooled.numpy().flatten(), expected.numpy(), atol=1e-6)


def test_last_pooling_uses_last_valid_position_not_padding():
    from jaeger.nnlib.v2.layers import MaskedLastPooling

    x = tf.reshape(tf.range(1 * 1 * 8, dtype=tf.float32), (1, 1, 8, 1))
    # real content at 0..4, padding at 5..7
    mask = tf.constant([[[True] * 5 + [False] * 3]])
    pooled = MaskedLastPooling()(x, mask=mask)
    np.testing.assert_allclose(pooled.numpy().flatten(), [4.0], atol=1e-6)


def test_last_pooling_fully_masked_sample_is_zero():
    from jaeger.nnlib.v2.layers import MaskedLastPooling

    x = tf.random.normal((2, 6, 16, 8))
    mask = tf.concat(
        [tf.ones((1, 6, 16), tf.bool), tf.zeros((1, 6, 16), tf.bool)], axis=0
    )
    pooled = MaskedLastPooling()(x, mask=mask)
    np.testing.assert_allclose(
        pooled[0].numpy(), tf.reduce_mean(x[0, :, -1, :], axis=0).numpy(), atol=1e-6
    )
    np.testing.assert_allclose(pooled[1].numpy(), np.zeros(8), atol=0.0)


def test_last_pooling_per_frame_independent():
    from jaeger.nnlib.v2.layers import MaskedLastPooling

    # 1 sample, 2 frames with different valid lengths
    x = tf.reshape(tf.range(2 * 8, dtype=tf.float32), (1, 2, 8, 1))
    mask = tf.constant([[[True] * 4 + [False] * 4, [True] * 8]])
    pooled = MaskedLastPooling()(x, mask=mask)
    # frame0 last valid index 3 -> value 3.0; frame1 last valid index 7 -> 15.0
    np.testing.assert_allclose(pooled.numpy().flatten(), [(3.0 + 15.0) / 2], atol=1e-6)


def test_builder_maps_last_pooling():
    from jaeger.nnlib.v2.layers import MaskedLastPooling

    builder = DynamicModelBuilder({"model": {"name": "jaeger"}, "training": {}})
    assert builder._get_pooler("last") is MaskedLastPooling
    assert builder._get_pooler("masked_last") is MaskedLastPooling


def test_last_pooling_mixed_bfloat16():
    from jaeger.nnlib.v2.layers import MaskedLastPooling

    old_policy = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        x = tf.cast(tf.random.normal((2, 6, 16, 8)), tf.bfloat16)
        mask = tf.concat(
            [tf.ones((2, 6, 10), tf.bool), tf.zeros((2, 6, 6), tf.bool)], axis=-1
        )
        pooled = MaskedLastPooling()(x, mask=mask)
        assert pooled.shape == (2, 8)
        assert pooled.dtype == tf.bfloat16
        assert np.all(np.isfinite(pooled.numpy()))
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


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
