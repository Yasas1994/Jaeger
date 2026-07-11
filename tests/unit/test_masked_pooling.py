from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
tf = pytest.importorskip("tensorflow")

from jaeger.nnlib.v2.layers import MaskedGlobalMaxPooling  # noqa: E402


def test_no_mask_matches_reduce_max():
    rng = np.random.default_rng(0)
    x = tf.constant(rng.normal(size=(2, 6, 10, 4)).astype("float32"))
    layer = MaskedGlobalMaxPooling()
    y = layer(x).numpy()
    ref = tf.reduce_max(x, axis=[1, 2]).numpy()
    np.testing.assert_allclose(y, ref, atol=1e-6)


def test_masked_positions_excluded_from_max():
    # positions 0 and 3 are masked out and hold huge values that must not win.
    vals = np.array([[[[100.0], [1.0], [3.0], [200.0]]]], dtype="float32")  # (1,1,4,1)
    mask = tf.constant([[[0.0, 1.0, 1.0, 0.0]]])  # (1,1,4)
    layer = MaskedGlobalMaxPooling()
    y = layer(tf.constant(vals), mask=mask).numpy()
    assert y.shape == (1, 1)
    np.testing.assert_allclose(y, [[3.0]], atol=1e-6)


def test_supports_masking_flag():
    assert MaskedGlobalMaxPooling().supports_masking is True
