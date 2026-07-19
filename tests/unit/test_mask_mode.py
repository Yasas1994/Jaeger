from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.v2.layers import MaskedConv1D


def _run_mask_mode(mask, mode, kernel_size=5, strides=1):
    """Run a single MaskedConv1D over ones and return its output mask.

    Output position i covers the input window [i, i+k-1] (valid padding).
    """
    length = mask.shape[-1]
    x = tf.ones((1, 1, length, 4))
    conv = MaskedConv1D(
        filters=4, kernel_size=kernel_size, strides=strides, mask_mode=mode
    )
    _ = conv(x, mask=mask)
    return conv.compute_mask(x, mask=mask).numpy().reshape(-1)


def _mask_with_n(length: int, n_pos: list[int]) -> tf.Tensor:
    m = np.ones((1, 1, length), dtype=bool)
    for p in n_pos:
        m[0, 0, p] = False
    return tf.constant(m)


def test_default_mask_mode_is_any():
    conv = MaskedConv1D(filters=4, kernel_size=5)
    assert conv.mask_mode == "any"


def test_invalid_mask_mode_rejected():
    with pytest.raises(ValueError, match="mask_mode"):
        MaskedConv1D(filters=4, kernel_size=5, mask_mode="bogus")


def test_isolated_n_any_kills_nothing():
    # every 5-window containing the N also contains valid inputs
    mask = _mask_with_n(20, [10])
    out = _run_mask_mode(mask, "any")
    assert out.all()


def test_isolated_n_strict_kills_kernel_window():
    mask = _mask_with_n(20, [10])
    out = _run_mask_mode(mask, "strict")
    # output starts 6..10 have windows [i, i+4] covering position 10
    expected = np.ones(16, dtype=bool)
    expected[6:11] = False
    np.testing.assert_array_equal(out, expected)


def test_isolated_n_majority_survives():
    mask = _mask_with_n(20, [10])
    out = _run_mask_mode(mask, "majority")
    assert out.all()


def test_short_n_run_any_kills_nothing():
    # run of 3 < k=5: no all-N window exists
    mask = _mask_with_n(20, [9, 10, 11])
    out = _run_mask_mode(mask, "any")
    assert out.all()


def test_long_n_run_any_kills_only_the_run():
    # run of 5 = k: exactly one all-N window (start 9)
    mask = _mask_with_n(20, [9, 10, 11, 12, 13])
    out = _run_mask_mode(mask, "any")
    expected = np.ones(16, dtype=bool)
    expected[9] = False
    np.testing.assert_array_equal(out, expected)


def test_long_n_run_strict_kills_run_plus_halo():
    mask = _mask_with_n(20, [9, 10, 11, 12, 13])
    out = _run_mask_mode(mask, "strict")
    # starts 5..13 have windows overlapping the run
    expected = np.ones(16, dtype=bool)
    expected[5:14] = False
    np.testing.assert_array_equal(out, expected)


def test_right_padding_any_keeps_real_content():
    # 10 real positions right-padded to 20
    mask = _mask_with_n(20, list(range(10, 20)))
    out_any = _run_mask_mode(mask, "any")
    out_strict = _run_mask_mode(mask, "strict")
    # any: all 10 real positions survive; only all-pad windows die
    assert out_any[:10].all()
    assert not out_any[10:].any()
    # strict: real positions 6-9 die too (their windows touch padding)
    assert out_strict[:6].all()
    assert not out_strict[6:].any()


def test_mask_mode_serialization_roundtrip():
    for mode in ("strict", "majority", "any"):
        conv = MaskedConv1D(filters=4, kernel_size=5, mask_mode=mode)
        restored = MaskedConv1D.from_config(conv.get_config())
        assert restored.mask_mode == mode
