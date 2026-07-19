from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from jaeger.nnlib.builder import DynamicModelBuilder
from jaeger.nnlib.v2.layers import (
    MaskedBatchNorm,
    MaskedDYT,
    MaskedLayerNormalization,
    ResidualBlock,
)
from jaeger.utils.misc import load_model_config

_BLOCK = dict(filters=16, kernel_size=5, strides=1, activation="gelu")


def _x(batch=2, strands=6, length=32, channels=16):
    return tf.random.normal((batch, strands, length, channels))


def test_default_norm_is_masked_batchnorm():
    block = ResidualBlock(block_number=0, **_BLOCK)
    assert block.norm_type == "masked_batchnorm"
    block(_x())
    assert isinstance(block.bn1, MaskedBatchNorm)
    assert isinstance(block.bn2, MaskedBatchNorm)


@pytest.mark.parametrize(
    "norm_type,cls",
    [
        ("masked_batchnorm", MaskedBatchNorm),
        ("masked_layernorm", MaskedLayerNormalization),
        ("masked_dyt", MaskedDYT),
    ],
)
def test_norm_type_builds_and_runs(norm_type, cls):
    block = ResidualBlock(block_number=0, norm_type=norm_type, **_BLOCK)
    assert isinstance(block.bn1, cls)
    assert isinstance(block.bn2, cls)
    y = block(_x())
    assert y.shape == (2, 6, 32, 16)
    assert np.all(np.isfinite(y.numpy()))


def test_norm_type_with_strided_bypass():
    # strides > 1 builds the 1x1 bypass conv; all norm types must build bn3
    for norm_type in ("masked_batchnorm", "masked_layernorm", "masked_dyt"):
        block = ResidualBlock(
            block_number=0, norm_type=norm_type, **{**_BLOCK, "strides": 2}
        )
        y = block(_x())
        assert y.shape[2] == 16
        assert np.all(np.isfinite(y.numpy()))


def test_invalid_norm_type_rejected():
    with pytest.raises(ValueError, match="norm_type"):
        ResidualBlock(block_number=0, norm_type="bogus", **_BLOCK)


def test_return_nmd_requires_batchnorm():
    with pytest.raises(ValueError, match="return_nmd"):
        ResidualBlock(
            block_number=0, norm_type="masked_layernorm", return_nmd=True, **_BLOCK
        )


def test_masked_dyt_zeroes_masked_positions():
    # unit-level: MaskedDYT re-zeroes masked positions when given a mask
    dyt = MaskedDYT()
    x = _x(batch=1, strands=1, length=32, channels=8)
    mask = tf.concat([tf.ones((1, 1, 16)), tf.zeros((1, 1, 16))], axis=-1)
    out = dyt(x, mask=tf.cast(mask, tf.bool))
    np.testing.assert_allclose(out.numpy()[:, :, 16:], 0.0, atol=1e-5)


def test_masked_layernorm_masked_equals_truncated():
    block = ResidualBlock(block_number=0, norm_type="masked_layernorm", **_BLOCK)
    x = _x(batch=1, strands=1, length=32)
    mask = tf.concat([tf.ones((1, 1, 16)), tf.zeros((1, 1, 16))], axis=-1)
    y_masked = block(x, mask=tf.cast(mask, tf.bool))
    y_trunc = block(x[:, :, :16])
    # only positions whose conv windows stay fully inside the valid region
    # are identical (k=5 -> positions 0..11); boundary positions legitimately
    # see partial windows
    np.testing.assert_allclose(
        y_masked.numpy()[:, :, :10], y_trunc.numpy()[:, :, :10], atol=1e-4
    )


def test_builder_config_norm_type(tmp_path):
    cfg = load_model_config(Path("train_config/hyena_test.yaml"))
    cfg["model"]["representation_learner"]["hidden_layers"] = [
        {
            "name": "residual_block",
            "config": {
                "use_1x1conv": False,
                "block_size": 1,
                "filters": 32,
                "kernel_size": 5,
                "strides": 1,
                "activation": "gelu",
                "norm_type": "masked_layernorm",
            },
        }
    ]
    cfg["model"]["classifier"]["input_shape"] = 32
    cfg["model"]["projection"]["input_shape"] = 32
    shutil.rmtree(cfg["training"]["data_dir"], ignore_errors=True)
    builder = DynamicModelBuilder(cfg)
    models = builder.build_fragment_classifier()
    rep = models["rep_model"]
    blocks = [
        layer for layer in rep._flatten_layers() if isinstance(layer, ResidualBlock)
    ]
    assert blocks, "no ResidualBlock found in rep model"
    for b in blocks:
        assert b.norm_type == "masked_layernorm"
        assert isinstance(b.bn1, MaskedLayerNormalization)


def test_norm_type_serialization_roundtrip():
    block = ResidualBlock(block_number=0, norm_type="masked_dyt", **_BLOCK)
    restored = ResidualBlock.from_config(block.get_config())
    assert restored.norm_type == "masked_dyt"
