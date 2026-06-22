import warnings

import pytest
import tensorflow as tf
from click import UsageError

from jaeger.commands.train import (
    _precision_policy_name,
    _resolve_batch_size,
    _resolve_precision,
)


class TestResolvePrecision:
    def test_defaults_to_fp32(self):
        assert _resolve_precision("fp32", False) == "fp32"

    def test_mixed_precision_alias_returns_fp16(self):
        with pytest.warns(DeprecationWarning):
            assert _resolve_precision("fp32", True) == "fp16"

    def test_mixed_precision_conflict_raises(self):
        with pytest.raises(UsageError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _resolve_precision("bf16", True)


class TestPrecisionPolicyName:
    def test_mapping(self):
        assert _precision_policy_name("fp16") == "mixed_float16"
        assert _precision_policy_name("bf16") == "mixed_bfloat16"
        assert _precision_policy_name("fp32") is None

    def test_policy_fp16_sets(self):
        old = tf.keras.mixed_precision.global_policy().name
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            assert tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        finally:
            tf.keras.mixed_precision.set_global_policy(old)

    def test_policy_bf16_sets(self):
        old = tf.keras.mixed_precision.global_policy().name
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            assert tf.keras.mixed_precision.global_policy().name == "mixed_bfloat16"
        finally:
            tf.keras.mixed_precision.set_global_policy(old)


class TestResolveBatchSize:
    def test_global_fallback(self):
        assert _resolve_batch_size({"batch_size": 64}, "classifier") == 64
        assert _resolve_batch_size({"batch_size": 64}, "projection") == 64
        assert _resolve_batch_size({"batch_size": 64}, "reliability") == 64

    def test_branch_specific_override(self):
        cfg = {
            "batch_size": 64,
            "classifier_batch_size": 128,
            "projection_batch_size": 32,
            "reliability_batch_size": 256,
        }
        assert _resolve_batch_size(cfg, "classifier") == 128
        assert _resolve_batch_size(cfg, "projection") == 32
        assert _resolve_batch_size(cfg, "reliability") == 256

    def test_partial_override_uses_global(self):
        cfg = {"batch_size": 64, "classifier_batch_size": 128}
        assert _resolve_batch_size(cfg, "classifier") == 128
        assert _resolve_batch_size(cfg, "projection") == 64
