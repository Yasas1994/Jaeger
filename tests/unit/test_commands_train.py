import warnings

import pytest
import tensorflow as tf
from click import UsageError

from jaeger.commands.train import _precision_policy_name, _resolve_precision


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
