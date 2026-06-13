"""Tests for jaeger.utils.stats."""

from __future__ import annotations

import numpy as np

from jaeger.utils import stats


class TestSignificantTopClass:
    def test_significant_difference(self):
        rng = np.random.default_rng(0)
        logits1 = rng.normal(2.0, 0.1, size=30)
        logits2 = rng.normal(0.0, 0.1, size=30)
        result = stats.significant_top_class(logits1, logits2, alpha=0.05)
        assert "p_value" in result
        assert "significant" in result
        assert result["significant"]

    def test_no_significant_difference(self):
        rng = np.random.default_rng(0)
        logits1 = rng.normal(0.0, 0.1, size=30)
        logits2 = rng.normal(0.0, 0.1, size=30)
        result = stats.significant_top_class(logits1, logits2, alpha=0.05)
        assert not result["significant"]


class TestWelchT:
    def test_welch_t_one_tailed_greater(self):
        mean1, var1, n1 = 10.0, 1.0, 30
        mean2, var2, n2 = 5.0, 1.0, 30
        t, df, p = stats.welch_t_one_tailed(
            mean1, var1, n1, mean2, var2, n2, alternative="greater"
        )
        assert p < 0.05
        assert t > 0

    def test_welch_t_one_tailed_less(self):
        mean1, var1, n1 = 5.0, 1.0, 30
        mean2, var2, n2 = 10.0, 1.0, 30
        t, df, p = stats.welch_t_one_tailed(
            mean1, var1, n1, mean2, var2, n2, alternative="less"
        )
        assert p < 0.05
        assert t < 0
