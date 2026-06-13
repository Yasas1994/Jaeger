"""Statistical utilities for Jaeger.

One-tailed t-tests and Welch's t-test helpers used for comparing
class logits and confidence scores.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats


def significant_top_class(logits_class1, logits_class2, alpha=0.05):
    """One-tailed paired t-test: are top-1 logits significantly higher than top-2?

    Parameters
    ----------
    logits_class1, logits_class2:
        Arrays of per-window logits for the top two classes.
    alpha:
        Significance threshold (default 0.05).

    Returns
    -------
    dict with keys ``t_stat``, ``p_value``, ``significant``.
    """
    diffs = np.array(logits_class1) - np.array(logits_class2)
    t_stat, p_two_tailed = stats.ttest_1samp(diffs, 0)
    p_one_tailed = p_two_tailed / 2 if t_stat > 0 else 1 - (p_two_tailed / 2)
    significant = p_one_tailed < alpha
    return {"t_stat": t_stat, "p_value": p_one_tailed, "significant": significant}


def welch_t_one_tailed(mean1, var1, n1, mean2, var2, n2, alternative="greater"):
    """One-tailed Welch's t-test using summary statistics.

    Parameters
    ----------
    mean1, var1, n1:
        Mean, variance, and sample size for group 1.
    mean2, var2, n2:
        Mean, variance, and sample size for group 2.
    alternative:
        ``"greater"`` tests mean1 > mean2, ``"less"`` tests mean1 < mean2.

    Returns
    -------
    ``(t_stat, df, p_value)``
    """
    se = sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean1 - mean2) / se

    df_num = (var1 / n1 + var2 / n2) ** 2
    df_denom = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
    df = df_num / df_denom

    if alternative == "greater":
        p = 1 - stats.t.cdf(t_stat, df)
    elif alternative == "less":
        p = stats.t.cdf(t_stat, df)
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    return t_stat, df, p
