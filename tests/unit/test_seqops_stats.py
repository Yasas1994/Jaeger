"""Tests for jaeger.seqops.stats."""

from __future__ import annotations

import math

import pytest

from jaeger.seqops import stats


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("AAAA", 0.0),
        ("ACGT", 2.0),
        ("AACG", pytest.approx(-(0.5 * math.log2(0.5) + 0.25 * math.log2(0.25) * 2))),
    ],
)
def test_shannon_entropy(seq: str, expected: float):
    assert stats.shannon_entropy(seq) == expected


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("GC", 1.0),
        ("AT", 0.0),
        ("AAAA", 0.0),
    ],
)
def test_calculate_gc_content(seq: str, expected: float):
    assert stats.calculate_gc_content(seq) == expected


@pytest.mark.parametrize(
    "seq, expected",
    [
        ("ACGTNN", 1 / 3),
        ("ACGT", 0.0),
        ("NNNN", 1.0),
    ],
)
def test_calculate_percentage_of_n(seq: str, expected: float):
    assert stats.calculate_percentage_of_n(seq) == pytest.approx(expected)


def test_gc_skew():
    seq = "GGGCTT"
    skew = stats.gc_skew(seq)
    # (G-C)/(G+C) = (3-1)/(3+1) = 0.5
    assert skew == pytest.approx(0.5)
