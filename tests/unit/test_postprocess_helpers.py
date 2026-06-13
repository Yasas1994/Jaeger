"""Tests for jaeger.postprocess.helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jaeger.postprocess import helpers


class TestMathHelpers:
    def test_sigmoid(self):
        assert helpers.sigmoid(0.0) == pytest.approx(0.5)
        assert helpers.sigmoid(100.0) > 0.99

    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = helpers.softmax(x)
        assert np.allclose(out.sum(axis=-1), 1.0)
        assert out[0, 2] > out[0, 0]

    def test_logsumexp(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = helpers.logsumexp(x, axis=-1)
        expected = np.log(np.sum(np.exp(x)))
        assert out[0] == pytest.approx(expected)

    def test_energy_binary_two_logits(self):
        x = np.array([[1.0, 2.0]])
        e = helpers.energy(x, axis=-1)
        expected = -np.log(np.sum(np.exp(x), axis=-1))
        assert np.allclose(e, expected)

    def test_energy_single_logit(self):
        x = np.array([[1.0], [2.0]])
        e = helpers.energy(x)
        stacked = np.concatenate([x, np.zeros_like(x)], axis=-1)
        expected = -np.log(np.sum(np.exp(stacked), axis=-1))
        assert np.allclose(e, expected)

    def test_normalize_2d(self):
        x = np.array([[3.0, 4.0], [0.0, 1.0]])
        n = helpers.normalize(x)
        assert np.allclose(np.mean(n, axis=1), 0.0)
        assert np.allclose(np.std(n, axis=1), 1.0)

    def test_normalize_l2(self):
        x = np.array([[3.0, 4.0], [0.0, 1.0]])
        n = helpers.normalize_l2(x)
        assert np.linalg.norm(n[0]) == pytest.approx(1.0)

    def test_scale_range(self):
        x = np.array([0.0, 5.0, 10.0])
        out = helpers.scale_range(x.copy(), 0.0, 1.0)
        assert out[0] == pytest.approx(0.0)
        assert out[-1] == pytest.approx(1.0)


class TestEntropyHelpers:
    def test_shanon_entropy(self):
        # For a single probability p, returns -p*log2(p).
        assert helpers.shanon_entropy(0.5) == pytest.approx(0.5)

    def test_binary_entropy(self):
        assert helpers.binary_entropy(0.5) == pytest.approx(1.0)

    def test_softmax_entropy(self):
        p = np.array([[0.5, 0.25, 0.25]])
        e = helpers.softmax_entropy(p)
        assert 0.0 < e[0] <= math.log2(3)


class TestRunHelpers:
    def test_find_runs(self):
        x = np.array([1, 1, 2, 2, 2, 3])
        values, lengths, starts = helpers.find_runs(x)
        assert values.tolist() == [1, 2, 3]
        assert lengths.tolist() == [2, 3, 1]
        assert starts.tolist() == [0, 2, 5]

    def test_update_dict(self):
        x = np.array([0, 1, 2, 0, 1])
        d = helpers.update_dict(np.unique(x, return_counts=True), num_classes=3)
        assert sum(d.values()) == len(x)

    def test_get_window_summary(self):
        x = np.array([0, 0, 1, 1, 2])
        s = helpers.get_window_summary(
            x, class_map={0: "A", 1: "B", 2: "C"}, classes=["A", "B", "C"]
        )
        assert isinstance(s, str)

    def test_merge_overlapping_ranges(self):
        ranges = [[1, 5], [4, 10], [12, 15]]
        merged = helpers.merge_overlapping_ranges(ranges)
        assert merged == [[1, 10], [12, 15]]

    def test_consecutive(self):
        x = np.array([1, 2, 3, 5, 6, 10])
        groups = helpers.consecutive(x)
        assert len(groups) == 3


class TestSequenceHelpers:
    def test_calculate_gc_content(self):
        assert helpers.calculate_gc_content("GCAT") == 0.5

    def test_calculate_percentage_of_n(self):
        assert helpers.calculate_percentage_of_n("ACNN") == 0.5

    def test_gc_skew(self):
        result = helpers.gc_skew("GGCCTT", window=6)
        assert isinstance(result, dict)
