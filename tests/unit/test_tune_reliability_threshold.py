"""Tests for scripts.tune_reliability_threshold."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import tune_reliability_threshold as tuner


class TestTuneThreshold:
    def test_perfectly_separable_scores(self):
        # ID samples have high reliability, OOD samples have low reliability.
        scores = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        best_threshold, metrics = tuner.tune_threshold(scores, labels, metric="f1-id")
        assert 0.2 < best_threshold < 0.8
        assert metrics["best_f1_id"] == pytest.approx(1.0)

    def test_threshold_grid_has_expected_range(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 1, 0, 0])
        best_threshold, _ = tuner.tune_threshold(
            scores,
            labels,
            metric="f1-id",
            min_threshold=0.0,
            max_threshold=1.0,
            step=0.25,
        )
        assert best_threshold in {0.0, 0.25, 0.5, 0.75, 1.0}

    def test_youden_metric(self):
        scores = np.array([0.9, 0.8, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        best_threshold, metrics = tuner.tune_threshold(scores, labels, metric="youden")
        # Youden's J is maximized when all positives are above and all negatives below.
        assert 0.1 < best_threshold < 0.8
        assert metrics["best_youden_j"] == pytest.approx(1.0)


class TestSelectBestThreshold:
    def test_selects_max_f1_id(self):
        thresholds = np.array([0.0, 0.5, 1.0])
        f1_id = np.array([0.7, 0.9, 0.6])
        best = tuner._select_best_threshold(thresholds, f1_id, metric_name="f1-id")
        assert best == 0.5
