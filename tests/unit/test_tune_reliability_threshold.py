"""Tests for jaeger.postprocess.threshold."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from jaeger.postprocess.threshold import (
    _select_best_threshold,
    calibration_summary,
    collect_scores_and_labels,
    extract_labels_from_dataset,
    predict_reliability_scores,
    tune_reliability_threshold,
)


def _batched_label_dataset(label_rows, batch_size=2):
    labels = tf.constant(np.array(label_rows, dtype=np.float32))
    features = tf.zeros((labels.shape[0], 1), dtype=tf.float32)
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)


class TestExtractLabelsFromDataset:
    def test_column_vector_labels_are_squeezed_not_argmaxed(self):
        # Binary reliability labels arrive as (batch, 1) float columns; they
        # must be squeezed, not argmaxed (which would zero every label).
        ds = _batched_label_dataset([[1.0], [0.0], [1.0], [1.0], [0.0]])
        labels = extract_labels_from_dataset(ds)
        np.testing.assert_array_equal(labels, np.array([1, 0, 1, 1, 0], dtype=np.int32))

    def test_one_hot_labels_are_argmaxed(self):
        ds = _batched_label_dataset([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        labels = extract_labels_from_dataset(ds)
        np.testing.assert_array_equal(labels, np.array([1, 0, 2], dtype=np.int32))


class _StubPredictModel:
    def __init__(self, outputs):
        self.outputs = np.asarray(outputs, dtype=np.float32).reshape(-1, 1)

    def predict(self, data, batch_size=64, verbose=0):
        return self.outputs


class TestPredictReliabilityScores:
    def test_logits_are_converted_to_probabilities(self):
        logits = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        model = _StubPredictModel(logits)
        scores = predict_reliability_scores(model, np.zeros((3, 1)))
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        np.testing.assert_allclose(scores, 1.0 / (1.0 + np.exp(-logits)), rtol=1e-5)

    def test_probabilities_pass_through_unchanged(self):
        model = _StubPredictModel([0.2, 0.9, 0.5])
        scores = predict_reliability_scores(model, np.zeros((3, 1)))
        np.testing.assert_allclose(scores, [0.2, 0.9, 0.5], rtol=1e-6)


class _IdentityLogitModel:
    """predict_on_batch returns a bounded logit derived from the input id.

    The id is affine-mapped to [-2, 2] so the sigmoid in the score conversion
    stays in its non-saturated regime and every score is unique.
    """

    def predict_on_batch(self, x):
        arr = np.asarray(x, dtype=np.float32)
        ids = arr.reshape(arr.shape[0], -1)[:, 0]
        return (ids - 20.0) / 10.0


class TestCollectScoresAndLabels:
    def test_scores_and_labels_stay_aligned_for_shuffled_dataset(self):
        # Validation datasets are shuffled without a fixed seed, so collecting
        # labels and scores in two separate passes would misalign them; a
        # single pass must keep every score paired with its own label.
        n = 40
        ids = np.arange(n, dtype=np.float32)
        features = tf.constant(ids.reshape(n, 1))
        labels = tf.constant((ids >= n / 2).astype(np.float32).reshape(n, 1))
        ds = (
            tf.data.Dataset.from_tensor_slices((features, labels))
            .shuffle(buffer_size=n)
            .batch(8)
        )

        scores, out_labels = collect_scores_and_labels(_IdentityLogitModel(), ds)

        order = np.argsort(scores)
        expected = (np.sort(ids) >= n / 2).astype(np.int32)
        np.testing.assert_array_equal(out_labels[order], expected)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_partial_last_batch_is_handled(self):
        features = tf.constant(np.arange(5, dtype=np.float32).reshape(5, 1))
        labels = tf.constant(
            np.array([[0.0], [1.0], [0.0], [1.0], [1.0]], dtype=np.float32)
        )
        ds = tf.data.Dataset.from_tensor_slices((features, labels)).batch(2)

        scores, out_labels = collect_scores_and_labels(_IdentityLogitModel(), ds)

        assert scores.shape == (5,)
        np.testing.assert_array_equal(
            out_labels, np.array([0, 1, 0, 1, 1], dtype=np.int32)
        )


class TestTuneThreshold:
    def test_perfectly_separable_scores(self):
        # ID samples have high reliability, OOD samples have low reliability.
        scores = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        best_threshold, rows, _ = tune_reliability_threshold(
            scores, labels, metric="f1-id"
        )
        assert 0.2 < best_threshold < 0.8
        assert max(r["f1_id"] for r in rows) == pytest.approx(1.0)

    def test_threshold_grid_has_expected_range(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1, 1, 0, 0])
        best_threshold, _, _ = tune_reliability_threshold(
            scores,
            labels,
            metric="f1-id",
            min_threshold=0.0,
            max_threshold=1.0,
            step=0.25,
        )
        assert best_threshold in {0.0, 0.25, 0.5, 0.75, 1.0}

    def test_single_class_labels_raise(self):
        # Tuning on a single class is meaningless; fail loudly instead of
        # silently writing a degenerate threshold.
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([1, 1, 1])
        with pytest.raises(ValueError):
            tune_reliability_threshold(scores, labels)

    def test_youden_metric(self):
        scores = np.array([0.9, 0.8, 0.1, 0.1])
        labels = np.array([1, 1, 0, 0])
        best_threshold, rows, _ = tune_reliability_threshold(
            scores, labels, metric="youden"
        )
        # Youden's J is maximized when all positives are above and all negatives below.
        assert 0.1 < best_threshold < 0.8
        assert max(r["youden_j"] for r in rows) == pytest.approx(1.0)


class TestSelectBestThreshold:
    def test_selects_max_f1_id(self):
        thresholds = np.array([0.0, 0.5, 1.0])
        f1_id = np.array([0.7, 0.9, 0.6])
        best = _select_best_threshold(thresholds, f1_id, metric_name="f1-id")
        assert best == 0.5


class TestCalibration:
    def test_perfectly_calibrated(self):
        # scores exactly equal the 0/1 labels -> perfect calibration.
        scores = np.array([1.0, 1.0, 0.0, 0.0])
        labels = np.array([1, 1, 0, 0])
        ece, brier, rows = calibration_summary(scores, labels, n_bins=10)
        assert ece == pytest.approx(0.0, abs=1e-6)
        assert brier == pytest.approx(0.0, abs=1e-6)
        assert sum(r["count"] for r in rows) == len(labels)

    def test_ece_within_unit_interval(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200).astype(np.float32)
        labels = (scores > 0.5).astype(np.int32)
        ece, brier, _ = calibration_summary(scores, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= brier <= 1.0
