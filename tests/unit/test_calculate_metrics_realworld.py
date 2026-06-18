"""Tests for scripts.calculate_metrics_realworld."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import calculate_metrics_realworld as metrics


class TestBuildGroundTruth:
    def test_cellular_is_negative_viral_is_positive(self):
        labels = pd.DataFrame(
            {
                "contig_id": ["c1", "c2", "c3", "c4"],
                "fraction": ["cellular", "phage", "virus", "cellular"],
            }
        )
        y_true, classes = metrics.build_ground_truth(labels)
        np.testing.assert_array_equal(y_true, np.array([0, 1, 1, 0]))
        assert classes.tolist() == ["cellular", "phage", "virus"]


class TestBuildPredictions:
    def test_phage_and_virus_are_viral(self):
        preds = pd.DataFrame(
            {
                "contig_id": ["c1", "c2", "c3", "c4"],
                "prediction": ["bacteria", "phage", "virus", "archaea"],
                "reliability_score": [0.9, 0.8, 0.7, 0.6],
            }
        )
        y_pred = metrics.build_viral_predictions(preds)
        np.testing.assert_array_equal(y_pred, np.array([0, 1, 1, 0]))

    def test_reliability_cutoff_filters_predictions(self):
        preds = pd.DataFrame(
            {
                "contig_id": ["c1", "c2", "c3"],
                "prediction": ["phage", "phage", "phage"],
                "reliability_score": [0.9, 0.4, 0.6],
            }
        )
        y_pred = metrics.build_viral_predictions(preds, reliability_cutoff=0.5)
        # Below-cutoff predictions are treated as uncertain (negative for viral task)
        np.testing.assert_array_equal(y_pred, np.array([1, 0, 1]))


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = metrics.compute_binary_metrics(y_true, y_pred, sample_name="test")
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_imperfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = metrics.compute_binary_metrics(y_true, y_pred, sample_name="test")
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(0.8)


class TestEvaluateSample:
    def test_writes_metrics_and_confusion_matrix(self, tmp_path: Path):
        pred_path = tmp_path / "gut.tsv"
        label_path = tmp_path / "gut_labels.tsv"
        output_dir = tmp_path / "metrics"

        preds = pd.DataFrame(
            {
                "contig_id": ["c1", "c2", "c3", "c4"],
                "prediction": ["bacteria", "phage", "virus", "archaea"],
                "reliability_score": [0.9, 0.9, 0.9, 0.9],
                "phage_score": [0.1, 0.8, 0.2, 0.05],
                "virus_score": [0.05, 0.1, 0.85, 0.05],
                "bacteria_score": [0.85, 0.05, 0.05, 0.85],
                "archaea_score": [0.05, 0.05, 0.05, 0.85],
            }
        )
        labels = pd.DataFrame(
            {
                "contig_id": ["c1", "c2", "c3", "c4"],
                "contig_length": [1000, 2000, 3000, 4000],
                "fraction": ["cellular", "phage", "virus", "cellular"],
                "sample_id": ["S1"] * 4,
                "biome": ["gut"] * 4,
            }
        )
        preds.to_csv(pred_path, sep="\t", index=False)
        labels.to_csv(label_path, sep="\t", index=False)

        sample_metrics, cm = metrics.evaluate_sample(
            pred_path, label_path, output_dir, reliability_cutoff=0.5
        )

        assert sample_metrics["sample"] == "gut"
        assert sample_metrics["precision"] == pytest.approx(1.0)
        assert sample_metrics["recall"] == pytest.approx(1.0)
        assert (output_dir / "gut_metrics.json").exists()
        assert (output_dir / "gut_confusion_matrix.npy").exists()
        np.testing.assert_array_equal(cm, np.array([[2, 0], [0, 2]]))


class TestLoadPredictions:
    def test_replaces_triple_underscore_with_comma(self, tmp_path: Path):
        pred_path = tmp_path / "test.tsv"
        df = pd.DataFrame(
            {
                "contig_id": ["c1___sample1", "c2___sample1"],
                "prediction": ["phage", "bacteria"],
                "reliability_score": [0.8, 0.7],
            }
        )
        df.to_csv(pred_path, sep="\t", index=False)
        loaded = metrics.load_predictions(pred_path)
        assert loaded["contig_id"].tolist() == ["c1,sample1", "c2,sample1"]
