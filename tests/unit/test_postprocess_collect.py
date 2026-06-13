"""Tests for jaeger.postprocess.collect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jaeger.postprocess import collect


class TestFracAboveThreshold:
    def test_empty(self):
        assert collect.frac_above_threshold(None) == "-"

    def test_zero(self):
        assert collect.frac_above_threshold(np.array([])) == "0.00"

    def test_half(self):
        arr = np.array([[0.6, 0.3], [0.2, 0.8]])
        assert collect.frac_above_threshold(arr, threshold=0.5) == "0.50"


class TestPredToDict:
    def _make_y_pred(self, with_reliability: bool = True):
        # 6 windows from 2 contigs (3 + 3), 3 classes.
        y_pred = {
            "prediction": np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.6, 0.3, 0.1],
                    [0.5, 0.3, 0.2],
                    [0.2, 0.3, 0.5],
                    [0.1, 0.3, 0.6],
                    [0.1, 0.2, 0.7],
                ],
                dtype=np.float32,
            ),
            "meta_0": np.array(["c1", "c1", "c1", "c2", "c2", "c2"]),
            "meta_2": np.array([0, 0, 1, 0, 0, 1]),  # split flags
            "meta_4": np.array([1000, 1000, 1000, 2000, 2000, 2000]),
            "meta_5": np.full(6, 5, dtype=np.float32),
            "meta_6": np.full(6, 5, dtype=np.float32),
            "meta_7": np.full(6, 5, dtype=np.float32),
            "meta_8": np.full(6, 5, dtype=np.float32),
            "meta_9": np.zeros(6, dtype=np.float32),
        }
        if with_reliability:
            y_pred["reliability"] = np.array(
                [[0.8], [0.7], [0.6], [0.6], [0.9], [0.8]], dtype=np.float32
            )
        return y_pred

    def test_pred_to_dict_with_reliability(self):
        y_pred = self._make_y_pred(with_reliability=True)
        kwargs = {
            "fsize": 500,
            "class_map": {"num_classes": 3},
            "term_repeats": pd.DataFrame(
                {
                    "contig_id": ["c1", "c2"],
                    "terminal_repeats": [0, 1],
                    "repeat_length": [0, 10],
                }
            ),
        }
        data, data_full = collect.pred_to_dict(y_pred, **kwargs)
        assert data["headers"].tolist() == ["c1", "c2"]
        assert len(data_full["predictions"]) == 2
        assert data["has_reliability"] is True

    def test_pred_to_dict_without_reliability(self):
        y_pred = self._make_y_pred(with_reliability=False)
        kwargs = {
            "fsize": 500,
            "class_map": {"num_classes": 3},
            "term_repeats": pd.DataFrame(
                {
                    "contig_id": ["c1", "c2"],
                    "terminal_repeats": [0, 1],
                    "repeat_length": [0, 10],
                }
            ),
        }
        data, _ = collect.pred_to_dict(y_pred, **kwargs)
        assert data["ood"] is None
        assert data["has_reliability"] is False


class TestGenerateSummary:
    def test_generate_summary(self):
        data = {
            "headers": np.array(["c1"]),
            "length": np.array([1000]),
            "consensus": np.array([2]),
            "per_class_counts": [{0: 0, 1: 1, 2: 2}],
            "pred_sum": np.array([[0.2, 0.3, 0.5]], dtype=np.float16),
            "pred_var": np.array([[0.01, 0.01, 0.01]], dtype=np.float16),
            "frag_pred": [np.array([2, 2, 1])],
            "ood": np.array([0.9], dtype=np.float16),
            "has_reliability": True,
            "entropy": np.array([0.5], dtype=np.float16),
            "energy": np.array([-1.0], dtype=np.float16),
            "host_contam": np.array([False]),
            "prophage_contam": np.array([False]),
            "gc": [np.array([0.5])],
            "ns": [np.array([0.0])],
            "repeats": pd.DataFrame(
                {"contig_id": ["c1"], "terminal_repeats": [0], "repeat_length": [0]}
            ),
        }
        df = collect.generate_summary(
            data, labels=["bacteria", "phage", "virus"], indices=[0, 1, 2]
        )
        assert isinstance(df, pd.DataFrame)
        assert df.iloc[0]["prediction"] == "virus"
        assert df.iloc[0]["reliability_score"] == pytest.approx(0.9)

    def test_generate_summary_without_reliability(self):
        data = {
            "headers": np.array(["c1"]),
            "length": np.array([1000]),
            "consensus": np.array([1]),
            "per_class_counts": [{0: 0, 1: 2, 2: 0}],
            "pred_sum": np.array([[0.2, 0.8, 0.0]], dtype=np.float16),
            "pred_var": np.array([[0.01, 0.01, 0.01]], dtype=np.float16),
            "frag_pred": [np.array([1, 1])],
            "ood": None,
            "has_reliability": False,
            "entropy": np.array([0.5], dtype=np.float16),
            "energy": np.array([-1.0], dtype=np.float16),
            "host_contam": np.array([False]),
            "prophage_contam": np.array([False]),
            "gc": [np.array([0.5])],
            "ns": [np.array([0.0])],
            "repeats": pd.DataFrame(
                {"contig_id": ["c1"], "terminal_repeats": [0], "repeat_length": [0]}
            ),
        }
        df = collect.generate_summary(
            data, labels=["bacteria", "phage", "virus"], indices=[0, 1, 2]
        )
        assert df.iloc[0]["reliability_score"] == "unavailable"


class TestWriteOutput:
    def test_write_output(self, tmp_path):
        data = {
            "headers": np.array(["c1"]),
            "length": np.array([1000]),
            "consensus": np.array([1]),
            "per_class_counts": [{0: 0, 1: 2, 2: 0}],
            "pred_sum": np.array([[0.2, 0.8, 0.0]], dtype=np.float16),
            "pred_var": np.array([[0.01, 0.01, 0.01]], dtype=np.float16),
            "frag_pred": [np.array([1, 1])],
            "ood": np.array([0.9], dtype=np.float16),
            "has_reliability": True,
            "entropy": np.array([0.5], dtype=np.float16),
            "energy": np.array([-1.0], dtype=np.float16),
            "host_contam": np.array([False]),
            "prophage_contam": np.array([False]),
            "gc": [np.array([0.5])],
            "ns": [np.array([0.0])],
            "repeats": pd.DataFrame(
                {"contig_id": ["c1"], "terminal_repeats": [0], "repeat_length": [0]}
            ),
        }
        out = tmp_path / "out.tsv"
        out_phage = tmp_path / "phage.tsv"
        n_rows = collect.write_output(
            data,
            reliability_cutoff=0.5,
            phage_score=0.5,
            labels=["bacteria", "phage", "virus"],
            indices=[0, 1, 2],
            output_table_path=str(out),
            output_phage_table_path=str(out_phage),
        )
        assert n_rows == 1
        assert out.exists()
        assert out_phage.exists()

    def test_write_output_without_reliability(self, tmp_path):
        data = {
            "headers": np.array(["c1"]),
            "length": np.array([1000]),
            "consensus": np.array([1]),
            "per_class_counts": [{0: 0, 1: 2, 2: 0}],
            "pred_sum": np.array([[0.2, 0.8, 0.0]], dtype=np.float16),
            "pred_var": np.array([[0.01, 0.01, 0.01]], dtype=np.float16),
            "frag_pred": [np.array([1, 1])],
            "ood": None,
            "has_reliability": False,
            "entropy": np.array([0.5], dtype=np.float16),
            "energy": np.array([-1.0], dtype=np.float16),
            "host_contam": np.array([False]),
            "prophage_contam": np.array([False]),
            "gc": [np.array([0.5])],
            "ns": [np.array([0.0])],
            "repeats": pd.DataFrame(
                {"contig_id": ["c1"], "terminal_repeats": [0], "repeat_length": [0]}
            ),
        }
        out = tmp_path / "out.tsv"
        out_phage = tmp_path / "phage.tsv"
        n_rows = collect.write_output(
            data,
            reliability_cutoff=0.5,
            phage_score=0.5,
            labels=["bacteria", "phage", "virus"],
            indices=[0, 1, 2],
            output_table_path=str(out),
            output_phage_table_path=str(out_phage),
        )
        assert n_rows == 1
        assert out.exists()
        assert out_phage.exists()
        df = pd.read_table(out)
        assert df.iloc[0]["reliability_score"] == "unavailable"
