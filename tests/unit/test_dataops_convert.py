"""Tests for jaeger.dataops.convert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaeger.dataops import convert


class TestOneHotMemoryGuard:
    """Memory guard prevents OOM on huge one-hot conversions."""

    def test_estimate_nucleotide_onehot_memory(self):
        # 1M rows, crop 500, one-hot nucleotide
        # 1_000_000 * 2 * 500 * 4 * 4 = 16_000_000_000 bytes
        assert convert._estimate_onehot_memory(
            total_rows=1_000_000,
            crop_size=500,
            format="nucleotide",
            one_hot=True,
        ) == 16_000_000_000

    def test_estimate_no_memory_for_non_onehot(self):
        assert (
            convert._estimate_onehot_memory(
                total_rows=1_000_000,
                crop_size=500,
                format="nucleotide",
                one_hot=False,
            )
            == 0
        )

    def test_check_onehot_memory_raises_when_too_large(self):
        with pytest.raises(MemoryError, match="one-hot"):
            convert._check_onehot_memory(
                estimated_bytes=100,
                available_bytes=100,
                fraction=0.5,
            )

    def test_check_onehot_memory_passes_when_small(self):
        # Should not raise
        convert._check_onehot_memory(
            estimated_bytes=1,
            available_bytes=100,
            fraction=0.5,
        )

    def test_guard_rejects_huge_onehot_conversion(self, tmp_path: Path):
        """A tiny CSV with enormous crop_size should be rejected before allocation."""
        csv = tmp_path / "big.csv"
        csv.write_text("0,ATGC\n")
        with pytest.raises(MemoryError, match="one-hot"):
            convert.convert_dataset(
                input_path=str(csv),
                output_path=str(tmp_path / "out.npz"),
                format="nucleotide",
                one_hot=True,
                crop_size=1_000_000_000,
                num_classes=2,
                num_workers=1,
                _available_memory_bytes=1024,
            )


class TestConvertDataset:
    def test_translated_integer(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="translated",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()
        data = np.load(out)
        assert "translated" in data
        assert "label" in data
        assert "pad_int" in data
        assert "lengths" in data
        assert data["pad_int"].item() == 0
        assert data["translated"].dtype == np.int32

    def test_nucleotide_integer(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="nucleotide",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()
        data = np.load(out)
        assert "nucleotide" in data
        assert "label" in data
        assert "pad_int" in data
        assert data["nucleotide"].dtype == np.int32

    def test_lowercase_consistency(self, tmp_path: Path):
        """Lowercase sequences must encode identically to uppercase."""
        csv = tmp_path / "mixed.csv"
        csv.write_text("0,ATGCatgcNN\n")
        out_upper = tmp_path / "out_upper.npz"
        convert.convert_dataset(
            input_path=str(csv),
            output_path=str(out_upper),
            format="nucleotide",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        csv_lower = tmp_path / "mixed_lower.csv"
        csv_lower.write_text("0,atgcATGCnn\n")
        out_lower = tmp_path / "out_lower.npz"
        convert.convert_dataset(
            input_path=str(csv_lower),
            output_path=str(out_lower),
            format="nucleotide",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        upper_data = np.load(out_upper)
        lower_data = np.load(out_lower)
        assert np.array_equal(upper_data["nucleotide"], lower_data["nucleotide"])

    def test_both_onehot(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="both",
            one_hot=True,
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()
        data = np.load(out)
        assert "nucleotide" in data
        assert "translated" in data
        assert "label" in data
        assert data["nucleotide"].dtype == np.float32
        assert data["translated"].dtype == np.float32

    def test_codon_map_aa_id(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="translated",
            codon_map="AA_ID",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        data = np.load(out)
        assert "translated" in data
        assert data["vocab_size"].item() == 21

    def test_pad_int(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="nucleotide",
            pad_int=7,
            crop_size=30,
            num_classes=2,
            num_workers=1,
        )
        data = np.load(out)
        assert data["pad_int"].item() == 7
        nuc = data["nucleotide"]
        # Actual token indices are offset by +1, so 0 is never used.
        assert (nuc == 0).sum() == 0

    def test_pad_int_collision_raises(self, simple_csv_path: str, tmp_path: Path):
        with pytest.raises(ValueError, match="collides"):
            convert.convert_dataset(
                input_path=simple_csv_path,
                output_path=str(tmp_path / "out.npz"),
                format="nucleotide",
                pad_int=1,
                crop_size=24,
                num_classes=2,
                num_workers=1,
            )

    def test_stride_sliding_window(self, tmp_path: Path):
        csv = tmp_path / "long.csv"
        # 1000 bp sequence, label 0
        csv.write_text(f"0,{'ATGC' * 250}\n")
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=str(csv),
            output_path=str(out),
            format="nucleotide",
            crop_size=500,
            stride=500,
            num_classes=2,
            num_workers=1,
        )
        data = np.load(out)
        assert data["nucleotide"].shape[0] == 2

    def test_invalid_format(self, simple_csv_path: str, tmp_path: Path):
        with pytest.raises(ValueError):
            convert.convert_dataset(
                input_path=simple_csv_path,
                output_path=str(tmp_path / "out.npz"),
                format="unknown",
                crop_size=24,
                num_classes=2,
            )

    def test_default_workers(self, simple_csv_path: str, tmp_path: Path):
        """convert_dataset should complete with the default worker count.

        This is a regression guard against hangs caused by the default
        ``fork``-based ``multiprocessing.Pool`` on Linux when the parent
        process has background threads.
        """
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="translated",
            crop_size=24,
            num_classes=2,
        )
        assert out.exists()
        data = np.load(out)
        assert "translated" in data
        assert "label" in data

    @pytest.mark.skipif(
        not convert.HAS_NUMBA, reason="Numba not installed"
    )
    def test_numba_outputs_match_python(self, simple_csv_path: str, tmp_path: Path, monkeypatch):
        """Numba path must produce the same arrays as the pure-Python path."""
        assert convert.HAS_NUMBA, "Numba must be available for this test"

        out_numba = tmp_path / "out_numba.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out_numba),
            format="both",
            one_hot=True,
            crop_size=30,
            num_classes=2,
            num_workers=1,
        )
        data_numba = dict(np.load(out_numba))

        monkeypatch.setattr(convert, "HAS_NUMBA", False)
        out_py = tmp_path / "out_py.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out_py),
            format="both",
            one_hot=True,
            crop_size=30,
            num_classes=2,
            num_workers=1,
        )
        data_py = dict(np.load(out_py))

        assert np.array_equal(data_numba["nucleotide"], data_py["nucleotide"])
        assert np.array_equal(data_numba["translated"], data_py["translated"])
        assert np.array_equal(data_numba["label"], data_py["label"])
        assert np.array_equal(data_numba["lengths"], data_py["lengths"])
        assert data_numba["pad_int"].item() == data_py["pad_int"].item()
        assert data_numba["vocab_size"].item() == data_py["vocab_size"].item()

    def test_fallback_when_numba_disabled(self, simple_csv_path: str, tmp_path: Path, monkeypatch):
        """Pure-Python fallback must still produce valid output."""
        monkeypatch.setattr(convert, "HAS_NUMBA", False)
        out = tmp_path / "out_py.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="translated",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        data = dict(np.load(out))
        assert "translated" in data
        assert "label" in data
        assert data["translated"].dtype == np.int32
        assert data["translated"].shape[0] > 0
