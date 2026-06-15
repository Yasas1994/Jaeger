"""Tests for jaeger.dataops.convert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaeger.dataops import convert


class TestConvertDataset:
    def test_numpy_full(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="numpy_full",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()
        data = np.load(out)
        assert "translated" in data
        assert "label" in data

    def test_numpy_raw(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="numpy_raw",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()

    def test_numpy_raw_variable(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="numpy_raw_variable",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()

    def test_invalid_format(self, simple_csv_path: str, tmp_path: Path):
        with pytest.raises(ValueError):
            convert.convert_dataset(
                input_path=simple_csv_path,
                output_path=str(tmp_path / "out.npz"),
                format="unknown",
                crop_size=24,
                num_classes=2,
            )
