"""Tests for jaeger.dataops.convert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaeger.dataops import convert


class TestFeatureHelpers:
    def test_int64_feature(self):
        feature = convert._int64_feature([1, 2, 3])
        assert list(feature.int64_list.value) == [1, 2, 3]

    def test_float_feature(self):
        feature = convert._float_feature([1.0, 2.0])
        assert list(feature.float_list.value) == pytest.approx([1.0, 2.0])


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


class TestMapHelpers:
    def test_get_codon_map_codon_id(self):
        from jaeger.dataops.convert import _get_codon_map

        mapping = _get_codon_map("codon_id")
        assert len(mapping) == 64
        assert mapping[0] == 0

    def test_get_codon_map_invalid(self):
        from jaeger.dataops.convert import _get_codon_map

        with pytest.raises(ValueError):
            _get_codon_map("not_a_map")

    def test_parse_nucleotide_map_default(self):
        from jaeger.dataops.convert import _parse_nucleotide_map

        m = _parse_nucleotide_map(None)
        assert m["A"] == 1 and m["N"] == 0

    def test_parse_nucleotide_map_custom(self):
        from jaeger.dataops.convert import _parse_nucleotide_map

        m = _parse_nucleotide_map('{"A":0,"G":1,"T":2,"C":3,"N":4}')
        assert m == {"A": 0, "G": 1, "T": 2, "C": 3, "N": 4}

    def test_parse_nucleotide_map_missing_base(self):
        from jaeger.dataops.convert import _parse_nucleotide_map

        with pytest.raises(ValueError):
            _parse_nucleotide_map('{"A":0}')

    def test_parse_nucleotide_map_invalid_json(self):
        from jaeger.dataops.convert import _parse_nucleotide_map

        with pytest.raises(ValueError):
            _parse_nucleotide_map("not json")
