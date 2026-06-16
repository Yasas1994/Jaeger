"""Tests for jaeger.dataops.convert."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaeger.dataops import convert


class TestConvertDataset:
    def test_nucleotide_format(self, simple_csv_path: str, tmp_path: Path):
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
        assert "labels" in data

    def test_translated_format(self, simple_csv_path: str, tmp_path: Path):
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

    def test_both_format(self, simple_csv_path: str, tmp_path: Path):
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=simple_csv_path,
            output_path=str(out),
            format="both",
            crop_size=24,
            num_classes=2,
            num_workers=1,
        )
        assert out.exists()
        data = np.load(out)
        assert "nucleotide" in data
        assert "translated" in data

    def test_invalid_format(self, simple_csv_path: str, tmp_path: Path):
        with pytest.raises(ValueError):
            convert.convert_dataset(
                input_path=simple_csv_path,
                output_path=str(tmp_path / "out.npz"),
                format="tfrecord",
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


class TestNucleotideEncoder:
    def test_encode_nucleotide_int(self):
        from jaeger.dataops.convert import (
            _encode_nucleotide_batch,
            _build_nucleotide_lookups,
        )

        sequences = np.full((1, 4), ord("N"), dtype=np.uint8)
        sequences[0, :4] = np.frombuffer(b"ATGC", dtype=np.uint8)
        lengths = np.array([4], dtype=np.int32)
        user_map = {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
        ascii_to_user, comp_user, ascii_to_oh, comp_oh = _build_nucleotide_lookups(
            user_map
        )
        out = _encode_nucleotide_batch(
            sequences, lengths, 4, ascii_to_user, comp_user, ascii_to_oh, comp_oh, False
        )
        assert out.shape == (1, 2, 4)
        # forward strand: A=1, T=3, G=2, C=4
        assert out[0, 0].tolist() == [1, 3, 2, 4]

    def test_encode_nucleotide_onehot(self):
        from jaeger.dataops.convert import (
            _encode_nucleotide_batch,
            _build_nucleotide_lookups,
        )

        sequences = np.full((1, 4), ord("N"), dtype=np.uint8)
        sequences[0, :4] = np.frombuffer(b"ATGC", dtype=np.uint8)
        lengths = np.array([4], dtype=np.int32)
        user_map = {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
        ascii_to_user, comp_user, ascii_to_oh, comp_oh = _build_nucleotide_lookups(
            user_map
        )
        out = _encode_nucleotide_batch(
            sequences, lengths, 4, ascii_to_user, comp_user, ascii_to_oh, comp_oh, True
        )
        assert out.shape == (1, 2, 4, 4)
        assert out[0, 0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]  # A
        assert out[0, 0, 2].tolist() == [0.0, 1.0, 0.0, 0.0]  # G


class TestCropHelpers:
    def test_crop_starts_single_short_sequence(self):
        assert convert._crop_starts(10, 20, 5) == [0]

    def test_crop_starts_zero_stride(self):
        assert convert._crop_starts(100, 20, 0) == [0]

    def test_crop_starts_exact_fit(self):
        assert convert._crop_starts(60, 20, 20) == [0, 20, 40]

    def test_crop_starts_with_tail_window(self):
        starts = convert._crop_starts(55, 20, 20)
        assert starts == [0, 20, 35]
        assert starts[-1] + 20 == 55

    def test_generate_crops_multiple_sizes(self):
        crops = list(convert._generate_crops(25, [20, 10], 10))
        expected = [
            (20, 0, 20),
            (20, 5, 20),
            (10, 0, 10),
            (10, 10, 10),
            (10, 15, 10),
        ]
        assert crops == expected

    def test_pad_array_int_rank3(self):
        arr = np.ones((2, 3, 2), dtype=np.int32)
        padded = convert._pad_array(arr, 5, 0)
        assert padded.shape == (2, 3, 5)
        assert np.all(padded[..., 2:] == 0)

    def test_pad_array_float_rank4(self):
        arr = np.ones((1, 2, 3, 2), dtype=np.float32)
        padded = convert._pad_array(arr, 4, 0.0)
        assert padded.shape == (1, 2, 3, 4)
        assert np.all(padded[..., 2:] == 0.0)

    def test_pad_array_truncate(self):
        arr = np.ones((2, 3), dtype=np.int32)
        padded = convert._pad_array(arr, 2, 0)
        assert padded.shape == (2, 2)


class TestConvertToNpz:
    def _csv(self, tmp_path: Path, lines: list[str]) -> str:
        path = tmp_path / "input.csv"
        path.write_text("\n".join(lines))
        return str(path)

    def test_nucleotide_integer(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGC", "1,GGGGGGGGGGGG"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[12],
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        assert out.exists()
        data = np.load(out)
        assert "nucleotide" in data
        assert "labels" in data
        assert data["labels"].tolist() == [0, 1]
        assert data["nucleotide"].shape[0] == 2

    def test_translated_integer(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGCATGCATGCATGC"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="translated",
            crop_sizes=[24],
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        data = np.load(out)
        assert "translated" in data
        assert "codon_map" in data
        assert data["translated"].shape[1] == 6

    def test_both(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGCATGCATGCATGC"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="both",
            crop_sizes=[24],
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        data = np.load(out)
        assert "nucleotide" in data
        assert "translated" in data

    def test_one_hot_nucleotide(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGC"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[4],
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=True,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        data = np.load(out)
        assert data["nucleotide"].ndim == 4
        assert data["nucleotide"].dtype == np.float32

    def test_multi_crop_and_stride(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0," + "A" * 25])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[20, 10],
            stride=10,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        data = np.load(out)
        # 25-mer with crop 20 (2 crops: 0,5) and crop 10 (3 crops: 0,10,15)
        assert data["labels"].shape[0] == 5

    def test_dicodon(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGCATGCATGCATGC"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="translated",
            crop_sizes=[24],
            stride=0,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="cod_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        data = np.load(out)
        assert data["codon_map"] == "cod_id"
        assert data["translated"].shape[1] == 6
        # first dicodon position should be encoded (value > 0)
        assert np.any(data["translated"][0, 0] > 0)

    def test_invalid_format(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGC"])
        out = tmp_path / "out.npz"
        with pytest.raises(ValueError):
            convert._convert_to_npz(
                input_path=csv,
                output_path=str(out),
                fmt="unknown",
                crop_sizes=[4],
                stride=0,
                num_classes=2,
                num_workers=1,
                one_hot=False,
                pad_int=0,
                codon_map_name="codon_id",
                nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
                compress="default",
            )
