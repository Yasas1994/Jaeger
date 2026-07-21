"""Tests for jaeger.dataops.convert."""

from __future__ import annotations

from pathlib import Path

import json

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
            sequences,
            lengths,
            4,
            ascii_to_user,
            comp_user,
            ascii_to_oh,
            comp_oh,
            False,
            0,
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
            sequences,
            lengths,
            4,
            ascii_to_user,
            comp_user,
            ascii_to_oh,
            comp_oh,
            True,
            0,
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
        crops = list(convert._generate_crops(25, [20, 10], [10, 10]))
        expected = [
            (20, 0, 20),
            (20, 5, 20),
            (10, 0, 10),
            (10, 10, 10),
            (10, 15, 10),
        ]
        assert crops == expected

    def test_generate_crops_per_size_strides(self):
        crops = list(convert._generate_crops(25, [20, 10], [5, 10]))
        expected = [
            # crop_size 20, stride 5 -> starts 0, 5, 5? Let's compute:
            # _crop_starts(25,20,5): range(0,6,5) -> [0,5]; last+crop=25 so no tail
            (20, 0, 20),
            (20, 5, 20),
            # crop_size 10, stride 10 -> starts 0,10,15? wait range(0,16,10) -> [0,10]; 10+10=20<25 -> append 15
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


class TestOverlap:
    def test_per_crop_strides_produce_expected_crops(self, tmp_path: Path):
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("0," + "A" * 25 + "\n")
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=str(csv_path),
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[20, 10],
            strides=[10, 5],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        data = np.load(out)
        # crop 20 with stride 10 -> starts [0, 5] (tail window)
        # crop 10 with stride 5  -> starts [0, 5, 10, 15]
        assert data["labels"].shape[0] == 6

    def test_overlap_memory_estimate_multiplier(self):
        single = convert._estimate_onehot_memory(
            total_rows=1,
            crop_size=100,
            fmt="nucleotide",
            one_hot=True,
            stride=100,
        )
        overlap = convert._estimate_onehot_memory(
            total_rows=1,
            crop_size=100,
            fmt="nucleotide",
            one_hot=True,
            stride=50,
        )
        assert overlap == single * 2


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
            strides=[0],
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
            strides=[0],
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
            strides=[0],
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
            strides=[0],
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
            strides=[10, 10],
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
            strides=[0],
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
                strides=[0],
                num_classes=2,
                num_workers=1,
                one_hot=False,
                pad_int=0,
                codon_map_name="codon_id",
                nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
                compress="default",
            )

    def test_one_hot_nucleotide_multi_crop(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGC"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[12, 8],
            strides=[0, 0],
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
        assert data["nucleotide"].shape == (2, 2, 12, 4)
        assert np.all(data["nucleotide"][1, :, 8:, :] == 0.0)

    def test_nucleotide_fast_path_unpadded(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGC", "1,GGGG"])
        out = tmp_path / "out.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            pad=False,
        )
        data = np.load(out, allow_pickle=True)
        assert data["padded"].item() is False
        assert data["nucleotide"].dtype == object
        assert data["nucleotide"][1].shape == (2, 4)


class TestMemoryEstimate:
    def test_per_row_nucleotide_integer(self):
        b = convert._estimate_output_bytes_per_row(
            12, "nucleotide", one_hot=False, codon_map_len=None
        )
        assert b == 2 * 12 * 4

    def test_per_row_nucleotide_onehot(self):
        b = convert._estimate_output_bytes_per_row(
            12, "nucleotide", one_hot=True, codon_map_len=None
        )
        assert b == 2 * 12 * 4 * 4

    def test_per_row_both(self):
        b = convert._estimate_output_bytes_per_row(
            12, "both", one_hot=False, codon_map_len=64
        )
        seq_len = 12 // 3 - 1
        expected = 2 * 12 * 4 + 6 * seq_len * 4
        assert b == expected


class TestFinalizeBatchArrays:
    def test_unpadded_nucleotide_object_array(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGCATGC\n1,GGGG\n")
        result = convert._process_chunk_npz(
            lines=csv.read_text().splitlines(),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            one_hot=False,
            pad_int=0,
            nucleotide_lookups=convert._build_nucleotide_lookups(
                {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
            ),
            codon_lut=np.empty(0, dtype=np.int32),
            codon_map_len=64,
            standard_codon_lut3=np.empty(0, dtype=np.int32),
            dicodon_lut=np.empty(0, dtype=np.int32),
            ascii_lut=np.zeros(256, dtype=np.int8),
            comp_lut=np.zeros(256, dtype=np.int8),
        )
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="nucleotide",
            crop_sizes=[12],
            one_hot=False,
            codon_map_len=None,
            pad=False,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["nucleotide"].dtype == object
        assert finalized["nucleotide"][0].shape == (2, 12)
        assert finalized["nucleotide"][1].shape == (2, 4)

    def test_padded_onehot_nucleotide_multi_crop(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGCATGC\n")
        result = convert._process_chunk_npz(
            lines=csv.read_text().splitlines(),
            fmt="nucleotide",
            crop_sizes=[12, 8],
            strides=[0, 0],
            one_hot=True,
            pad_int=0,
            nucleotide_lookups=convert._build_nucleotide_lookups(
                {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
            ),
            codon_lut=np.empty(0, dtype=np.int32),
            codon_map_len=64,
            standard_codon_lut3=np.empty(0, dtype=np.int32),
            dicodon_lut=np.empty(0, dtype=np.int32),
            ascii_lut=np.zeros(256, dtype=np.int8),
            comp_lut=np.zeros(256, dtype=np.int8),
        )
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="nucleotide",
            crop_sizes=[12, 8],
            one_hot=True,
            codon_map_len=None,
            pad=True,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["nucleotide"].ndim == 4
        assert finalized["nucleotide"].shape == (2, 2, 12, 4)
        assert np.all(finalized["nucleotide"][1, :, 8:, :] == 0.0)

    def test_padded_integer_nucleotide_multi_crop(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGCATGC\n")
        result = convert._process_chunk_npz(
            lines=csv.read_text().splitlines(),
            fmt="nucleotide",
            crop_sizes=[12, 8],
            strides=[0, 0],
            one_hot=False,
            pad_int=0,
            nucleotide_lookups=convert._build_nucleotide_lookups(
                {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
            ),
            codon_lut=np.empty(0, dtype=np.int32),
            codon_map_len=64,
            standard_codon_lut3=np.empty(0, dtype=np.int32),
            dicodon_lut=np.empty(0, dtype=np.int32),
            ascii_lut=np.zeros(256, dtype=np.int8),
            comp_lut=np.zeros(256, dtype=np.int8),
        )
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="nucleotide",
            crop_sizes=[12, 8],
            one_hot=False,
            codon_map_len=None,
            pad=True,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["nucleotide"].ndim == 3
        assert finalized["nucleotide"].shape == (2, 2, 12)
        assert np.all(finalized["nucleotide"][1, :, 8:] == 0)

    def test_unpadded_translated_object_array(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGCATGCATGCATGCATGC\n")
        codon_lut = convert._build_codon_lut(convert._get_codon_map("codon_id"))
        ascii_lut, comp_lut = convert._build_numba_lookups()[1:]
        result = convert._process_chunk_npz(
            lines=csv.read_text().splitlines(),
            fmt="translated",
            crop_sizes=[24],
            strides=[0],
            one_hot=False,
            pad_int=0,
            nucleotide_lookups=convert._build_nucleotide_lookups(
                {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
            ),
            codon_lut=codon_lut,
            codon_map_len=64,
            standard_codon_lut3=np.empty(0, dtype=np.int32),
            dicodon_lut=np.empty(0, dtype=np.int32),
            ascii_lut=ascii_lut,
            comp_lut=comp_lut,
        )
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="translated",
            crop_sizes=[24],
            one_hot=False,
            codon_map_len=64,
            pad=False,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["translated"].dtype == object
        assert finalized["translated"][0].shape[0] == 6
        assert finalized["translated"][0].dtype == np.int32

    def test_unpadded_both_object_arrays(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGCATGCATGCATGCATGC\n")
        codon_lut = convert._build_codon_lut(convert._get_codon_map("codon_id"))
        ascii_lut, comp_lut = convert._build_numba_lookups()[1:]
        result = convert._process_chunk_npz(
            lines=csv.read_text().splitlines(),
            fmt="both",
            crop_sizes=[24],
            strides=[0],
            one_hot=False,
            pad_int=0,
            nucleotide_lookups=convert._build_nucleotide_lookups(
                {"A": 1, "G": 2, "T": 3, "C": 4, "N": 0}
            ),
            codon_lut=codon_lut,
            codon_map_len=64,
            standard_codon_lut3=np.empty(0, dtype=np.int32),
            dicodon_lut=np.empty(0, dtype=np.int32),
            ascii_lut=ascii_lut,
            comp_lut=comp_lut,
        )
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="both",
            crop_sizes=[24],
            one_hot=False,
            codon_map_len=64,
            pad=False,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["nucleotide"].dtype == object
        assert finalized["translated"].dtype == object
        assert isinstance(finalized["nucleotide_map"], str)
        assert isinstance(finalized["codon_map"], str)

    def test_empty_arrays_padded(self, tmp_path: Path):
        result = {
            "nucleotide": [],
            "translated": [],
            "labels": np.empty((0,), dtype=np.int32),
            "lengths": np.empty((0,), dtype=np.int32),
            "translated_lengths": np.empty((0,), dtype=np.int32),
        }
        finalized = convert._finalize_batch_arrays(
            result,
            fmt="both",
            crop_sizes=[12],
            one_hot=False,
            codon_map_len=64,
            pad=True,
            pad_int=0,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            codon_map_name="codon_id",
        )
        assert finalized["nucleotide"].shape == (0,)
        assert finalized["translated"].shape == (0,)


class TestStreamingConvert:
    def _csv(self, tmp_path: Path, lines: list[str]) -> str:
        path = tmp_path / "input.csv"
        path.write_text("\n".join(lines))
        return str(path)

    def _load_stream(self, path: str) -> dict[str, np.ndarray]:
        """Load either a legacy NPZ or a sharded streaming NPZ."""
        data = np.load(path, allow_pickle=True)
        if "_jaeger_manifest" in data:
            manifest = json.loads(str(data["_jaeger_manifest"].item()))
            keys = manifest["keys"]
            num_shards = manifest["num_shards"]
            out: dict[str, np.ndarray] = {}
            for key in keys:
                shards = [data[f"{key}_{i:05d}"] for i in range(num_shards)]
                if shards and shards[0].dtype == object:
                    total = sum(len(s) for s in shards)
                    arr = np.empty(total, dtype=object)
                    idx = 0
                    for s in shards:
                        arr[idx : idx + len(s)] = s
                        idx += len(s)
                    out[key] = arr
                else:
                    out[key] = np.concatenate(shards, axis=0)
            return out
        return {k: data[k] for k in data.files}

    def test_streaming_matches_fast_path(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0," + "A" * 25, "1," + "G" * 25])
        fast = tmp_path / "fast.npz"
        stream = tmp_path / "stream.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(fast),
            fmt="nucleotide",
            crop_sizes=[20],
            strides=[10],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        convert._convert_to_npz_streaming(
            input_path=csv,
            output_path=str(stream),
            fmt="nucleotide",
            crop_sizes=[20],
            strides=[10],
            num_classes=2,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            max_memory_bytes=1 * 1024 * 1024,
            pad=True,
        )
        fast_data = np.load(fast)
        stream_data = self._load_stream(str(stream))
        assert np.array_equal(fast_data["labels"], stream_data["labels"])
        assert np.array_equal(fast_data["nucleotide"], stream_data["nucleotide"])

    def test_streaming_translated_matches_fast_path(self, tmp_path: Path):
        csv = self._csv(
            tmp_path, ["0,ATGCATGCATGCATGCATGCATGC", "1,GGGGGGGGGGGGGGGGGGGGGGGG"]
        )
        fast = tmp_path / "fast.npz"
        stream = tmp_path / "stream.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(fast),
            fmt="translated",
            crop_sizes=[24],
            strides=[0],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        convert._convert_to_npz_streaming(
            input_path=csv,
            output_path=str(stream),
            fmt="translated",
            crop_sizes=[24],
            strides=[0],
            num_classes=2,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            max_memory_bytes=1 * 1024 * 1024,
            pad=True,
        )
        fast_data = np.load(fast)
        stream_data = self._load_stream(str(stream))
        assert np.array_equal(fast_data["labels"], stream_data["labels"])
        assert np.array_equal(fast_data["translated"], stream_data["translated"])

    def test_streaming_unpadded_matches_padded_fast_path(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0,ATGCATGCATGC", "1,GGGG"])
        fast = tmp_path / "fast.npz"
        stream = tmp_path / "stream.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(fast),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            pad=True,
        )
        convert._convert_to_npz_streaming(
            input_path=csv,
            output_path=str(stream),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            num_classes=2,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            max_memory_bytes=1 * 1024 * 1024,
            pad=False,
        )
        fast_data = np.load(fast)
        stream_data = self._load_stream(str(stream))
        assert np.array_equal(fast_data["labels"], stream_data["labels"])
        # Pad each unpadded object array to the fast-path padded length.
        for i, crop in enumerate(stream_data["nucleotide"]):
            padded = np.concatenate(
                [crop, np.zeros((crop.shape[0], 12 - crop.shape[1]), dtype=crop.dtype)],
                axis=1,
            )
            assert np.array_equal(fast_data["nucleotide"][i], padded)

    def test_streaming_forces_multiple_batches(self, tmp_path: Path):
        csv = self._csv(
            tmp_path,
            ["0,ATGCATGCATGC", "1,GGGGGGGGGGGG", "0,TACGTACGTACG", "1,CCCCCCCCCCCC"],
        )
        fast = tmp_path / "fast.npz"
        stream = tmp_path / "stream.npz"
        convert._convert_to_npz(
            input_path=csv,
            output_path=str(fast),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            num_classes=2,
            num_workers=1,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
        )
        convert._convert_to_npz_streaming(
            input_path=csv,
            output_path=str(stream),
            fmt="nucleotide",
            crop_sizes=[12],
            strides=[0],
            num_classes=2,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            max_memory_bytes=64,
            pad=True,
        )
        fast_data = np.load(fast)
        stream_data = self._load_stream(str(stream))
        assert np.array_equal(fast_data["labels"], stream_data["labels"])
        assert np.array_equal(fast_data["nucleotide"], stream_data["nucleotide"])

    def test_streaming_unpadded(self, tmp_path: Path):
        csv = self._csv(tmp_path, ["0," + "A" * 25, "1," + "G" * 25])
        out = tmp_path / "out.npz"
        convert._convert_to_npz_streaming(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[20],
            strides=[10],
            num_classes=2,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            max_memory_bytes=64,
            pad=False,
        )
        data = np.load(out, allow_pickle=True)
        manifest = json.loads(str(data["_jaeger_manifest"].item()))
        assert manifest["padded"] is False
        stream_data = self._load_stream(str(out))
        assert stream_data["nucleotide"].dtype == object


class TestFeatureDtype:
    def test_resolve_feature_dtype_translated_integer(self):
        assert convert._resolve_feature_dtype(
            "translated",
            one_hot=False,
            dtype_arg="auto",
            codon_map_len=64,
            nucleotide_map={},
        ) == np.dtype(np.int8)

    def test_resolve_feature_dtype_dicodon_integer(self):
        assert convert._resolve_feature_dtype(
            "translated",
            one_hot=False,
            dtype_arg="auto",
            codon_map_len=4096,
            nucleotide_map={},
        ) == np.dtype(np.int16)

    def test_resolve_feature_dtype_onehot_ignored(self):
        assert convert._resolve_feature_dtype(
            "translated",
            one_hot=True,
            dtype_arg="auto",
            codon_map_len=64,
            nucleotide_map={},
        ) == np.dtype(np.float32)

    def test_resolve_feature_dtype_explicit(self):
        assert convert._resolve_feature_dtype(
            "translated",
            one_hot=False,
            dtype_arg="int16",
            codon_map_len=64,
            nucleotide_map={},
        ) == np.dtype(np.int16)

    def test_nucleotide_integer_is_int8(self):
        assert convert._resolve_feature_dtype(
            "nucleotide",
            one_hot=False,
            dtype_arg="auto",
            codon_map_len=None,
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
        ) == np.dtype(np.int8)

    def test_optimize_data_saves_int8_translated(self, tmp_path: Path):
        csv = tmp_path / "input.csv"
        csv.write_text("0,ATGCATGCATGCATGCATGCATGC\n1,GGGGGGGGGGGG\n")
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=str(csv),
            output_path=str(out),
            format="translated",
            crop_size=24,
            num_classes=2,
            num_workers=1,
            one_hot=False,
        )
        data = np.load(out, allow_pickle=True)
        assert data["translated"][0].dtype == np.dtype(np.int8)

    def test_optimize_data_dtype_int32_override(self, tmp_path: Path):
        csv = tmp_path / "input.csv"
        csv.write_text("0,ATGCATGCATGCATGCATGCATGC\n1,GGGGGGGGGGGG\n")
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=str(csv),
            output_path=str(out),
            format="translated",
            crop_size=24,
            num_classes=2,
            num_workers=1,
            one_hot=False,
            dtype="int32",
        )
        data = np.load(out, allow_pickle=True)
        assert data["translated"][0].dtype == np.dtype(np.int32)


class TestBalanceClasses:
    """Tests for class-balanced sharding (``balance_classes``)."""

    def _blocked_csv(self, tmp_path: Path, per_class: int = 30) -> str:
        """CSV with classes in contiguous blocks (worst-case ordering)."""
        lines = []
        for c in range(3):
            for _ in range(per_class):
                lines.append(f"{c}," + "ACGT" * 6)
        path = tmp_path / "input.csv"
        path.write_text("\n".join(lines))
        return str(path)

    @staticmethod
    def _max_run(labels: np.ndarray) -> int:
        best = cur = 1
        for i in range(1, len(labels)):
            cur = cur + 1 if labels[i] == labels[i - 1] else 1
            best = max(best, cur)
        return best

    def _stream_convert(self, csv: str, out: Path, balance: bool, seed: int = 42):
        convert._convert_to_npz_streaming(
            input_path=csv,
            output_path=str(out),
            fmt="nucleotide",
            crop_sizes=[24],
            strides=[0],
            num_classes=3,
            one_hot=False,
            pad_int=0,
            codon_map_name="codon_id",
            nucleotide_map={"A": 1, "G": 2, "T": 3, "C": 4, "N": 0},
            compress="default",
            max_memory_bytes=6000,  # ~30 rows per shard
            pad=True,
            balance_classes=balance,
            shuffle_seed=seed,
        )

    def _shard_labels(self, path: Path) -> list[np.ndarray]:
        data = np.load(path, allow_pickle=True)
        manifest = json.loads(str(data["_jaeger_manifest"].item()))
        return [data[f"labels_{i:05d}"] for i in range(manifest["num_shards"])]

    def test_shards_match_global_class_mix(self, tmp_path: Path):
        csv = self._blocked_csv(tmp_path)
        out = tmp_path / "out.npz"
        self._stream_convert(csv, out, balance=True)
        for labels in self._shard_labels(out):
            counts = np.bincount(labels, minlength=3)
            assert counts.max() - counts.min() <= 1

    def test_classes_interleaved_within_shard(self, tmp_path: Path):
        csv = self._blocked_csv(tmp_path)
        out = tmp_path / "out.npz"
        self._stream_convert(csv, out, balance=True)
        for labels in self._shard_labels(out):
            assert self._max_run(labels) <= 2

    def test_deterministic_with_same_seed(self, tmp_path: Path):
        csv = self._blocked_csv(tmp_path)
        out1, out2 = tmp_path / "a.npz", tmp_path / "b.npz"
        self._stream_convert(csv, out1, balance=True)
        self._stream_convert(csv, out2, balance=True)
        for l1, l2 in zip(self._shard_labels(out1), self._shard_labels(out2)):
            assert np.array_equal(l1, l2)

    def test_same_rows_as_unbalanced(self, tmp_path: Path):
        csv = self._blocked_csv(tmp_path)
        out_bal, out_plain = tmp_path / "bal.npz", tmp_path / "plain.npz"
        self._stream_convert(csv, out_bal, balance=True)
        self._stream_convert(csv, out_plain, balance=False)
        bal = np.concatenate(self._shard_labels(out_bal))
        plain = np.concatenate(self._shard_labels(out_plain))
        assert np.array_equal(np.sort(bal), np.sort(plain))

    def test_flag_off_preserves_csv_order(self, tmp_path: Path):
        csv = self._blocked_csv(tmp_path, per_class=10)
        out = tmp_path / "out.npz"
        self._stream_convert(csv, out, balance=False)
        labels = np.concatenate(self._shard_labels(out))
        assert labels.tolist() == [0] * 10 + [1] * 10 + [2] * 10

    def test_fast_path_interleaves(self, tmp_path: Path):
        csv = self._blocked_csv(tmp_path)
        out = tmp_path / "out.npz"
        convert.convert_dataset(
            input_path=csv,
            output_path=str(out),
            format="nucleotide",
            crop_size=24,
            num_classes=3,
            num_workers=1,
            balance_classes=True,
        )
        data = np.load(out)
        counts = np.bincount(data["labels"], minlength=3)
        assert counts.tolist() == [30, 30, 30]
        assert self._max_run(data["labels"]) <= 2
