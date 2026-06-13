"""Tests for jaeger.dataops.dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaeger.dataops import dataset


class TestReadSequences:
    def test_read_fasta(self, unlabeled_fasta_path: str):
        records = dataset.read_sequences(
            Path(unlabeled_fasta_path), intype="FASTA", class_id=1
        )
        assert len(records) == 2
        assert records[0][2] == 1

    def test_read_csv(self, tmp_path: Path):
        csv = tmp_path / "in.csv"
        csv.write_text("0,ATGCATGC,seq1\n1,GCTAGCTA,seq2\n")
        records = dataset.read_sequences(
            csv, intype="CSV", seq_col=1, class_col=0
        )
        assert len(records) == 2
        assert records[0][1] == "ATGCATGC"

    def test_read_invalid_type(self, tmp_path: Path):
        with pytest.raises(ValueError):
            dataset.read_sequences(tmp_path / "x.txt", intype="TXT")


class TestGenerateFragments:
    def test_generate_fragments(self):
        records = [("seq1", "A" * 100, 0)]
        frags = dataset.generate_fragments(records, frag_len=30, overlap=10)
        assert len(frags) > 0
        assert all(len(f[1]) <= 30 for f in frags)

    def test_generate_fragments_too_short(self):
        records = [("seq1", "A" * 10, 0)]
        frags = dataset.generate_fragments(records, frag_len=30, overlap=10)
        assert len(frags) == 0


class TestSplitDataset:
    def test_split_dataset(self):
        records = [(f"seq{i}", "A" * 50, 0) for i in range(10)]
        train, val, test = dataset.split_dataset(records, 0.6, 0.2, 0.2)
        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2


class TestWriteOutput:
    def test_write_csv_output(self, tmp_path: Path):
        records = [("s1", "ATGC", 0), ("s2", "GCTA", 1)]
        dataset.write_output(records, records[:1], records[1:], tmp_path, outtype="CSV")
        # write_output places files next to the output directory, not inside it.
        assert (tmp_path.parent / f"{tmp_path.name}_train.csv").exists()

    def test_write_fasta_output(self, tmp_path: Path):
        records = [("s1", "ATGC", 0)]
        dataset.write_output(records, [], [], tmp_path, outtype="FASTA")
        assert (tmp_path.parent / f"{tmp_path.name}_train.fasta").exists()

    def test_write_output_invalid_type(self, tmp_path: Path):
        records = [("s1", "ATGC", 0)]
        with pytest.raises(ValueError):
            dataset.write_output(records, [], [], tmp_path, outtype="TXT")
