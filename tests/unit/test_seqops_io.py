"""Tests for jaeger.seqops.io."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaeger.seqops import io as seqio


class TestReadWriteFasta:
    def test_write_and_read_fasta(self, tmp_path: Path):
        out = tmp_path / "out.fasta"
        records = [
            {"id": "seq1", "seq": "ATGC" * 10},
            {"id": "seq2", "seq": "GCTA" * 10},
        ]
        seqio.write_fasta(records, out)
        read = seqio.read_sequences(str(out), input_type="FASTA")
        assert len(read) == 2
        assert read[0]["id"] == "seq1"
        assert read[0]["seq"] == records[0]["seq"]

    def test_write_fasta_entry(self, tmp_path: Path):
        out = tmp_path / "entry.fasta"
        with open(out, "w") as fh:
            seqio.write_fasta_entry(fh, "test", "ATGC" * 20, label=1)
        content = out.read_text()
        assert ">test__class=1" in content
        assert "ATGC" in content

    def test_write_fasta_record_with_label(self, tmp_path: Path):
        out = tmp_path / "record.fasta"
        with open(out, "w") as fh:
            seqio.write_fasta_record(fh, "h", "ATGC", label="1")
        assert out.read_text() == ">h\nATGC,1\n"

    def test_read_csv(self, tmp_path: Path):
        csv = tmp_path / "input.csv"
        csv.write_text("id1,ATGC,0\nid2,GCTA,1\n")
        records = seqio.read_sequences(str(csv), input_type="CSV")
        assert len(records) == 2
        assert records[0] == {"id": "id1", "seq": "ATGC", "label": "0"}

    def test_read_csv_no_label(self, tmp_path: Path):
        csv = tmp_path / "input.csv"
        csv.write_text("id1,ATGC\n")
        records = seqio.read_sequences(str(csv), input_type="CSV")
        assert records[0]["label"] is None

    def test_read_invalid_type(self, tmp_path: Path):
        with pytest.raises(ValueError):
            seqio.read_sequences(str(tmp_path / "x.txt"), input_type="TXT")


class TestFragmentGenerators:
    def test_fragment_generator(self, unlabeled_fasta_path: str):
        frags = list(
            seqio.fragment_generator(
                unlabeled_fasta_path,
                fragsize=16,
                stride=16,
                num=2,
                no_progress=True,
                dustmask=False,
            )
        )
        assert len(frags) > 0
        parts = frags[0].split(",")
        assert len(parts) == 11
        assert len(parts[0]) == 16

    def test_fragment_generator_lib(self, unlabeled_fasta_path: str):
        frags = list(
            seqio.fragment_generator_lib(
                unlabeled_fasta_path, fragsize=16, stride=16, num=2
            )
        )
        assert len(frags) > 0
        parts = frags[0].split(",")
        assert len(parts) == 6


class TestValidateFastaEntries:
    def test_validate_ok(self, unlabeled_fasta_path: str):
        count = seqio.validate_fasta_entries(unlabeled_fasta_path, min_len=1)
        assert count == 2

    def test_validate_too_short(self, tmp_path: Path):
        fasta = tmp_path / "short.fasta"
        fasta.write_text(">s\nA\n")
        with pytest.raises(Exception):
            seqio.validate_fasta_entries(str(fasta), min_len=10)


class TestWriteFastaFromResults:
    def test_write_fasta_from_results(self, tmp_path: Path):
        import pandas as pd

        df = pd.DataFrame(
            {
                "reliability_score": [0.9, 0.3, 0.8],
                "prediction": [1, 0, 1],
                "sequence": ["ATGC", "GCTA", "TTTT"],
                "index": [0, 1, 2],
            }
        )
        data = {"contig_1": (df, 0, 12)}
        out = tmp_path / "phages.fasta"
        seqio.write_fasta_from_results(
            data, str(out), reliability_cutoff=0.5, phage_score=1
        )
        text = out.read_text()
        assert ">contig_1_0" in text
        assert ">contig_1_2" in text
        assert ">contig_1_1" not in text
