"""Tests for jaeger.dataops.split."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaeger.dataops import split


class TestSplitCore:
    def test_split_core_sequential(self, tmp_path: Path):
        fasta = tmp_path / "in.fasta"
        fasta.write_text(">genome\n" + "ATGC" * 500 + "\n")
        out = tmp_path / "out.fasta"
        split.split_core(
            input=str(fasta),
            output=str(out),
            minlen=50,
            maxlen=100,
            overlap=0,
            coverage=None,
            circular=False,
            max_n_prop=0.3,
            seed=42,
            shuffle=False,
        )
        assert out.exists()
        text = out.read_text()
        assert ">genome_frag" in text

    def test_split_core_coverage(self, tmp_path: Path):
        fasta = tmp_path / "in.fasta"
        fasta.write_text(">genome\n" + "ATGC" * 500 + "\n")
        out = tmp_path / "out.fasta"
        split.split_core(
            input=str(fasta),
            output=str(out),
            minlen=50,
            maxlen=100,
            overlap=0,
            coverage=1,
            circular=False,
            max_n_prop=0.3,
            seed=42,
            shuffle=False,
        )
        assert out.exists()

    def test_invalid_lengths(self, tmp_path: Path):
        with pytest.raises(ValueError):
            split.split_core(
                input=str(tmp_path / "in.fasta"),
                output=str(tmp_path / "out.fasta"),
                minlen=0,
                maxlen=10,
            )
