"""Tests for jaeger.dataops.ood."""

from __future__ import annotations

from pathlib import Path


from jaeger.dataops import ood


class TestShuffleCoreFasta:
    def test_dinuc_shuffle_fasta_to_fasta(self, tmp_path: Path):
        inp = tmp_path / "in.fasta"
        inp.write_text(">seq1\nATGCATGCATGCATGC\n")
        out = tmp_path / "out.fasta"
        ood.shuffle_core(
            input=str(inp),
            output=str(out),
            itype="FASTA",
            otype="FASTA",
            dinuc=True,
            num_tandem_repeats=0,
        )
        assert out.exists()
        text = out.read_text()
        # Expect original + shuffled entries (and class label appended).
        assert text.count(">") >= 2

    def test_kmer_shuffle_fasta_to_csv(self, tmp_path: Path):
        inp = tmp_path / "in.fasta"
        inp.write_text(">seq1\nATGCATGCATGCATGC\n")
        out = tmp_path / "out.csv"
        ood.shuffle_core(
            input=str(inp),
            output=str(out),
            itype="FASTA",
            otype="CSV",
            dinuc=False,
            k=1,
            num_tandem_repeats=0,
        )
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 2

    def test_tandem_repeats_added(self, tmp_path: Path):
        inp = tmp_path / "in.fasta"
        inp.write_text(">seq1\nATGCATGCATGCATGC\n")
        out = tmp_path / "out.fasta"
        ood.shuffle_core(
            input=str(inp),
            output=str(out),
            itype="FASTA",
            otype="FASTA",
            dinuc=False,
            k=1,
            num_tandem_repeats=2,
        )
        text = out.read_text()
        assert "tandem_repeat" in text


class TestShuffleCoreCsv:
    def test_csv_to_csv(self, tmp_path: Path):
        inp = tmp_path / "in.csv"
        inp.write_text("0,ATGCATGCATGCATGC,seq1\n")
        out = tmp_path / "out.csv"
        ood.shuffle_core(
            input=str(inp),
            output=str(out),
            itype="CSV",
            otype="CSV",
            dinuc=True,
            num_tandem_repeats=0,
        )
        assert out.exists()

    def test_csv_to_fasta(self, tmp_path: Path):
        inp = tmp_path / "in.csv"
        inp.write_text("0,ATGCATGCATGCATGC,seq1\n")
        out = tmp_path / "out.fasta"
        ood.shuffle_core(
            input=str(inp),
            output=str(out),
            itype="CSV",
            otype="FASTA",
            dinuc=True,
            num_tandem_repeats=0,
        )
        assert out.exists()
        assert ">" in out.read_text()
