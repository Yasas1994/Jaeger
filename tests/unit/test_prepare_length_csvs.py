# tests/unit/test_prepare_length_csvs.py
import csv

from scripts.prepare_length_csvs import main


def test_prepare_length_csvs_custom_lengths(tmp_path):
    fasta = tmp_path / "input.fasta"
    tsv = tmp_path / "labels.tsv"
    out_dir = tmp_path / "out"

    fasta.write_text(">seq1\n" + "A" * 600 + "\n>seq2\n" + "C" * 600 + "\n")
    tsv.write_text("seq1\tignore\tchromosome\nseq2\tignore\tvirus\n")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_length_csvs.py",
            "--fasta",
            str(fasta),
            "--tsv",
            str(tsv),
            "--out-dir",
            str(out_dir),
            "--lengths",
            "300",
            "600",
            "--val-frac",
            "0.5",
        ]
        main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "train_300.csv").exists()
    assert (out_dir / "val_300.csv").exists()
    assert (out_dir / "train_600.csv").exists()
    assert (out_dir / "val_600.csv").exists()

    with open(out_dir / "train_600.csv") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1
    assert len(rows[0][1]) == 600
