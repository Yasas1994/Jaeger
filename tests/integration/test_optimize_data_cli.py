"""CLI integration tests for `jaeger utils optimize-data`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from jaeger.cli import main


@pytest.fixture
def tiny_csv(tmp_path: Path) -> str:
    path = tmp_path / "train.csv"
    path.write_text("0,ATGCATGCATGCATGCATGCATGC\n1,GGGGGGGGGGGG\n")
    return str(path)


def test_optimize_data_help():
    runner = CliRunner()
    result = runner.invoke(main, ["utils", "optimize-data", "--help"])
    assert result.exit_code == 0
    assert "nucleotide" in result.output
    assert "translated" in result.output
    assert "both" in result.output


def test_optimize_data_translated(tiny_csv: str, tmp_path: Path):
    out = tmp_path / "train.npz"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "utils",
            "optimize-data",
            "-i",
            tiny_csv,
            "-o",
            str(out),
            "--format",
            "translated",
            "--crop-size",
            "24",
            "--num-classes",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = np.load(out)
    assert "translated" in data
    assert "labels" in data


def test_optimize_data_nucleotide_onehot(tiny_csv: str, tmp_path: Path):
    out = tmp_path / "train.npz"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "utils",
            "optimize-data",
            "-i",
            tiny_csv,
            "-o",
            str(out),
            "--format",
            "nucleotide",
            "--crop-size",
            "12",
            "--one-hot",
            "--pad",
            "--num-classes",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    data = np.load(out)
    assert data["nucleotide"].ndim == 4
    assert data["nucleotide"].dtype == np.float32
