"""Shared pytest fixtures for the Jaeger test suite."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# Keep TF reasonably quiet during tests.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


@pytest.fixture
def tmp_path_with_cleanup(tmp_path: Path):
    """Yield a temporary path and clean it up after the test."""
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def dna_sequence() -> str:
    return "ATGCATGCATGCATGCATGCATGC"


@pytest.fixture
def labeled_fasta_path(tmp_path: Path) -> str:
    path = tmp_path / "labeled.fasta"
    path.write_text(
        ">seq1__class=0\n"
        "ATGCATGCATGCATGCATGCATGC\n"
        ">seq2__class=1\n"
        "GCTAGCTAGCTAGCTAGCTAGCTA\n"
    )
    return str(path)


@pytest.fixture
def unlabeled_fasta_path(tmp_path: Path) -> str:
    path = tmp_path / "unlabeled.fasta"
    path.write_text(
        ">seq1\n"
        "ATGCATGCATGCATGCATGCATGCATGCATGC\n"
        ">seq2\n"
        "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA\n"
    )
    return str(path)


@pytest.fixture
def simple_csv_path(tmp_path: Path) -> str:
    path = tmp_path / "simple.csv"
    path.write_text("0,ATGCATGCATGCATGCATGCATGC,seq1\n")
    return str(path)


@pytest.fixture
def random_logits() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(4, 3)).astype(np.float32)


@pytest.fixture
def small_onehot_sequence() -> np.ndarray:
    """One-hot encoded sequence of length 9 over 4 nucleotides."""
    tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.int32)
    identity = np.identity(4, dtype=np.float32)
    return identity[tokens]
