"""Tests for jaeger.data.loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaeger.data import loaders


@pytest.fixture
def numpy_raw_file(tmp_path: Path):
    path = tmp_path / "raw.npz"
    # 2 samples, int8 DNA tokens + one-hot labels; crop_size matches sequence length.
    sequences = np.array([[0, 1, 2, 3] * 7] * 2, dtype=np.int8)[:, :26]
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    lengths = np.array([26, 26], dtype=np.int32)
    np.savez(path, sequences=sequences, labels=labels, lengths=lengths)
    return str(path)


@pytest.fixture
def numpy_full_file(tmp_path: Path):
    path = tmp_path / "full.npz"
    # crop_size=27 gives seq_len=27//3-1=8, matching the (6, 8) shape.
    translated = np.random.randint(0, 65, size=(2, 6, 8), dtype=np.int32)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.savez(path, translated=translated, label=labels)
    return str(path)


class TestNumpyRawLoader:
    def test_load_numpy_raw_dataset(self, numpy_raw_file: str):
        ds = loaders._load_numpy_raw_dataset(
            numpy_raw_file, crop_size=26, ngram_width=3, num_classes=2
        )
        batch = next(iter(ds.batch(2)))
        assert "translated" in batch[0]
        assert batch[1].shape.as_list() == [2, 2]


class TestNumpyRawVariableLoader:
    def test_load_numpy_raw_variable_dataset(self, numpy_raw_file: str):
        ds = loaders._load_numpy_raw_variable_dataset(
            numpy_raw_file, ngram_width=3, num_classes=2
        )
        batch = next(iter(ds.batch(2)))
        assert "translated" in batch[0]


class TestNumpyFullLoader:
    def test_load_numpy_full_dataset(self, numpy_full_file: str):
        ds = loaders._load_numpy_full_dataset(
            numpy_full_file, input_type="translated", use_embedding_layer=True, codon_depth=65, crop_size=27
        )
        batch = next(iter(ds.batch(2)))
        assert "translated" in batch[0]
        assert batch[1].shape.as_list() == [2, 2]
