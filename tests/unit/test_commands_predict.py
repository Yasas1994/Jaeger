"""Tests for jaeger.commands.predict helpers."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from jaeger.commands.predict import _save_auxiliary_outputs


def _fake_y_pred() -> dict[str, np.ndarray]:
    return {
        "prediction": np.zeros((2, 2), dtype=np.float32),
        "embedding": np.ones((2, 8), dtype=np.float32),
        "nmd": np.ones((2, 4), dtype=np.float32),
        "meta_0": np.array(["contig_1", "contig_2"], dtype=object),
    }


def test_default_does_not_save_vectors(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=False,
        save_nmd=False,
    )
    assert not (tmp_path / "sample_embedding.npz").exists()
    assert not (tmp_path / "sample_nmd.npz").exists()


def test_save_embedding_flag_writes_file(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=True,
        save_nmd=False,
    )
    assert (tmp_path / "sample_embedding.npz").exists()
    assert not (tmp_path / "sample_nmd.npz").exists()

    loaded = np.load(tmp_path / "sample_embedding.npz", allow_pickle=True)
    assert np.array_equal(loaded["embedding"], y_pred["embedding"])
    assert np.array_equal(loaded["headers"], y_pred["meta_0"])


def test_save_nmd_flag_writes_file(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=False,
        save_nmd=True,
    )
    assert not (tmp_path / "sample_embedding.npz").exists()
    assert (tmp_path / "sample_nmd.npz").exists()

    loaded = np.load(tmp_path / "sample_nmd.npz", allow_pickle=True)
    assert np.array_equal(loaded["embedding"], y_pred["nmd"])
    assert np.array_equal(loaded["headers"], y_pred["meta_0"])


def test_save_both_flags_writes_both_files(tmp_path: Path) -> None:
    y_pred = _fake_y_pred()
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=True,
        save_nmd=True,
    )
    assert (tmp_path / "sample_embedding.npz").exists()
    assert (tmp_path / "sample_nmd.npz").exists()


def test_missing_outputs_are_ignored(tmp_path: Path) -> None:
    """If the model does not expose embedding/nmd, flags must not crash."""
    y_pred = {
        "prediction": np.zeros((2, 2), dtype=np.float32),
        "meta_0": np.array(["contig_1", "contig_2"], dtype=object),
    }
    _save_auxiliary_outputs(
        y_pred,
        tmp_path,
        "sample",
        save_embedding=True,
        save_nmd=True,
    )
    assert not (tmp_path / "sample_embedding.npz").exists()
    assert not (tmp_path / "sample_nmd.npz").exists()
