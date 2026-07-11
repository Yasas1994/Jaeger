"""Tests for jaeger.commands.predict helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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

    # headers are object-dtype strings, so allow_pickle is required
    with np.load(tmp_path / "sample_embedding.npz", allow_pickle=True) as loaded:
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

    # headers are object-dtype strings, so allow_pickle is required
    with np.load(tmp_path / "sample_nmd.npz", allow_pickle=True) as loaded:
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

    # headers are object-dtype strings, so allow_pickle is required
    with np.load(tmp_path / "sample_embedding.npz", allow_pickle=True) as loaded:
        assert np.array_equal(loaded["embedding"], y_pred["embedding"])
        assert np.array_equal(loaded["headers"], y_pred["meta_0"])
    with np.load(tmp_path / "sample_nmd.npz", allow_pickle=True) as loaded:
        assert np.array_equal(loaded["embedding"], y_pred["nmd"])
        assert np.array_equal(loaded["headers"], y_pred["meta_0"])


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


def test_write_prediction_outputs_writes_both_contigs(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression test: the post-processing helper must handle multi-contig y_pred."""
    import pandas as pd
    from unittest.mock import MagicMock

    from jaeger.commands.predict import _write_prediction_outputs

    written: dict = {}

    def fake_write_output(data, **kw):
        written["data"] = data
        written["kwargs"] = kw
        return len(data["headers"])

    monkeypatch.setattr("jaeger.postprocess.collect.write_output", fake_write_output)

    fake_model = MagicMock()
    fake_model.class_map = {
        "class": ["bacteria", "phage", "eukarya", "archaea", "plasmid", "virus"],
        "index": [0, 1, 2, 3, 4, 5],
        "num_classes": 6,
    }

    y_pred = {
        "prediction": np.zeros((2, 6), dtype=np.float32),
        "meta_0": np.array(["long_contig", "short_contig"], dtype=object),
        "meta_2": np.array([1, 1], dtype=np.int32),
        "meta_4": np.array([2500, 1800], dtype=np.int32),
        "meta_5": np.array([0.25, 0.25], dtype=np.float32),
        "meta_6": np.array([0.25, 0.25], dtype=np.float32),
        "meta_7": np.array([0.25, 0.25], dtype=np.float32),
        "meta_8": np.array([0.25, 0.25], dtype=np.float32),
        "meta_9": np.array([0.0, 0.0], dtype=np.float32),
    }

    _write_prediction_outputs(
        y_pred,
        model=fake_model,
        model_name="test_model",
        model_info={"graph": tmp_path / "graph"},
        input_file_path=tmp_path / "input.fasta",
        output_table_path=tmp_path / "out.tsv",
        output_phage_table_path=tmp_path / "out_phages.tsv",
        file_base="input",
        OUTPUT_DIR=tmp_path,
        term_repeats=pd.DataFrame(
            {"contig_id": [], "terminal_repeats": [], "repeat_length": []}
        ),
        num=2,
        logger=MagicMock(),
        fsize=2000,
        rc=0.5,
        pc=1,
    )

    assert "data" in written
    assert set(written["data"]["headers"]) == {"long_contig", "short_contig"}
    assert written["kwargs"]["output_table_path"] == tmp_path / "out.tsv"


def test_concat_predictions_concatenates_matching_keys() -> None:
    from jaeger.commands.predict import _concat_predictions

    a = {"prediction": np.zeros((2, 3)), "meta_0": np.array(["c1", "c2"])}
    b = {"prediction": np.ones((1, 3)), "meta_0": np.array(["c3"])}
    out = _concat_predictions(a, b)
    assert out["prediction"].shape == (3, 3)
    assert out["meta_0"].tolist() == ["c1", "c2", "c3"]


def test_concat_predictions_empty_second_pass_returns_first() -> None:
    # Two-pass inference where no contigs fall in the short length range:
    # the second pass returns an empty dict and must not crash the merge.
    from jaeger.commands.predict import _concat_predictions

    a = {"prediction": np.zeros((2, 3)), "meta_0": np.array(["c1", "c2"])}
    out = _concat_predictions(a, {})
    assert out["prediction"].shape == (2, 3)
    assert out["meta_0"].tolist() == ["c1", "c2"]


def test_concat_predictions_empty_first_pass_returns_second() -> None:
    # All contigs in the short length range: first pass is empty and the
    # second pass predictions must survive (not be replaced by {}).
    from jaeger.commands.predict import _concat_predictions

    b = {"prediction": np.ones((1, 3)), "meta_0": np.array(["c3"])}
    out = _concat_predictions({}, b)
    assert out["prediction"].shape == (1, 3)
    assert out["meta_0"].tolist() == ["c3"]
