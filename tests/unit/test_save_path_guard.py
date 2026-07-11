from __future__ import annotations

import pytest

from jaeger.nnlib.builder import DynamicModelBuilder


def _make_builder(save_path, force: bool) -> DynamicModelBuilder:
    config = {
        "model": {"name": "jaeger"},
        "training": {"model_saving": {"path": str(save_path)}},
        "force": force,
    }
    return DynamicModelBuilder(config)


def test_non_empty_save_dir_without_force_aborts(tmp_path):
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    (save_dir / "existing.weights.h5").write_bytes(b"old model")

    builder = _make_builder(save_dir, force=False)

    with pytest.raises(SystemExit):
        builder.ensure_save_path_available()
    # directory contents must be untouched
    assert (save_dir / "existing.weights.h5").exists()


def test_non_empty_save_dir_with_force_is_cleared(tmp_path):
    save_dir = tmp_path / "model"
    save_dir.mkdir()
    (save_dir / "existing.weights.h5").write_bytes(b"old model")

    builder = _make_builder(save_dir, force=True)

    builder.ensure_save_path_available()  # must not raise
    path, _ = builder._prepare_save_path(clear=True)
    assert path == save_dir
    assert not any(save_dir.iterdir())


def test_empty_save_dir_never_aborts(tmp_path):
    for force in (False, True):
        save_dir = tmp_path / ("model_force" if force else "model_noforce")
        builder = _make_builder(save_dir, force=force)
        builder.ensure_save_path_available()  # must not raise
        path, _ = builder._prepare_save_path(clear=True)
        assert path.exists()
