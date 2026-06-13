"""Tests for jaeger.nnlib.conversion."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
import tensorflow as tf

from jaeger.nnlib import conversion


LOGGER = logging.getLogger("jaeger")


@pytest.fixture
def dummy_saved_model(tmp_path: Path):
    """Create a tiny SavedModel with a serving signature."""
    class Model(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32)])
        def serving_default(self, inputs):
            return {"prediction": inputs * 2.0}

    export_dir = tmp_path / "dummy_graph"
    tf.saved_model.save(Model(), str(export_dir))
    return str(export_dir)


class TestResolveModel:
    def test_resolve_none(self):
        assert conversion._resolve_model("nonexistent_model_name_xyz") is None


class TestConvertXla:
    def test_convert_xla(self, dummy_saved_model: str, tmp_path: Path):
        output_dir = tmp_path / "xla_out"
        conversion._convert_xla(
            graph_dir=Path(dummy_saved_model),
            output_dir=output_dir,
            model_name="dummy",
            log=LOGGER,
        )
        assert (output_dir / "saved_model.pb").exists()
        loaded = tf.saved_model.load(str(output_dir))
        out = loaded.signatures["serving_default"](tf.constant([[1.0, 2.0, 3.0, 4.0]]))
        assert "prediction" in out
        assert out["prediction"].numpy().tolist()[0] == pytest.approx([2.0, 4.0, 6.0, 8.0])


@pytest.mark.skipif(shutil.which("mmseqs") is not None, reason="skipping conversion dispatcher smoke")
class TestConvertGraphDispatcher:
    def test_unknown_mode(self, tmp_path: Path):
        with pytest.raises(ValueError):
            conversion.convert_graph("dummy", str(tmp_path / "out"), mode="unknown")
