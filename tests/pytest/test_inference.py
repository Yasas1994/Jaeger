"""Tests for the inference pipeline (InferModel and _load_string_processor_config)."""

from pathlib import Path

import pytest
import tensorflow as tf
import yaml

# Skip all tests if tensorflow is not available
try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")


class TestLoadStringProcessorConfig:
    """Test _load_string_processor_config fixes for input_type and seq_onehot."""

    def test_input_type_extracted_from_embedding_type(self, tmp_path):
        """input_type should be inferred from embedding.type when not explicitly set."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None, 64],
                    "frames": 6,
                    "strands": 2,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        # Create a dummy SavedModel
        graph_dir = tmp_path / "test_graph"
        _create_dummy_savedmodel(graph_dir)

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        assert model.string_processor_config["input_type"] == "translated"

    def test_input_type_defaults_to_translated(self, tmp_path):
        """input_type should default to 'translated' if embedding.type is missing."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "input_shape": [6, None, 64],
                    "frames": 6,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        graph_dir = tmp_path / "test_graph"
        _create_dummy_savedmodel(graph_dir)

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        assert model.string_processor_config["input_type"] == "translated"

    def test_seq_onehot_inferred_from_input_shape(self, tmp_path):
        """seq_onehot should be inferred True when input_shape has depth > 1."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None, 64],  # 3 dims = one-hot
                    "frames": 6,
                    "strands": 2,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        graph_dir = tmp_path / "test_graph"
        _create_dummy_savedmodel(graph_dir)

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        assert model.string_processor_config["seq_onehot"] is True
        assert model.string_processor_config["codon_depth"] == 64

    def test_seq_onehot_inferred_false_for_embedding_lookup(self, tmp_path):
        """seq_onehot should be inferred False when input_shape has only 2 dims."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None],  # 2 dims = embedding lookup
                    "frames": 6,
                    "strands": 2,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        graph_dir = tmp_path / "test_graph"
        _create_dummy_savedmodel(graph_dir)

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        assert model.string_processor_config["seq_onehot"] is False
        assert model.string_processor_config["codon_depth"] == 1

    def test_explicit_seq_onehot_overrides_inference(self, tmp_path):
        """Explicitly set seq_onehot should not be overridden by input_shape inference."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None, 64],
                    "frames": 6,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                    "seq_onehot": False,  # explicitly False
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        graph_dir = tmp_path / "test_graph"
        _create_dummy_savedmodel(graph_dir)

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        assert model.string_processor_config["seq_onehot"] is False
        assert model.string_processor_config["codon_depth"] == 1

    def test_path_as_string_accepted(self, tmp_path):
        """_load_string_processor_config should accept string paths, not just Path objects."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None, 64],
                    "frames": 6,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        graph_dir = tmp_path / "test_graph"
        _create_dummy_savedmodel(graph_dir)

        # Pass paths as strings
        model = InferModel(
            {
                "graph": str(graph_dir),
                "classes": str(classes_file),
                "project": str(project_file),
            }
        )

        assert model.string_processor_config["input_type"] == "translated"


class TestInferModelPrediction:
    """Test InferModel prediction with SavedModel signatures."""

    def test_inference_fn_called_with_keyword_inputs(self, tmp_path):
        """_predict_step should call inference_fn with inputs= keyword argument."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None, 4],
                    "frames": 6,
                    "strands": 2,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                    "seq_onehot": True,
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        # Create a SavedModel that expects inputs= keyword
        graph_dir = tmp_path / "test_graph"
        _create_savedmodel_with_keyword_input(graph_dir, input_shape=(None, 6, None, 4))

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        # Create dummy input batch
        dummy_input = {
            "translated": tf.random.normal([2, 6, 10, 4]),
        }

        # This should not raise TypeError about positional arguments
        result = model._predict_step(dummy_input)
        assert "prediction" in result or "output" in result


class TestLegacyAndNewPipeline:
    """Test that both legacy and new inference pipelines work."""

    def test_legacy_model_multiple_inputs(self, tmp_path):
        """Legacy models with multiple inputs should still load correctly."""
        from jaeger.nnlib.inference import InferModel

        # Create a legacy-style SavedModel with multiple inputs
        graph_dir = tmp_path / "legacy_graph"
        _create_legacy_savedmodel(graph_dir)

        classes_file = tmp_path / "classes.yaml"
        classes_file.write_text(
            yaml.safe_dump(
                {
                    "classes": [
                        {"class": "bacteria", "label": 0},
                        {"class": "phage", "label": 1},
                    ]
                }
            )
        )

        # Legacy models don't have project.yaml
        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
            }
        )

        # Should load without error
        assert model.loaded_model is not None
        assert "serving_default" in model.loaded_model.signatures

        # Legacy models have multiple inputs
        sig = model.loaded_model.signatures["serving_default"]
        inputs = sig.structured_input_signature[1]
        assert len(inputs) > 1, "Legacy model should have multiple inputs"

    def test_new_model_single_input(self, tmp_path):
        """New models with single 'inputs' keyword should work."""
        from jaeger.nnlib.inference import InferModel

        project_file = tmp_path / "project.yaml"
        classes_file = tmp_path / "classes.yaml"

        project = {
            "model": {
                "embedding": {
                    "type": "translated",
                    "input_shape": [6, None, 4],
                    "frames": 6,
                    "strands": 2,
                },
                "string_processor": {
                    "codon": "CODON",
                    "codon_id": "CODON_ID",
                    "seq_onehot": True,
                },
            }
        }
        project_file.write_text(yaml.safe_dump(project))
        classes_file.write_text(yaml.safe_dump({"classes": []}))

        graph_dir = tmp_path / "test_graph"
        _create_savedmodel_with_keyword_input(graph_dir, input_shape=(None, 6, None, 4))

        model = InferModel(
            {
                "graph": graph_dir,
                "classes": classes_file,
                "project": project_file,
            }
        )

        # Verify signature
        sig = model.loaded_model.signatures["serving_default"]
        input_spec = sig.structured_input_signature[1].get("inputs")
        assert input_spec is not None


# ─── Helpers ────────────────────────────────────────────────────────────────


def _create_dummy_savedmodel(export_dir: Path):
    """Create a minimal SavedModel for testing."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    class DummyModel(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=(None, 6, None, 64), dtype=tf.float32, name="inputs"
                )
            ]
        )
        def serving_default(self, inputs):
            return {
                "prediction": tf.zeros([tf.shape(inputs)[0], 2]),
                "reliability": tf.zeros([tf.shape(inputs)[0], 1]),
            }

    model = DummyModel()
    tf.saved_model.save(
        model, str(export_dir), signatures={"serving_default": model.serving_default}
    )


def _create_savedmodel_with_keyword_input(export_dir: Path, input_shape):
    """Create a SavedModel that expects inputs= keyword argument."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    class KeywordInputModel(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="inputs")
            ]
        )
        def serving_default(self, inputs):
            batch_size = tf.shape(inputs)[0]
            return {
                "prediction": tf.zeros([batch_size, 2]),
                "reliability": tf.zeros([batch_size, 1]),
            }

    model = KeywordInputModel()
    tf.saved_model.save(
        model, str(export_dir), signatures={"serving_default": model.serving_default}
    )


def _create_legacy_savedmodel(export_dir: Path):
    """Create a legacy-style SavedModel with multiple named inputs."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    class LegacyModel(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="inputs_1"),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="inputs_2"),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="inputs_3"),
            ]
        )
        def serving_default(self, inputs_1, inputs_2, inputs_3):
            batch_size = tf.shape(inputs_1)[0]
            return {
                "output": tf.zeros([batch_size, 4]),
                "embedding": tf.zeros([batch_size, 128]),
            }

    model = LegacyModel()
    tf.saved_model.save(
        model, str(export_dir), signatures={"serving_default": model.serving_default}
    )
