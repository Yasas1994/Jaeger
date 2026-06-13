import sys
import tensorflow as tf
import logging
from pathlib import Path
import shutil
from jaeger.utils.misc import AvailableModels

logger = logging.getLogger("Jaeger")


class EnsembleModel(tf.Module):
    def __init__(self, model_paths, method="mean"):
        super().__init__()
        self.method = method
        self.models = [tf.saved_model.load(p) for p in model_paths]
        self._signatures = [m.signatures["serving_default"] for m in self.models]

    @tf.function
    def __call__(self, inputs):
        """
        Combine outputs from all sub-models.

        Supported methods:
          - mean: average logits/other tensors.
          - sum:  sum logits/other tensors.
          - mv:   majority vote over ``prediction`` logits; other tensors are
                  averaged over models that voted for the winning class.
          - none: fallback to mean (keeps a valid SavedModel output).
        """
        outputs = [sig(inputs) for sig in self._signatures]

        # Use only keys common to all model outputs.
        common_keys = set(outputs[0].keys())
        for out in outputs[1:]:
            common_keys &= set(out.keys())

        if self.method in ("mean", "sum", "none"):
            combined = {}
            for key in common_keys:
                stacked = tf.stack([out[key] for out in outputs], axis=0)
                if self.method == "sum":
                    combined[key] = tf.reduce_sum(stacked, axis=0)
                else:
                    combined[key] = tf.reduce_mean(stacked, axis=0)
            return combined

        # Majority vote over the prediction head.
        if "prediction" not in common_keys:
            raise ValueError("Majority voting requires a 'prediction' output key")

        preds = tf.stack([out["prediction"] for out in outputs], axis=0)
        argmaxes = tf.argmax(preds, axis=-1, output_type=tf.int32)
        n_classes = tf.shape(preds)[-1]

        votes = tf.one_hot(argmaxes, depth=n_classes)
        vote_counts = tf.reduce_sum(votes, axis=0)
        majority_class = tf.argmax(vote_counts, axis=-1, output_type=tf.int32)

        majority_mask = tf.one_hot(majority_class, depth=n_classes)
        majority_mask = tf.expand_dims(majority_mask, axis=0)
        masked = preds * tf.cast(majority_mask, preds.dtype)

        sums = tf.reduce_sum(masked, axis=0)
        counts = tf.reduce_sum(tf.cast(tf.not_equal(masked, 0.0), preds.dtype), axis=0)
        majority_means = sums / tf.maximum(counts, 1.0)

        combined = {}
        for key in common_keys:
            stacked = tf.stack([out[key] for out in outputs], axis=0)
            if key == "prediction":
                combined[key] = majority_means
            else:
                combined[key] = tf.reduce_mean(stacked, axis=0)
        return combined


def _resolve_model(path):
    """Resolve a user-supplied path to a model dict with graph/project/classes."""
    p = Path(path)

    # Direct path to a SavedModel graph directory.
    if p.is_dir() and p.name.endswith("_graph") and (p / "saved_model.pb").exists():
        graph_dir = p
        base_name = graph_dir.name.removesuffix("_graph")
        project = None
        classes = None
        for parent in [graph_dir.parent] + list(graph_dir.parents):
            if project is None:
                candidates = list(parent.glob(f"{base_name}_project.yaml"))
                if candidates:
                    project = candidates[0]
            if classes is None:
                candidates = list(parent.glob(f"{base_name}_classes.yaml"))
                if candidates:
                    classes = candidates[0]
            if project and classes:
                break
        return {"graph": graph_dir, "project": project, "classes": classes}

    # Otherwise expect a Jaeger experiment/model directory containing a
    # ``model`` sub-directory scanned by AvailableModels.
    info = AvailableModels(p).info
    if not info:
        return None
    # ``info`` is a dict mapping model name -> dict of artifacts.
    return next(iter(info.values()))


def combine_models_core(**kwargs):
    inputs = kwargs.get("input")
    method = kwargs.get("comb", "mean")

    models = []
    for raw_path in inputs:
        model = _resolve_model(raw_path)
        if model is None:
            logger.error(f"No Jaeger model found at: {raw_path}")
            logger.error(
                "Provide either the experiment directory that contains a "
                "'model' sub-directory, or the direct path to a *_graph "
                "SavedModel directory."
            )
            sys.exit(1)
        missing = [k for k in ("graph", "project", "classes") if not model.get(k)]
        if missing:
            logger.error(
                f"Model at {raw_path} is missing required artifacts: {', '.join(missing)}"
            )
            sys.exit(1)
        models.append(model)

    out_dir = Path(kwargs.get("output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = out_dir / "model"
    out_model_path.mkdir(parents=True, exist_ok=True)

    graph_dir = models[0]["graph"]
    model_name = graph_dir.name.removesuffix("_graph")
    out_graph_path = out_model_path / f"{model_name}_{len(models)}_ensemble_graph"
    out_project_path = (
        out_model_path / f"{model_name}_{len(models)}_ensemble_project.yaml"
    )
    out_class_path = (
        out_model_path / f"{model_name}_{len(models)}_ensemble_classes.yaml"
    )

    model_paths = [m["graph"] for m in models]
    project_path = models[0]["project"]
    class_path = models[0]["classes"]

    ensemble = EnsembleModel(model_paths, method=method)

    # Get the input signature dynamically from the first sub-model.
    input_spec = list(ensemble._signatures[0].structured_input_signature[1].values())[0]

    tf.saved_model.save(
        ensemble,
        export_dir=out_graph_path,
        signatures={
            "serving_default": ensemble.__call__.get_concrete_function(input_spec)
        },
    )
    shutil.copy(project_path, out_project_path)
    shutil.copy(class_path, out_class_path)

    print(f"✅ Ensemble SavedModel successfully created at: {out_graph_path}")
