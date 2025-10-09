import tensorflow as tf
import logging
from pathlib import Path
import shutil
from jaeger.utils.misc import json_to_dict, AvailableModels, get_model_id
logger = logging.getLogger("Jaeger")

from icecream import ic


class EnsembleModel(tf.Module):
    def __init__(self, model_paths):
        super().__init__()
        # Track submodels properly
        self.models = [tf.saved_model.load(p) for p in model_paths]
        # Use a non-reserved name
        self._signatures = [m.signatures["serving_default"] for m in self.models]

    @tf.function
    def __call__(self, inputs):
        """
        Average ensemble over all model outputs.
        Each model output: dict with keys ['nmd', 'embedding', 'reliability', 'gate', 'prediction']
        """
        outputs = [sig(inputs) for sig in self._signatures]

        combined = {}
        for key in outputs[0].keys():
            stacked = tf.stack([out[key] for out in outputs], axis=0)
            combined[key] = tf.reduce_mean(stacked, axis=0)
        return combined



def combine_models_core(**kwargs):
    model_info = [tuple(AvailableModels(i).info.values()) for i in kwargs.get('input')]
    aggregation = kwargs.get('comb')
    
    out_dir = Path(kwargs.get('output'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = out_dir / "model"
    model_name = model_info[0][0].get('graph').name.rsplit('_', 1)[0]
    out_graph_path = out_model_path / f"{model_name}_{len(model_info)}_ensemble_graph"
    out_project_path = out_model_path / f"{model_name}_{len(model_info)}_ensemble_project.yaml"
    out_class_path = out_model_path / f"{model_name}_{len(model_info)}_ensemble_classes.yaml"

    ic([i[0].get('graph') for i in model_info])


    # Load models as callable objects
    model_paths = [p[0].get('graph') for p in model_info]
    project_path = model_info[0][0].get('project')
    class_path =  model_info[0][0].get('classes')
    # preds = [sig(x)["output_0"] for sig in signatures]  # adjust key if needed
    # ic(preds)
    # === 3. Define the ensemble function ===

    ensemble = EnsembleModel(model_paths)

    # Get the input signature dynamically
    input_spec = list(ensemble._signatures[0].structured_input_signature[1].values())[0]

    # Save the new ensemble as a single SavedModel
    tf.saved_model.save(
        ensemble,
        export_dir=out_graph_path,
        signatures={
            "serving_default": ensemble.__call__.get_concrete_function(input_spec)
        },
    )
    shutil.copy(project_path, out_project_path)
    shutil.copy(class_path, out_class_path)

    print("✅ Ensemble SavedModel successfully created at: ensemble_savedmodel")