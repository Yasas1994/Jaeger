"""Forward parity test between the legacy TensorFlow builder and PyTorch builder.

This test builds the same tiny fragment classifier with both backends, copies the
TensorFlow weights into the PyTorch model, and asserts that the forward outputs
agree within tolerance.
"""

import os

# Suppress TensorFlow logging before importing it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

from jaeger.nnlib.builder import DynamicModelBuilder as TFBuilder
from jaeger.nnlib.pytorch.builder import ModelBuilder as PTBuilder


CONFIG = {
    "model": {
        "name": "parity_test",
        "classifier_out_dim": 3,
        "reliability_out_dim": 1,
        "class_label_map": [
            {"class": "phage", "label": 1},
            {"class": "bacteria", "label": 0},
            {"class": "eukarya", "label": 2},
        ],
        "embedding": {
            "input_type": "translated",
            "use_embedding_layer": True,
            "embedding_size": 32,
            "input_shape": [6, None],
            "vocab_size": 65,
            "codon_depth": 1,
            "embedding_regularizer": "l2",
            "embedding_regularizer_w": 1e-5,
        },
        "string_processor": {
            "codon": "CODON",
            "codon_id": "CODON_ID",
            "crop_size": 50,
            "input_type": "translated",
            "use_embedding_layer": True,
        },
        "representation_learner": {
            "hidden_layers": [
                {
                    "name": "masked_conv1d",
                    "config": {
                        "filters": 16,
                        "kernel_size": 3,
                        "padding": "same",
                    },
                },
            ],
            "pooling": "average",
        },
        "classifier": {
            "input_shape": 16,
            "hidden_layers": [{"name": "dense", "config": {"units": 3}}],
        },
    },
    "training": {
        "batch_size": 2,
        "optimizer": "adam",
        "optimizer_params": {"learning_rate": 1e-3},
    },
}


def _collect_tf_weights(layer, dest):
    """Recursively collect leaf-layer TF weights keyed by (layer_name, weight_name)."""
    has_sublayers = hasattr(layer, "layers") and layer.layers
    if not has_sublayers and hasattr(layer, "weights"):
        for weight in layer.weights:
            # Weight names from top-level leaf layers are short (e.g. "kernel"),
            # while nested leaf layers include the layer path
            # (e.g. "classifier_dense_0/kernel").
            full_name = weight.name
            if "/" in full_name:
                layer_name, weight_name = full_name.rsplit("/", 1)
            else:
                layer_name = layer.name
                weight_name = full_name
            dest[(layer_name, weight_name)] = weight
    if has_sublayers:
        for sub in layer.layers:
            _collect_tf_weights(sub, dest)


def _copy_tf_weights_to_pt(tf_model, pt_model):
    """Copy weights from the TF model to the PyTorch model in-place."""
    # PyTorch MaskedConv1D is lazy: run one forward pass to materialize the real
    # Conv1d with the correct in_channels before copying weights.
    dummy_x = torch.zeros(2, 6, 50, dtype=torch.int64)
    dummy_mask = torch.ones(2, 6, 50, dtype=torch.bool)
    pt_model.eval()
    with torch.no_grad():
        pt_model(dummy_x, mask=dummy_mask)

    tf_weights = {}
    for layer in tf_model.layers:
        _collect_tf_weights(layer, tf_weights)

    pt_state = pt_model.state_dict()

    def _get(name_pair):
        weight = tf_weights.get(name_pair)
        if weight is None:
            raise ValueError(f"Missing TensorFlow weight for {name_pair}")
        return torch.from_numpy(weight.numpy())

    def _set(pt_key, tf_weight, transform=None):
        if pt_key not in pt_state:
            raise ValueError(f"Missing PyTorch state_dict key: {pt_key}")
        tensor = _get(tf_weight)
        if transform is not None:
            tensor = transform(tensor)
        pt_state[pt_key].copy_(tensor)

    # Embedding layer: direct copy.
    _set("rep_model.embedding.embed.weight", ("embedding", "embeddings"))

    # MaskedConv1D: TF kernel shape (K, C_in, C_out) -> PyTorch (C_out, C_in, K).
    _set(
        "rep_model.blocks.0.conv.weight",
        ("rep_masked_conv1d_0", "kernel"),
        transform=lambda t: t.permute(2, 1, 0),
    )
    _set("rep_model.blocks.0.conv.bias", ("rep_masked_conv1d_0", "bias"))

    # Classification dense head: TF kernel shape (C_in, C_out) -> PyTorch (C_out, C_in).
    _set(
        "classification_head.net.0.weight",
        ("classifier_dense_0", "kernel"),
        transform=lambda t: t.T,
    )
    _set("classification_head.net.0.bias", ("classifier_dense_0", "bias"))


def test_forward_parity():
    tf_builder = TFBuilder(CONFIG)
    tf_models = tf_builder.build_fragment_classifier()
    tf_model = tf_models["jaeger_model"]

    pt_builder = PTBuilder(CONFIG)
    pt_models = pt_builder.build_fragment_classifier()
    pt_model = pt_models["jaeger_model"]
    pt_model.eval()

    _copy_tf_weights_to_pt(tf_model, pt_model)

    # Avoid zeros so the TF model's automatic mask (x != 0) is all-ones and
    # matches the all-ones PyTorch mask, sidestepping the fact that TF's
    # GlobalAveragePooling2D does not consume masks while PyTorch's pooler does.
    x = np.random.randint(1, 65, size=(2, 6, 50)).astype(np.int64)
    tf_out = tf_model.predict({"translated": x}, verbose=0)
    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(x), mask=torch.ones(2, 6, 50, dtype=torch.bool)
        )

    np.testing.assert_allclose(
        tf_out["prediction"],
        pt_out["prediction"].numpy(),
        atol=1e-4,
        rtol=1e-4,
    )
