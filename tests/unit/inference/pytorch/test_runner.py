import tempfile

import torch

from jaeger.inference.pytorch.runner import PyTorchInferenceRunner
from jaeger.nnlib.pytorch.builder import ModelBuilder


def _make_config(tmp_path):
    return {
        "model": {
            "name": "test_model",
            "classifier_out_dim": 3,
            "reliability_out_dim": 1,
            "class_label_map": [
                {"class": "phage", "label": 1},
                {"class": "bacteria", "label": 0},
            ],
            "embedding": {
                "input_type": "translated",
                "use_embedding_layer": False,
                "embedding_size": 32,
                "input_shape": [6, None],
                "vocab_size": 65,
                "codon_depth": 1,
            },
            "string_processor": {"codon": "CODON", "codon_id": "CODON_ID"},
            "representation_learner": {
                "hidden_layers": [
                    {
                        "name": "masked_conv1d",
                        "config": {"filters": 16, "kernel_size": 3, "padding": "same"},
                    }
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
            "optimizer_params": {"lr": 1e-3},
        },
    }


def test_inference_runner_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_config(tmpdir)
        runner = PyTorchInferenceRunner(config, device=torch.device("cpu"))
        x = torch.randint(0, 65, (4, 6, 50))
        mask = torch.ones(4, 6, 50, dtype=torch.bool)
        out = runner.predict(x, mask=mask, batch_size=2)
        assert out["prediction"].shape == (4, 3)


def test_inference_runner_from_checkpoint(tmp_path):
    config = _make_config(tmp_path)
    builder = ModelBuilder(config)
    models = builder.build_fragment_classifier()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": models["jaeger_model"].state_dict()}, checkpoint_path
    )

    runner = PyTorchInferenceRunner(
        config, checkpoint_path=checkpoint_path, device=torch.device("cpu")
    )
    x = torch.randint(0, 65, (2, 6, 50))
    out = runner.predict(x, batch_size=2)
    assert out["prediction"].shape == (2, 3)
