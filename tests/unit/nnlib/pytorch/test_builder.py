import pytest
import torch

from jaeger.nnlib.pytorch.builder import ModelBuilder


def _minimal_config():
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
                        "config": {
                            "filters": 16,
                            "kernel_size": 3,
                            "padding": "same",
                        },
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


def test_builder_creates_jaeger_model():
    builder = ModelBuilder(_minimal_config())
    models = builder.build_fragment_classifier()
    x = torch.randint(0, 65, (2, 6, 50))
    mask = torch.ones(2, 6, 50, dtype=torch.bool)
    out = models["jaeger_model"](x, mask)
    assert out["prediction"].shape == (2, 3)


def test_builder_compile_classifier():
    builder = ModelBuilder(_minimal_config())
    models = builder.build_fragment_classifier()
    model, optimizer, loss = builder.compile_model(models, "classifier")
    x = torch.randint(0, 65, (2, 6, 50))
    mask = torch.ones(2, 6, 50, dtype=torch.bool)
    out = model(x, mask)
    assert out.shape == (2, 3)
    assert optimizer is not None
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_builder_missing_classifier_raises():
    config = _minimal_config()
    del config["model"]["classifier"]
    builder = ModelBuilder(config)
    with pytest.raises(ValueError, match="classifier config is required"):
        builder.build_fragment_classifier()


def test_builder_mask_propagation():
    builder = ModelBuilder(_minimal_config())
    models = builder.build_fragment_classifier()
    model, _, _ = builder.compile_model(models, "classifier")
    model.eval()

    x = torch.zeros(2, 6, 50, dtype=torch.long)
    x[0, 0, 25] = 50

    mask_all = torch.ones(2, 6, 50, dtype=torch.bool)
    mask_masked = mask_all.clone()
    mask_masked[0, 0, 25] = False

    with torch.no_grad():
        out_all = model(x, mask_all)
        out_masked = model(x, mask_masked)

    assert out_all.shape == (2, 3)
    assert not torch.allclose(out_all[0], out_masked[0], atol=1e-6)
    assert torch.allclose(out_all[1], out_masked[1])


def _dvf_config():
    return {
        "model": {
            "name": "dvf_test",
            "classifier_out_dim": 3,
            "reliability_out_dim": 0,
            "model_type": "siamese",
            "embedding": {
                "input_type": "nucleotide",
                "use_embedding_layer": False,
                "vocab_size": 6,
                "onehot_dim": 4,
                "embedding_size": 4,
            },
            "string_processor": {"input_key": "nucleotide"},
            "representation_learner": {
                "branch_layers": [
                    {"name": "permute", "config": {"dims": [0, 2, 1]}},
                    {
                        "name": "conv1d",
                        "config": {
                            "in_channels": 4,
                            "out_channels": 8,
                            "kernel_size": 3,
                        },
                    },
                    {"name": "relu"},
                    {"name": "adaptive_max_pool1d", "config": {"output_size": 1}},
                    {"name": "squeeze_last"},
                    {"name": "linear", "config": {"in_features": 8, "out_features": 16}},
                ]
            },
            "classifier": {
                "hidden_layers": [
                    {"name": "linear", "config": {"in_features": 16, "out_features": 3}}
                ]
            },
        },
        "training": {"batch_size": 2, "optimizer": "adam", "optimizer_params": {}},
    }


def test_builder_siamese_dvf():
    builder = ModelBuilder(_dvf_config())
    models = builder.build_fragment_classifier()
    x = torch.randint(0, 6, (2, 2, 20))
    mask = torch.ones(2, 2, 20, dtype=torch.bool)
    out = models["jaeger_model"](x, mask)
    assert out["prediction"].shape == (2, 3)
    assert out["embedding"].shape == (2, 16)
