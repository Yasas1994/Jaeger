import torch

from jaeger.nnlib.pytorch.builder import ModelBuilder


def test_builder_creates_jaeger_model():
    config = {
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
    builder = ModelBuilder(config)
    models = builder.build_fragment_classifier()
    x = torch.randint(0, 65, (2, 6, 50))
    mask = torch.ones(2, 6, 50, dtype=torch.bool)
    out = models["jaeger_model"](x, mask)
    assert out["prediction"].shape == (2, 3)
