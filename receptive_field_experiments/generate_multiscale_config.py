"""Generate multi-scale and baseline configs for Zeus."""

from pathlib import Path

import yaml

TEMPLATE = Path("train_config/nn_config_1500bp_nmd_merge_6_class.yaml")
OUT_DIR = Path("receptive_field_experiments")
BASE_DIR = Path("/mnt/beegfs/bioinf/wijesekara/jaeger/receptive_field_experiments")

BRANCHES = [
    ("rf_031", 6),
    ("rf_063", 14),
    ("rf_127", 30),
    ("rf_191", 46),
]


def _residual_block(dilation):
    return {
        "name": "residual_block",
        "config": {
            "use_1x1conv": False,
            "block_size": 1,
            "filters": 128,
            "kernel_size": 5,
            "dilation_rate": dilation,
            "use_bias": True,
            "kernel_regularizer": "l2",
            "kernel_regularizer_w": 0.00001,
            "norm_type": "masked_layernorm",
        },
    }


def _shared_stem():
    return [
        {
            "name": "masked_conv1d",
            "config": {
                "filters": 128,
                "kernel_size": 7,
                "strides": 1,
                "dilation_rate": 1,
                "use_bias": True,
                "activation": None,
                "kernel_regularizer": "l2",
                "kernel_regularizer_w": 0.00001,
            },
        },
        {"name": "masked_layernorm"},
        {"name": "activation", "config": {"activation": "gelu"}},
    ]


def _multiscale_representation_learner():
    return {
        "hidden_layers": _shared_stem()
        + [
            {
                "name": "parallel_branches",
                "config": {
                    "merge": "concat",
                    "branches": [
                        {
                            "hidden_layers": [_residual_block(d)],
                            "pooling": "max",
                        }
                        for _, d in BRANCHES
                    ],
                },
            }
        ],
        "pooling": None,
    }


def _single_rf_representation_learner(dilation):
    return {
        "hidden_layers": _shared_stem() + [_residual_block(dilation), {"name": "nmd"}],
        "pooling": "max",
    }


def _configure_training(cfg):
    cfg["training"]["classifier_epochs"] = 10
    cfg["training"]["classifier_train_steps"] = 10000
    cfg["training"]["classifier_validation_steps"] = 1000
    cfg["training"]["projection_epochs"] = 0
    cfg["training"]["reliability_epochs"] = 0
    cfg["model"]["string_processor"]["mutate"] = False
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(TEMPLATE.read_text())

    # Multi-scale config
    cfg = yaml.safe_load(yaml.dump(template, sort_keys=False))
    cfg["model"].pop("reliability_model", None)
    cfg["model"]["name"] = "jaeger"
    cfg["model"]["experiment"] = "multiscale_rf"
    cfg["model"]["base_dir"] = str(BASE_DIR)
    cfg["model"]["representation_learner"] = _multiscale_representation_learner()
    cfg["model"]["classifier"]["input_shape"] = 512
    cfg["model"]["projection"]["input_shape"] = 512
    cfg = _configure_training(cfg)
    (OUT_DIR / "multiscale_rf.yaml").write_text(
        yaml.dump(cfg, sort_keys=False, default_flow_style=False)
    )

    # Single-RF baseline (RF 127)
    cfg = yaml.safe_load(yaml.dump(template, sort_keys=False))
    cfg["model"].pop("reliability_model", None)
    cfg["model"]["name"] = "jaeger"
    cfg["model"]["experiment"] = "single_rf_127_baseline"
    cfg["model"]["base_dir"] = str(BASE_DIR)
    cfg["model"]["representation_learner"] = _single_rf_representation_learner(30)
    cfg["model"]["classifier"]["input_shape"] = 128
    cfg["model"]["projection"]["input_shape"] = 128
    cfg = _configure_training(cfg)
    (OUT_DIR / "single_rf_127_baseline.yaml").write_text(
        yaml.dump(cfg, sort_keys=False, default_flow_style=False)
    )

    print("Wrote multiscale_rf.yaml and single_rf_127_baseline.yaml")


if __name__ == "__main__":
    main()
