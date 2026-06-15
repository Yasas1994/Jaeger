"""Smoke test for the PyTorch training and inference path.

Builds a tiny model and tiny CSV dataset, runs one training epoch, then runs
prediction against a small FASTA input. The test only asserts that the outputs
are produced without error; it does not assert performance.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from jaeger.cli import main as jaeger_cli
from jaeger.commands.train import train_fragment_core
from jaeger.utils.misc import load_model_config


def _write_tiny_csv(path: Path, n_samples: int = 16, seq_len: int = 96) -> None:
    """Write a tiny label-first CSV dataset."""
    rng = range(n_samples)
    rows = []
    for i in rng:
        label = i % 3
        seq = "ATGCATGC" * (seq_len // 8)
        rows.append(f"{label},{seq},seq{i}")
    path.write_text("\n".join(rows) + "\n")


def _write_tiny_fasta(path: Path, n_contigs: int = 2, seq_len: int = 2000) -> None:
    """Write a tiny FASTA file for inference."""
    seq = "ATGCATGC" * (seq_len // 8)
    lines = [f">contig{i}\n{seq}" for i in range(n_contigs)]
    path.write_text("\n".join(lines) + "\n")


def _tiny_config(tmp_path: Path) -> dict[str, object]:
    """Return a minimal Jaeger PyTorch training config dict."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    _write_tiny_csv(train_csv)
    _write_tiny_csv(val_csv)

    experiment_root = "experiments/tiny_smoke"
    classifier_dir = data_dir / experiment_root / "checkpoints" / "classifier"
    model_save_path = data_dir / experiment_root / "model"

    return {
        "model": {
            "name": "jaeger_tiny_smoke",
            "experiment": "tiny_smoke",
            "seed": 42,
            "classifier_out_dim": 3,
            "reliability_out_dim": 0,
            "class_label_map": [
                {"class": "chromosome", "label": 0},
                {"class": "virus", "label": 1},
                {"class": "plasmid", "label": 2},
            ],
            "activation": "gelu",
            "mode": "training",
            "embedding": {
                "use_embedding_layer": True,
                "input_type": "translated",
                "vocab_size": 65,
                "strands": 2,
                "frames": 6,
                "length": None,
                "input_shape": [6, None],
                "embedding_size": 8,
                "embedding_regularizer": "l2",
                "embedding_regularizer_w": 1.0e-05,
            },
            "string_processor": {
                "data_format": "csv",
                "seq_onehot": False,
                "codon": "CODON",
                "codon_id": "CODON_ID",
                "crop_size": 96,
                "buffer_size": 1000,
                "shuffle": False,
                "reshuffle_each_iteration": False,
                "mutate": False,
                "mutation_rate": 0.1,
                "shuffle_frames": False,
                "masking": False,
                "classifier_labels": [0, 1, 2],
                "classifier_labels_map": [0, 1, 2],
            },
            "representation_learner": {
                "hidden_layers": [
                    {
                        "name": "masked_conv1d",
                        "config": {
                            "filters": 8,
                            "kernel_size": 3,
                            "padding": "same",
                            "activation": None,
                            "use_bias": True,
                        },
                    },
                    {
                        "name": "masked_batchnorm",
                        "config": {"num_features": 8, "return_nmd": False},
                    },
                    {"name": "activation", "config": {"activation": "gelu"}},
                ],
                "pooling": "average",
            },
            "classifier": {
                "input_shape": 8,
                "dropout": 0.1,
                "hidden_layers": [
                    {"name": "dense", "config": {"units": 8, "activation": None}},
                    {"name": "dense", "config": {"units": 3, "activation": None}},
                ],
            },
        },
        "training": {
            "data_dir": str(data_dir),
            "experiment_root": experiment_root,
            "classifier_dir": str(classifier_dir),
            "reliability_dir": str(
                data_dir / experiment_root / "checkpoints" / "reliability"
            ),
            "classifier_epochs": 1,
            "reliability_epochs": 0,
            "projection_epochs": 0,
            "classifier_train_steps": 1000,
            "reliability_train_steps": 0,
            "classifier_validation_steps": 100,
            "reliability_validation_steps": 0,
            "batch_size": 4,
            "optimizer": "adam",
            "optimizer_params": {"learning_rate": 0.001},
            "loss_classifier": "categorical_crossentropy",
            "loss_params_classifier": {"from_logits": True},
            "metrics_classifier": [{"name": "categorical_accuracy", "params": None}],
            "callbacks": {"classifier": []},
            "model_saving": {
                "path": str(model_save_path),
                "save_weights": True,
                "save_exec_graph": False,
            },
            "fragment_classifier_data": {
                "train": [
                    {
                        "class": ["chromosome", "virus", "plasmid"],
                        "path": [str(train_csv)],
                        "label": [0, 1, 2],
                    }
                ],
                "validation": [
                    {
                        "class": ["chromosome", "virus", "plasmid"],
                        "path": [str(val_csv)],
                        "label": [0, 1, 2],
                    }
                ],
            },
        },
    }


def test_pytorch_train_predict_smoke(tmp_path: Path) -> None:
    """Train a tiny PyTorch Jaeger model and run a tiny prediction job."""
    config_path = tmp_path / "config.yaml"
    config = _tiny_config(tmp_path)

    import yaml

    config_path.write_text(yaml.safe_dump(config))

    # Train for one epoch and save the model.
    train_fragment_core(
        config=str(config_path),
        mixed_precision=False,
        from_last_checkpoint=False,
        force=False,
        only_classification_head=False,
        only_reliability_head=False,
        only_heads=False,
        only_save=False,
        save_model=True,
        self_supervised_pretraining=False,
        xla=False,
        meta=None,
    )

    experiment_root = (
        Path(config["training"]["data_dir"]) / config["training"]["experiment_root"]
    )
    model_dir = experiment_root / "model"
    assert model_dir.exists(), f"model directory was not created: {model_dir}"
    checkpoint = model_dir / "classifier.pt"
    assert checkpoint.exists(), f"classifier checkpoint was not created: {checkpoint}"

    # Prepare a tiny FASTA input and run prediction.
    fasta_path = tmp_path / "input.fasta"
    _write_tiny_fasta(fasta_path)

    output_dir = tmp_path / "predictions"
    runner = CliRunner()
    result = runner.invoke(
        jaeger_cli,
        [
            "predict",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(fasta_path),
            "--output",
            str(output_dir),
            "--cpu",
            "--overwrite",
            "--batch",
            "2",
        ],
    )
    assert result.exit_code == 0, f"predict failed: {result.output}\n{result.exception}"

    model_name = (
        load_model_config(config_path).get("model", {}).get("name", "jaeger_pytorch")
    )
    prediction_output = output_dir / model_name.replace(" ", "_") / "input.tsv"
    assert prediction_output.exists(), (
        f"prediction TSV was not created: {prediction_output}"
    )
