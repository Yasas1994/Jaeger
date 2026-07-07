"""Smoke test: train a fragment classifier with a single Hyena block."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jaeger.commands.train import train_fragment_core
from jaeger.utils.misc import load_model_config


def _make_tiny_csv(path: Path, n: int = 100) -> None:
    """Write a tiny label,sequence CSV for smoke testing."""
    rng = np.random.default_rng(42)
    alphabet = np.array(["A", "T", "G", "C"])
    with path.open("w") as f:
        for i in range(n):
            label = i % 6
            seq = "".join(rng.choice(alphabet, size=750))
            f.write(f"{label},{seq}\n")


def test_hyena_classifier_smoke():
    """Run one classifier epoch using the Hyena representation learner."""
    repo_cfg = load_model_config(Path("train_config/hyena_test.yaml"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_csv = tmp / "train.csv"
        val_csv = tmp / "val.csv"
        _make_tiny_csv(train_csv, 60)
        _make_tiny_csv(val_csv, 30)

        repo_cfg["training"]["data_dir"] = str(tmp)
        repo_cfg["training"]["classifier_dir"] = str(tmp / "checkpoints" / "classifier")
        repo_cfg["training"]["projection_dir"] = str(tmp / "checkpoints" / "projection")
        repo_cfg["training"]["classifier_epochs"] = 1
        repo_cfg["training"]["classifier_train_steps"] = 2
        repo_cfg["training"]["classifier_validation_steps"] = 1
        repo_cfg["training"]["model_saving"]["path"] = str(tmp / "model")

        history_csv = tmp / "classifier_history.csv"
        repo_cfg["training"]["callbacks"] = {
            "clean_old": True,
            "classifier": [
                {"name": "CSVLogger", "params": {"filename": str(history_csv)}}
            ],
            "projection": [],
        }
        repo_cfg["training"]["fragment_classifier_data"] = {
            "train": [{"class": ["b"], "label": [0], "path": [str(train_csv)]}],
            "validation": [{"class": ["b"], "label": [0], "path": [str(val_csv)]}],
        }

        config_path = tmp / "config.yaml"
        config_path.write_text(yaml.safe_dump(repo_cfg))

        train_fragment_core(
            config=str(config_path),
            precision="fp32",
            from_last_checkpoint=False,
            force=True,
            only_classification_head=False,
            only_reliability_head=False,
            only_heads=False,
            only_save=False,
            save_model=False,
            masking=None,
            self_supervised_pretraining=False,
            xla=False,
            ignore_convergence=False,
            meta=None,
        )

        assert history_csv.exists(), "Classifier training history was not written"
        history = pd.read_csv(history_csv)
        assert "loss" in history.columns, "loss not found in training history"
        assert np.isfinite(history["loss"].iloc[0]), "loss is not finite"
        print("All checks passed.")


if __name__ == "__main__":
    test_hyena_classifier_smoke()
