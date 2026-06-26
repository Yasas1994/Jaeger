#!/usr/bin/env python3
"""Plot training-history curves for the multiscale + global-context sweep.

Usage:
    python plot_training_history.py

The script searches under ``experiments/`` for each run, reads the classifier
and projection ``training.log`` CSVs, and saves comparison figures to
``figures/``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def find_logs(root: Path) -> dict[str, dict[str, Path]]:
    """Return {experiment_name: {branch: training.log Path}}."""
    experiments: dict[str, dict[str, Path]] = {}
    for branch in ("classifier",):
        for log_file in sorted(root.rglob(f"{branch}/training.log")):
            # experiment dir is e.g. experiments/experiment_<name>_<seed>
            
            exp_dir = log_file.parents[2]
            #print(exp_dir)
            m = re.match(r"experiment_(.+)_\d+", exp_dir.name)
            name = m.group(1) if m else exp_dir.name
            experiments.setdefault(name, {})[branch] = log_file
    return experiments


def read_log(path: Path) -> dict[str, list[float]]:
    """Read a Keras CSVLogger file, tolerating repeated headers."""
    columns: dict[str, list[float]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row or "epoch" not in row:
                continue
            for key, val in row.items():
                if key is None:
                    continue
                try:
                    columns.setdefault(key, []).append(float(val))
                except ValueError:
                    pass
    return columns


def plot_branch(
    experiments: dict[str, dict[str, Path]],
    branch: str,
    out_dir: Path,
) -> None:
    """Plot loss and metric curves for one branch across experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, logs in experiments.items():
        log_path = logs.get(branch)
        if log_path is None:
            continue
        data = read_log(log_path)
        if not data:
            continue
        epochs = data.get("epoch")
        if epochs is None:
            epochs = list(range(1, len(data.get("loss", [])) + 1))
        label = name.replace("multiscale_rf_", "")
        axes[0].plot(epochs, data.get("loss", []), label=f"{label} train")
        axes[0].plot(epochs, data.get("val_loss", []), label=f"{label} val", linestyle="--")
        axes[1].plot(epochs, data.get("categorical_accuracy", []), label=f"{label} acc")
        axes[1].plot(epochs, data.get("val_categorical_accuracy", []), label=f"{label} val_acc", linestyle="--")

    axes[0].set_title(f"{branch.capitalize()} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"{branch.capitalize()} categorical accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{branch}_history.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    experiments = find_logs(root)
    print(experiments)
    if not experiments:
        print("No training logs found under experiments/")
        return

    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)

    for branch in ["classifier",]:
        plot_branch(experiments, branch, out_dir)


if __name__ == "__main__":
    main()
