#!/usr/bin/env python3
"""Generate benchmark figures from report and evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_f1_per_class(metrics: pd.DataFrame, out_dir: Path) -> None:
    f1_cols = [c for c in metrics.columns if c.startswith("f1_class_")]
    melted = metrics.melt(
        id_vars=["length_bp", "input_type"],
        value_vars=f1_cols,
        var_name="class",
        value_name="f1",
    )
    melted["class"] = melted["class"].str.replace("f1_class_", "class ")
    g = sns.catplot(
        data=melted,
        x="class",
        y="f1",
        hue="input_type",
        col="length_bp",
        kind="bar",
    )
    g.fig.suptitle("Per-class F1 by length and input type", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "f1_per_class_bar.png")
    plt.close()


def plot_accuracy_vs_length(metrics: pd.DataFrame, out_dir: Path) -> None:
    metrics["length_bp"] = metrics["length_bp"].astype(int)
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=metrics,
        x="length_bp",
        y="overall_accuracy",
        hue="input_type",
        marker="o",
    )
    sns.lineplot(
        data=metrics,
        x="length_bp",
        y="balanced_accuracy",
        hue="input_type",
        marker="s",
        linestyle="--",
    )
    plt.title("Accuracy vs. sequence length")
    plt.xlabel("Length (bp)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_length.png")
    plt.close()


def plot_confusion_matrix_grid(metrics: pd.DataFrame, out_dir: Path) -> None:
    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    class_names = ["chromosome", "virus", "plasmid"]
    for idx, (_, row) in enumerate(metrics.iterrows()):
        cm = np.load(row["cm_path"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{row['input_type']} {row['length_bp']}bp")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_grid.png")
    plt.close()


def plot_training_curves(report: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    report["best_val_accuracy"] = report["best_val_accuracy"].astype(float)
    sns.lineplot(data=report, x="best_epoch", y="best_val_accuracy", hue="experiment")
    plt.title("Best validation accuracy per experiment")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-csv", type=Path, default=Path("benchmark_report.csv"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("evaluation_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_report/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(args.report_csv)
    metrics = pd.read_csv(args.metrics_csv)

    plot_f1_per_class(metrics, args.output_dir)
    plot_accuracy_vs_length(metrics, args.output_dir)
    plot_confusion_matrix_grid(metrics, args.output_dir)
    plot_training_curves(report, args.output_dir)

    print(f"Figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
