#!/usr/bin/env python3
"""Evaluate Jaeger predictions on real-world metagenome assemblies.

Matches Jaeger per-contig TSV predictions against fraction-based labels
(e.g. cellular vs phage/virus fractions) and reports precision, recall,
F1, confusion matrices, and reliability statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


VIRAL_FRACTIONS = {"phage", "virus", "viral"}


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a Jaeger prediction TSV and normalize contig IDs."""
    df = pd.read_csv(path, sep="\t")
    if "contig_id" in df.columns:
        df["contig_id"] = df["contig_id"].str.replace("___", ",", regex=False)
    return df


def load_labels(path: Path) -> pd.DataFrame:
    """Load a real-world label TSV."""
    return pd.read_csv(path, sep="\t")


def build_ground_truth(labels_df: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """Return binary viral labels (1=viral, 0=cellular) and the sorted class list."""
    if "fraction" not in labels_df.columns:
        raise ValueError("Label file must contain a 'fraction' column")
    y_true = labels_df["fraction"].isin(VIRAL_FRACTIONS).astype(int).to_numpy()
    classes = pd.Series(labels_df["fraction"].unique()).sort_values()
    return y_true, classes


def build_viral_predictions(
    preds_df: pd.DataFrame, reliability_cutoff: float = 0.0
) -> np.ndarray:
    """Return binary viral predictions, optionally filtered by reliability cutoff.

    Predictions below the reliability cutoff are treated as uncertain and are
    assigned the negative (cellular) class for the binary viral-detection task.
    """
    if "prediction" not in preds_df.columns:
        raise ValueError("Prediction file must contain a 'prediction' column")
    is_viral = preds_df["prediction"].isin(VIRAL_FRACTIONS)
    if "reliability_score" in preds_df.columns and reliability_cutoff > 0:
        reliable = preds_df["reliability_score"] >= reliability_cutoff
        is_viral = is_viral & reliable
    return is_viral.astype(int).to_numpy()


def compute_binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sample_name: str
) -> dict[str, float]:
    """Compute precision, recall, F1, accuracy, and balanced accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "sample": sample_name,
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "f1": float(f1[1]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def compute_per_class_metrics(
    labels_df: pd.DataFrame, preds_df: pd.DataFrame
) -> dict[str, float]:
    """Compute per-class precision/recall/F1 against the fraction labels."""
    merged = pd.merge(labels_df, preds_df, on="contig_id", how="inner")
    if merged.empty:
        return {}
    y_true = merged["fraction"].to_numpy()
    y_pred = merged["prediction"].to_numpy()
    classes = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    metrics: dict[str, float] = {}
    for i, cls in enumerate(classes):
        metrics[f"precision_{cls}"] = float(precision[i])
        metrics[f"recall_{cls}"] = float(recall[i])
        metrics[f"f1_{cls}"] = float(f1[i])
        metrics[f"support_{cls}"] = float(support[i])
    return metrics


def evaluate_sample(
    pred_path: Path,
    label_path: Path,
    output_dir: Path,
    reliability_cutoff: float = 0.0,
) -> tuple[dict[str, object], np.ndarray]:
    """Evaluate one sample and write metrics files.

    Metrics are computed on the intersection of prediction and label
    ``contig_id`` values, so label files covering more contigs than the
    prediction file (or vice versa) are handled correctly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_name = pred_path.stem

    preds_df = load_predictions(pred_path)
    labels_df = load_labels(label_path)

    merged = pd.merge(labels_df, preds_df, on="contig_id", how="inner")
    if merged.empty:
        raise ValueError(
            f"No overlapping contig_ids between predictions ({pred_path.name}) "
            f"and labels ({label_path.name})"
        )

    y_true, _ = build_ground_truth(merged)
    y_pred = build_viral_predictions(merged, reliability_cutoff=reliability_cutoff)

    binary_metrics = compute_binary_metrics(y_true, y_pred, sample_name)
    per_class = compute_per_class_metrics(labels_df, preds_df)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    result: dict[str, object] = {
        **binary_metrics,
        **per_class,
        "num_contigs": int(len(y_true)),
        "num_viral_true": int(np.sum(y_true == 1)),
        "num_cellular_true": int(np.sum(y_true == 0)),
        "num_viral_pred": int(np.sum(y_pred == 1)),
        "reliability_cutoff": float(reliability_cutoff),
    }

    # Reliability statistics
    if "reliability_score" in preds_df.columns:
        result["mean_reliability"] = float(preds_df["reliability_score"].mean())
        result["median_reliability"] = float(preds_df["reliability_score"].median())
        result["frac_above_cutoff"] = float(
            np.mean(preds_df["reliability_score"] >= reliability_cutoff)
        )

    # Write outputs
    with open(output_dir / f"{sample_name}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(output_dir / f"{sample_name}_confusion_matrix.npy", cm)
    pd.DataFrame([result]).to_csv(
        output_dir / f"{sample_name}_metrics.csv", index=False
    )

    return result, cm


def discover_sample_pairs(
    predictions_dir: Path, labels_dir: Path
) -> list[tuple[Path, Path, str]]:
    """Pair prediction TSVs with label TSVs by sample stem.

    Prediction files inherit the input FASTA name (e.g.
    ``gut_scaffolds_gt1500.tsv``) while label files are stored as
    ``<sample>_labels.tsv`` (e.g. ``gut_labels.tsv``). Each prediction stem is
    matched to the longest label stem that equals it or is a ``_``-separated
    prefix of it, so both exact names and suffixed prediction names pair.
    """
    pred_files = {p.stem: p for p in predictions_dir.glob("*.tsv")}
    label_files = {
        p.stem.replace("_labels", ""): p for p in labels_dir.glob("*_labels.tsv")
    }

    def _match_label(sample_name: str) -> Path | None:
        tokens = sample_name.split("_")
        for end in range(len(tokens), 0, -1):
            candidate = "_".join(tokens[:end])
            if candidate in label_files:
                return label_files[candidate]
        return None

    pairs: list[tuple[Path, Path, str]] = []
    for sample_name, pred_path in pred_files.items():
        label_path = _match_label(sample_name)
        if label_path is None:
            print(
                f"Warning: no label file found for sample '{sample_name}'",
                file=sys.stderr,
            )
            continue
        pairs.append((pred_path, label_path, sample_name))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Jaeger predictions on real-world fraction labels."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory with .tsv predictions",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory with *_labels.tsv files",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to write metrics"
    )
    parser.add_argument(
        "--reliability-cutoff",
        type=float,
        default=0.0,
        help="Minimum reliability score for a prediction to be accepted",
    )
    args = parser.parse_args()

    if not args.predictions_dir.is_dir():
        print(
            f"Predictions directory not found: {args.predictions_dir}", file=sys.stderr
        )
        return 1
    if not args.labels_dir.is_dir():
        print(f"Labels directory not found: {args.labels_dir}", file=sys.stderr)
        return 1

    pairs = discover_sample_pairs(args.predictions_dir, args.labels_dir)
    if not pairs:
        print("No matching prediction/label pairs found.", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for pred_path, label_path, sample_name in pairs:
        print(f"Evaluating {sample_name}...")
        result, cm = evaluate_sample(
            pred_path,
            label_path,
            args.output_dir,
            reliability_cutoff=args.reliability_cutoff,
        )
        rows.append(result)
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1:        {result['f1']:.4f}")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  CM:\n{cm}")

    if rows:
        aggregated = pd.DataFrame(rows)
        aggregated.to_csv(args.output_dir / "aggregated_metrics.csv", index=False)
        print(
            f"Wrote aggregated metrics to {args.output_dir / 'aggregated_metrics.csv'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
