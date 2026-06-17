#!/usr/bin/env python3
"""Evaluate a trained Jaeger SavedModel against a NumPy validation set.

Loads the Keras model produced by `save_exec_graph: true` and computes
overall and per-class metrics from an NPZ file with `features`/`labels`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from jaeger.nnlib.v2.layers import MaskedBatchNorm, MaskedConv1D, ResidualBlock_wrapper
from jaeger.nnlib.v2.nmd import NMDLayer, NMDMerge


def load_jaeger_model(model_dir: Path) -> tf.keras.Model:
    """Load a Jaeger SavedModel, supplying custom layer classes."""
    custom_objects = {
        "MaskedConv1D": MaskedConv1D,
        "MaskedBatchNorm": MaskedBatchNorm,
        "ResidualBlock_wrapper": ResidualBlock_wrapper,
        "NMDLayer": NMDLayer,
        "NMDMerge": NMDMerge,
    }
    return tf.keras.models.load_model(str(model_dir), custom_objects=custom_objects)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
    return_cm: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Return accuracy, balanced accuracy, and per-class precision/recall/F1.

    Optionally also return the confusion matrix when ``return_cm`` is True.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    pred_labels = np.argmax(y_pred, axis=-1)
    true_labels = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true

    overall_acc = float(accuracy_score(true_labels, pred_labels))
    balanced_acc = float(balanced_accuracy_score(true_labels, pred_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

    metrics: dict[str, float] = {
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
    }
    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = float(precision[i])
        metrics[f"recall_class_{i}"] = float(recall[i])
        metrics[f"f1_class_{i}"] = float(f1[i])
        metrics[f"support_class_{i}"] = float(support[i])

    if return_cm:
        return metrics, cm
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Jaeger SavedModel on a NumPy validation set."
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Path to SavedModel directory."
    )
    parser.add_argument(
        "--npz", type=Path, required=True, help="Path to validation NPZ file."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Optional sample limit."
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None, help="Optional CSV path for metrics."
    )
    parser.add_argument(
        "--output-cm",
        type=Path,
        default=None,
        help="Path to save confusion matrix .npy",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Model directory not found: {args.model_dir}", file=sys.stderr)
        return 1
    if not args.npz.exists():
        print(f"NPZ file not found: {args.npz}", file=sys.stderr)
        return 1

    print(f"Loading model from {args.model_dir}...")
    model = load_jaeger_model(args.model_dir)

    print(f"Loading validation data from {args.npz}...")
    data = np.load(args.npz)
    features = data["features"]
    labels = data["labels"]

    if args.max_samples is not None:
        features = features[: args.max_samples]
        labels = labels[: args.max_samples]

    print(f"Running inference on {len(features)} samples...")
    preds = model.predict(features, batch_size=args.batch_size, verbose=1)

    if args.output_cm is not None:
        metrics, cm = compute_metrics(
            labels, preds, num_classes=int(labels.max()) + 1, return_cm=True
        )
        np.save(args.output_cm, cm)
        print(f"\nSaved confusion matrix to {args.output_cm}")
    else:
        metrics = compute_metrics(labels, preds, num_classes=int(labels.max()) + 1)

    print("\nResults:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")

    if args.output_csv is not None:
        import csv

        fieldnames = ["model_dir", "npz", "num_samples"] + sorted(metrics.keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            row = {
                "model_dir": str(args.model_dir),
                "npz": str(args.npz),
                "num_samples": len(features),
            }
            row.update(metrics)
            writer.writerow(row)
        print(f"\nWrote metrics to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
