#!/usr/bin/env python3
"""Tune the Jaeger reliability cutoff on the reliability validation NPZ.

Loads the saved Jaeger model, runs inference on the reliability validation
features (ID=1, OOD=0), sweeps reliability thresholds, and reports the cutoff
that best separates ID from OOD samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

from jaeger.nnlib.v2.layers import MaskedBatchNorm, MaskedConv1D, ResidualBlock_wrapper
from jaeger.nnlib.v2.nmd import NMDLayer, NMDMerge


CUSTOM_OBJECTS = {
    "MaskedConv1D": MaskedConv1D,
    "MaskedBatchNorm": MaskedBatchNorm,
    "ResidualBlock_wrapper": ResidualBlock_wrapper,
    "NMDLayer": NMDLayer,
    "NMDMerge": NMDMerge,
}


def load_model(model_path: Path) -> tf.keras.Model:
    """Load a saved Jaeger model with custom layer classes.

    Accepts either the SavedModel graph directory itself or the parent
    directory created by ``builder.save_model`` (which contains a
    ``*_graph`` subdirectory).
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    graph_path = model_path
    if not (model_path / "saved_model.pb").exists():
        graph_dirs = sorted(model_path.glob("*_graph"))
        if not graph_dirs:
            raise FileNotFoundError(f"No SavedModel graph found under {model_path}")
        graph_path = graph_dirs[0]

    return tf.keras.models.load_model(str(graph_path), custom_objects=CUSTOM_OBJECTS)


def load_val_data(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load reliability validation features and binary ID/OOD labels."""
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    labels = data["labels"]
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    return features, labels.astype(np.int32)


def predict_reliability(
    model: tf.keras.Model, features: np.ndarray, batch_size: int = 64
) -> np.ndarray:
    """Return a 1-D reliability score for each validation sample."""
    outputs = model.predict(features, batch_size=batch_size, verbose=1)

    # The saved jaeger_model returns a dict of heads.
    if isinstance(outputs, dict):
        reliability = outputs.get("reliability")
        if reliability is None:
            raise ValueError(
                "Model outputs do not contain a 'reliability' head. "
                "Make sure the saved model includes the reliability model."
            )
    elif isinstance(outputs, list):
        reliability = outputs[-1]
    else:
        reliability = outputs

    reliability = np.asarray(reliability)
    if reliability.ndim > 1:
        reliability = np.squeeze(reliability)
    return reliability.astype(np.float32)


def _select_best_threshold(
    thresholds: np.ndarray, values: np.ndarray, metric_name: str
) -> float:
    """Return the threshold that maximizes the supplied metric values."""
    best_idx = int(np.argmax(values))
    return float(thresholds[best_idx])


def tune_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1-id",
    min_threshold: float = 0.0,
    max_threshold: float = 0.95,
    step: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """Sweep thresholds and return the best cutoff plus summary metrics.

    Parameters
    ----------
    scores:
        1-D reliability scores.
    labels:
        1-D binary labels (1=ID, 0=OOD).
    metric:
        One of "f1-id", "f1-ood", "youden", "mcc".
    min_threshold, max_threshold, step:
        Threshold grid definition.

    Returns
    -------
    best_threshold, metrics_dict
    """
    thresholds = np.arange(min_threshold, max_threshold + step, step)
    id_f1s: list[float] = []
    ood_f1s: list[float] = []
    youden_js: list[float] = []
    mccs: list[float] = []

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)

        id_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        ood_f1 = f1_score(labels, preds, pos_label=0, zero_division=0)
        id_f1s.append(id_f1)
        ood_f1s.append(ood_f1)

        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden_js.append(tpr + tnr - 1.0)

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
        mccs.append(mcc)

    metric_to_values = {
        "f1-id": np.array(id_f1s),
        "f1-ood": np.array(ood_f1s),
        "youden": np.array(youden_js),
        "mcc": np.array(mccs),
    }
    values = metric_to_values.get(metric, metric_to_values["f1-id"])
    best_threshold = _select_best_threshold(thresholds, values, metric)

    # Compute AUROC for reporting
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall_curve, precision_curve)

    metrics = {
        "best_threshold": best_threshold,
        "best_metric": metric,
        "best_f1_id": float(np.max(id_f1s)),
        "best_f1_ood": float(np.max(ood_f1s)),
        "best_youden_j": float(np.max(youden_js)),
        "best_mcc": float(np.max(mccs)),
        "auroc": float(auroc),
        "auprc": float(auprc),
    }
    return best_threshold, metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune Jaeger reliability cutoff on the reliability validation NPZ."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="f1-id",
        choices=["f1-id", "f1-ood", "youden", "mcc"],
        help="Metric to maximize when selecting the cutoff",
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.0, help="Minimum threshold to try"
    )
    parser.add_argument(
        "--max-threshold", type=float, default=0.95, help="Maximum threshold to try"
    )
    parser.add_argument("--step", type=float, default=0.05, help="Threshold grid step")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Inference batch size"
    )
    args = parser.parse_args()

    if not args.val_npz.exists():
        print(f"Validation NPZ not found: {args.val_npz}", file=sys.stderr)
        return 1

    print(f"Loading validation data from {args.val_npz}...")
    features, labels = load_val_data(args.val_npz)
    print(
        f"Validation samples: {len(labels)} (ID={np.sum(labels == 1)}, OOD={np.sum(labels == 0)})"
    )

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)

    print("Running reliability inference...")
    scores = predict_reliability(model, features, batch_size=args.batch_size)

    print(f"Tuning threshold with metric={args.metric}...")
    best_threshold, metrics = tune_threshold(
        scores,
        labels,
        metric=args.metric,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )

    print(f"Best reliability threshold: {best_threshold:.3f}")
    print(f"  Best F1-ID:  {metrics['best_f1_id']:.4f}")
    print(f"  Best F1-OOD: {metrics['best_f1_ood']:.4f}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{best_threshold}\n")

    sweep_csv = args.output.with_name(args.output.stem + "_sweep.csv")
    # Recompute sweep table for the CSV output
    thresholds = np.arange(
        args.min_threshold, args.max_threshold + args.step, args.step
    )
    _, _ = tune_threshold(
        scores,
        labels,
        metric=args.metric,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )
    # Build a fresh sweep table without recomputing everything twice in production
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        id_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        ood_f1 = f1_score(labels, preds, pos_label=0, zero_division=0)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "f1_id": float(id_f1),
                "f1_ood": float(ood_f1),
                "youden_j": float(tpr + tnr - 1.0),
                "mcc": float(mcc),
            }
        )
    pd.DataFrame(rows).to_csv(sweep_csv, index=False)
    print(f"Wrote threshold sweep to {sweep_csv}")
    print(f"Wrote best threshold to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
