"""Reliability threshold tuning.

Runs a reliability model on validation data, sweeps candidate cutoffs, and
selects the threshold that best separates in-distribution (ID=1) from
out-of-distribution (OOD=0) samples. This is invoked from ``jaeger train``
right after the reliability head is fitted; it is not a standalone script.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import expit


# Metrics that can be used to select the best threshold. Each maps to a key
# in the per-threshold row produced by :func:`tune_reliability_threshold`.
SUPPORTED_METRICS = ("f1-id", "f1-ood", "youden", "mcc")


def _to_label_vector(y: Any) -> np.ndarray:
    """Normalize a batch of labels to a 1-D integer vector.

    Genuine one-hot labels (last dim > 1) are argmaxed; binary labels shaped
    ``(batch, 1)`` are squeezed. Argmaxing a single-column array would zero
    every label, so the two cases must be distinguished.
    """
    arr = np.asarray(y)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        arr = np.argmax(arr, axis=-1)
    return np.asarray(arr, dtype=np.int32).reshape(-1)


def extract_labels_from_dataset(dataset: Any) -> np.ndarray:
    """Collect integer labels from a ``tf.data`` dataset of ``(x, y, ...)``.

    Handles both scalar/integer labels and one-hot labels (argmax). The
    dataset must be re-iterable and yield labels in a deterministic order
    (i.e. a non-shuffled validation split).
    """
    labels: list[np.ndarray] = []
    for batch in dataset:
        y = batch[1] if isinstance(batch, (list, tuple)) else batch
        if isinstance(y, dict):
            # Defensive: pick the first label tensor if targets are a dict.
            y = next(iter(y.values()))
        labels.append(_to_label_vector(y))
    if not labels:
        return np.array([], dtype=np.int32)
    return np.concatenate(labels, axis=0)


def _extract_reliability_output(outputs: Any) -> np.ndarray:
    """Pull the reliability head output out of a model prediction.

    The combined Jaeger model returns a dict of heads; the reliability model
    returns a single tensor. Accept both, plus list outputs (take the last).
    Always returns a 1-D array, including for single-sample batches.
    """
    if isinstance(outputs, dict):
        rel = outputs.get("reliability")
        if rel is None:
            raise ValueError(
                "Model outputs do not contain a 'reliability' head. "
                "Make sure the model includes the reliability head."
            )
        outputs = rel
    elif isinstance(outputs, (list, tuple)):
        outputs = outputs[-1]
    arr = np.asarray(outputs, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != 1:
            raise ValueError(
                f"Expected a single reliability score per sample, "
                f"got output width {arr.shape[1]}"
            )
        arr = arr[:, 0]
    return arr


def _to_probabilities(scores: np.ndarray) -> np.ndarray:
    """Convert raw logits to probabilities; pass through values already in [0, 1].

    The reliability head is trained with ``from_logits=True`` and a linear
    final layer, so its outputs are raw logits. Anything falling outside
    ``[0, 1]`` cannot be a probability and is mapped through a sigmoid
    (monotonic, so rankings and AUROC are unchanged).
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size and (float(scores.min()) < 0.0 or float(scores.max()) > 1.0):
        return expit(scores).astype(np.float32)
    return scores


def predict_reliability_scores(
    model: Any, data: Any, batch_size: int = 64
) -> np.ndarray:
    """Return a 1-D reliability probability per sample.

    ``data`` may be a numpy array or a ``tf.data`` dataset. When a dataset is
    passed it is consumed exactly once; pair it with
    :func:`extract_labels_from_dataset` on a re-iterable, non-shuffled split
    to keep scores and labels aligned (or use
    :func:`collect_scores_and_labels`, which needs only one pass).
    """
    outputs = model.predict(data, batch_size=batch_size, verbose=0)
    return _to_probabilities(_extract_reliability_output(outputs))


def collect_scores_and_labels(
    model: Any, dataset: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate ``dataset`` once, returning aligned ``(scores, labels)``.

    Collecting labels and predictions in two separate passes is unsafe when
    the dataset is shuffled without a fixed seed (as Jaeger validation splits
    are): each pass then yields a different order and scores no longer line
    up with their labels. A single pass pairs each batch's predictions with
    its own labels, so alignment holds regardless of iteration order.
    """
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in dataset:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(
                "collect_scores_and_labels expects (x, y) batches, "
                f"got {type(batch).__name__}"
            )
        x, y = batch[0], batch[1]
        outputs = model.predict_on_batch(x)
        scores.append(_to_probabilities(_extract_reliability_output(outputs)))
        labels.append(_to_label_vector(y))
    if not scores:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    return np.concatenate(scores), np.concatenate(labels)


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def _safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(average_precision_score(labels, scores))
    except Exception:
        return float("nan")


def _select_best_threshold(
    thresholds: Any, values: Any, metric_name: str = ""
) -> float:
    """Return the threshold that maximizes ``values`` (simple argmax)."""
    thresholds = np.asarray(thresholds)
    values = np.asarray(values)
    return float(thresholds[int(np.argmax(values))])


def tune_reliability_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1-id",
    min_threshold: float = 0.0,
    max_threshold: float = 0.95,
    step: float = 0.05,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    """Sweep thresholds and pick the best cutoff.

    Parameters
    ----------
    scores, labels:
        1-D arrays of reliability scores and binary labels (1=ID, 0=OOD),
        aligned element-wise.
    metric:
        One of ``f1-id``, ``f1-ood``, ``youden``, ``mcc``.
    min_threshold, max_threshold, step:
        Inclusive grid definition.

    Returns
    -------
    best_threshold:
        The cutoff maximizing ``metric``.
    rows:
        Per-threshold metric rows (also written to the sweep CSV).
    summary:
        Scalar ``auroc`` and ``auprc`` over the full score distribution.
    """
    from sklearn.metrics import f1_score

    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}; choose from {SUPPORTED_METRICS}."
        )

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError(
            f"scores/labels length mismatch: {scores.shape[0]} vs {labels.shape[0]}"
        )
    if np.unique(labels).size < 2:
        raise ValueError(
            "threshold tuning requires both ID (1) and OOD (0) labels; "
            f"got a single class ({int(labels[0]) if labels.size else 'empty'})"
        )

    thresholds = np.arange(min_threshold, max_threshold + step, step)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        id_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        ood_f1 = f1_score(labels, preds, pos_label=0, zero_division=0)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
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

    metric_key = {
        "f1-id": "f1_id",
        "f1-ood": "f1_ood",
        "youden": "youden_j",
        "mcc": "mcc",
    }[metric]
    best_idx = int(np.argmax([r[metric_key] for r in rows]))
    best_threshold = float(rows[best_idx]["threshold"])

    summary = {
        "auroc": _safe_auroc(labels, scores),
        "auprc": _safe_auprc(labels, scores),
    }
    return best_threshold, rows, summary


def write_threshold_outputs(
    reliability_dir: str | Path,
    best_threshold: float,
    rows: Iterable[dict[str, float]],
) -> tuple[Path, Path]:
    """Write ``reliability_threshold.txt`` and the sweep CSV under ``reliability_dir``."""
    out_dir = Path(reliability_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / "reliability_threshold.txt"
    best_path.write_text(f"{best_threshold}\n")

    rows = list(rows)
    sweep_path = out_dir / "reliability_threshold_sweep.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with sweep_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        sweep_path.write_text("")
    return best_path, sweep_path


def calibration_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, list[dict[str, float]]]:
    """Measure reliability calibration of the ID/OOD scores.

    Assumes ``scores`` are probability-like in ``[0, 1]`` (the reliability
    head's ID probability). Returns the Expected Calibration Error (ECE),
    the Brier score, and one row per equal-width bin with the mean predicted
    score, the empirical ID rate, and the sample count. A well-calibrated
    model has ECE close to 0 (empirical rate ~= mean predicted score).
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    n = scores.shape[0]
    if n == 0:
        return float("nan"), float("nan"), []

    brier = float(np.mean((scores - labels) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(np.sum(mask))
        center = (lo + hi) / 2.0
        if count == 0:
            rows.append(
                {
                    "bin_center": center,
                    "mean_pred": float("nan"),
                    "empirical_id_rate": float("nan"),
                    "count": 0,
                }
            )
            continue
        mean_pred = float(np.mean(scores[mask]))
        empirical = float(np.mean(labels[mask]))
        ece += (count / n) * abs(empirical - mean_pred)
        rows.append(
            {
                "bin_center": center,
                "mean_pred": mean_pred,
                "empirical_id_rate": empirical,
                "count": count,
            }
        )
    return float(ece), brier, rows


def write_calibration_outputs(
    reliability_dir: str | Path,
    rows: Iterable[dict[str, float]],
) -> Path:
    """Write the per-bin calibration curve CSV under ``reliability_dir``."""
    out_dir = Path(reliability_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "reliability_calibration.csv"
    rows = list(rows)
    if rows:
        fieldnames = list(rows[0].keys())
        with cal_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        cal_path.write_text("")
    return cal_path
