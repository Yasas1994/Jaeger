"""Generate reliability training data from a trained classifier.

This module implements the post-classifier stage that:
  1. Runs inference on real training sequences and labels high-confidence
     correct predictions as ID (1) and high-confidence wrong predictions as
     OOD (0).
  2. Generates synthetic corrupted sequences, runs inference on them, and
     keeps high-confidence predictions as OOD (0).
  3. Writes the resulting records as CSV and converts them to an NPZ archive
     for the downstream reliability model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from jaeger.dataops.convert import convert_dataset
from jaeger.seqops.encode import CODON_ID, CODONS, process_string_train
from jaeger.seqops.synthetic import (
    apply_dinuc_shuffle,
    apply_kmer_shuffle,
    apply_shuffle,
    apply_subseq_repeat_window,
    apply_tandem_repeat_window,
)
from jaeger.utils.logging import get_logger

logger = get_logger(log_path=None, log_file=None, level=3)


def _read_csv_records(path: str) -> list[tuple[int, str]]:
    """Read a label,sequence CSV file (no header)."""
    records: list[tuple[int, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, seq = line.split(",", 1)
            records.append((int(label_str), seq))
    return records


def _write_csv(records: list[tuple[int, str]], path: str) -> None:
    """Write label,sequence records (no header)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for label, seq in records:
            f.write(f"{label},{seq}\n")


def _build_inference_dataset(
    csv_path: str,
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Build a dataset matching the classifier's training preprocessing."""
    ds = tf.data.TextLineDataset(csv_path, buffer_size=200)
    processor = process_string_train(
        codons=string_processor_config.get("codon") or CODONS,
        codon_num=string_processor_config.get("codon_id") or CODON_ID,
        codon_depth=string_processor_config.get("codon_depth") or 64,
        class_label_onehot=True,
        seq_onehot=string_processor_config.get("seq_onehot", True),
        num_classes=classifier_out_dim,
        crop_size=string_processor_config.get("crop_size"),
        input_type=string_processor_config.get("input_type", "translated"),
        masking=string_processor_config.get("masking", False),
        ngram_width=string_processor_config.get("ngram_width") or 3,
        shuffle_frames=string_processor_config.get("shuffle_frames", False),
    )
    ds = ds.map(processor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=tf.compat.v1.data.get_output_shapes(ds),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _extract_true_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """Collect one-hot labels from a dataset and return integer class indices."""
    labels = []
    for _, y in dataset:
        labels.append(y.numpy())
    labels = np.concatenate(labels, axis=0)
    if labels.ndim == 1:
        return labels.astype(np.int32)
    return np.argmax(labels, axis=1).astype(np.int32)


def _run_classifier_inference(
    classifier: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> np.ndarray:
    """Return softmax probabilities for every sample in *dataset*."""
    logits = classifier.predict(dataset, verbose=1)
    return tf.nn.softmax(logits).numpy()


def _select_id_ood_records(
    records: list[tuple[int, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Split records into ID (1) and OOD (0) based on classifier confidence."""
    id_records: list[tuple[int, str]] = []
    ood_records: list[tuple[int, str]] = []
    conf = np.max(y_pred, axis=1)
    pred_class = np.argmax(y_pred, axis=1)
    for i, (label, seq) in enumerate(records):
        if conf[i] < threshold:
            continue
        if pred_class[i] == y_true[i]:
            id_records.append((1, seq))
        else:
            ood_records.append((0, seq))
    return id_records, ood_records


def _normalize_perturbation_cfg(
    perturbations_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert flexible user config into a normalized list of perturbation specs.

    Supports legacy booleans (``shuffle: true``) and structured dicts
    (``shuffle: {enabled: true, mode: dinuc}``).
    """
    specs: list[dict[str, Any]] = []

    def _is_enabled(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return value.get("enabled", True)
        return bool(value)

    # ---- shuffle ----
    shuffle_value = perturbations_cfg.get("shuffle", True)
    if _is_enabled(shuffle_value):
        shuffle_dict = (
            shuffle_value if isinstance(shuffle_value, dict) else {"mode": "random"}
        )
        mode = shuffle_dict.get("mode", "random")
        if mode == "random":
            fn = apply_shuffle
            kwargs: dict[str, Any] = {}
        elif mode == "dinuc":
            fn = apply_dinuc_shuffle
            kwargs = {}
        elif mode == "kmer":
            fn = apply_kmer_shuffle
            kwargs = {"k": shuffle_dict.get("k", 2)}
        else:
            raise ValueError(f"Unsupported shuffle mode: {mode}")
        specs.append({"name": "shuffle", "fn": fn, "kwargs": kwargs})

    # ---- subsequence repeat ----
    subseq_value = perturbations_cfg.get("subseq_repeat", True)
    if _is_enabled(subseq_value):
        subseq_dict = subseq_value if isinstance(subseq_value, dict) else {}
        specs.append(
            {
                "name": "subseq_repeat",
                "fn": apply_subseq_repeat_window,
                "kwargs": {
                    "window_fraction": subseq_dict.get("window_fraction", 0.25),
                },
            }
        )

    # ---- tandem repeat ----
    tandem_value = perturbations_cfg.get("tandem_repeat", True)
    if _is_enabled(tandem_value):
        tandem_dict = tandem_value if isinstance(tandem_value, dict) else {}
        motif_range = tandem_dict.get("motif_length_range", [3, 10])
        specs.append(
            {
                "name": "tandem_repeat",
                "fn": apply_tandem_repeat_window,
                "kwargs": {
                    "motif_length_range": tuple(motif_range),
                    "window_fraction": tandem_dict.get("window_fraction", 0.25),
                    "num_repeats": tandem_dict.get("num_repeats"),
                },
            }
        )

    return specs


def _compute_perturbation_counts(
    records: list[tuple[int, str]],
    multiplier: float,
    specs: list[dict[str, Any]],
    perturbations_cfg: dict[str, Any],
) -> list[int]:
    """Return the number of synthetic samples to create for each perturbation spec.

    Count resolution order for each perturbation:
      1. ``count`` key in its config (absolute number).
      2. ``multiplier`` key in its config (fraction of *len(records)*).
      3. Even share of the remaining global *multiplier* budget across
         implicit (non-explicit) specs.
    """
    n = len(records)
    global_count = max(0, int(n * multiplier))
    if not specs:
        return []

    counts: list[int] = [0] * len(specs)
    explicit_indices: list[int] = []

    for i, spec in enumerate(specs):
        name = spec["name"]
        cfg = perturbations_cfg.get(name, {})
        if isinstance(cfg, dict):
            if "count" in cfg:
                counts[i] = max(0, int(cfg["count"]))
                explicit_indices.append(i)
                continue
            if "multiplier" in cfg:
                counts[i] = max(0, int(n * cfg["multiplier"]))
                explicit_indices.append(i)
                continue

    implicit_indices = [i for i in range(len(specs)) if i not in explicit_indices]
    if not implicit_indices:
        # All specs have explicit counts; honor them exactly.
        return counts

    allocated = sum(counts[i] for i in explicit_indices)
    remaining = max(0, global_count - allocated)
    per_implicit = remaining // len(implicit_indices)
    for i in implicit_indices:
        counts[i] = per_implicit
    leftover = remaining - sum(counts[i] for i in implicit_indices)
    for i in range(leftover):
        counts[implicit_indices[i % len(implicit_indices)]] += 1

    return counts


def _generate_synthetic_sequences(
    records: list[tuple[int, str]],
    multiplier: float,
    perturbations_cfg: dict[str, Any],
) -> list[str]:
    """Generate corrupted sequences from *records* according to *perturbations_cfg*."""
    synthetic: list[str] = []
    specs = _normalize_perturbation_cfg(perturbations_cfg)
    if not specs:
        return synthetic

    counts = _compute_perturbation_counts(records, multiplier, specs, perturbations_cfg)
    for spec, count in zip(specs, counts):
        for i in range(count):
            _, seq = records[i % len(records)]
            synthetic.append(spec["fn"](seq, **spec["kwargs"]))
    return synthetic


def _filter_synthetic_ood(
    classifier: tf.keras.Model,
    synthetic_seqs: list[str],
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    threshold: float,
    batch_size: int,
) -> list[tuple[int, str]]:
    """Keep corrupted sequences that the classifier predicts with high confidence."""
    if not synthetic_seqs:
        return []

    tmp_csv = Path(tempfile.gettempdir()) / "jaeger_synthetic_ood.csv"
    _write_csv([(0, s) for s in synthetic_seqs], str(tmp_csv))
    ds = _build_inference_dataset(
        str(tmp_csv),
        string_processor_config,
        classifier_out_dim,
        batch_size,
    )
    probs = _run_classifier_inference(classifier, ds)
    conf = np.max(probs, axis=1)
    return [(0, seq) for seq, c in zip(synthetic_seqs, conf) if c >= threshold]


def _convert_to_npz(
    csv_path: str,
    npz_path: str,
    string_processor_config: dict[str, Any],
    reliability_out_dim: int,
    model_cfg: dict[str, Any],
    generator_cfg: dict[str, Any] | None = None,
) -> None:
    """Convert a reliability CSV to the same NPZ format used for training."""
    generator_cfg = generator_cfg or {}
    sp_cfg = model_cfg.get("string_processor", {})
    crop_size = generator_cfg.get("crop_size")
    if crop_size is None:
        crop_size = string_processor_config.get("crop_size")
    if crop_size is None:
        crop_sizes = string_processor_config.get("crop_sizes")
        crop_size = max(crop_sizes) if crop_sizes else 500
    convert_dataset(
        input_path=csv_path,
        output_path=npz_path,
        format=string_processor_config.get("input_type", "translated"),
        crop_size=crop_size,
        stride=0,
        num_classes=reliability_out_dim,
        num_workers=1,
        one_hot=string_processor_config.get("seq_onehot", True),
        codon_map=sp_cfg.get("codon_id", "codon_id"),
        nucleotide_map=sp_cfg.get("nucleotide_map"),
        compress="fast",
        dtype="auto",
        pad=False,
    )


def generate_reliability_data(
    classifier: tf.keras.Model,
    raw_csv_path: str,
    output_dir: str,
    string_processor_config: dict[str, Any],
    model_cfg: dict[str, Any],
    classifier_out_dim: int,
    reliability_out_dim: int,
    batch_size: int,
    id_threshold: float = 0.8,
    synthetic_ood_threshold: float = 0.8,
    synthetic_ood_multiplier: float = 1.0,
    generator_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate reliability training data and return path metadata for the builder.

    Returns a dict shaped like ``builder._get_reliability_fragment_paths()``:
    ``{"train": {"paths": [...], ...}, "validation": {"paths": [...], ...}}``.
    """
    generator_cfg = generator_cfg or {}
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    train_npz = str(output_dir_path / "reliability_train.npz")
    val_npz = str(output_dir_path / "reliability_val.npz")
    if Path(train_npz).is_file() and Path(val_npz).is_file():
        logger.info(
            f"Reliability data already exists in {output_dir}; skipping generation"
        )
        return {
            "train": {"paths": [train_npz], "class": [], "label": []},
            "validation": {"paths": [val_npz], "class": [], "label": []},
        }

    raw_csv_paths = generator_cfg.get("raw_csv_paths") or {}
    train_csv_path = raw_csv_paths.get("train") or raw_csv_path
    val_csv_path = raw_csv_paths.get("val")

    if not train_csv_path:
        raise ValueError("A training raw CSV path must be provided")

    logger.info(f"Reading raw training sequences from {train_csv_path}")
    train_records = _read_csv_records(train_csv_path)
    if not train_records:
        raise ValueError(f"No records found in {train_csv_path}")

    # ---- ID / OOD from real data ----
    logger.info("Running classifier inference on real training sequences")
    train_ds = _build_inference_dataset(
        train_csv_path,
        string_processor_config,
        classifier_out_dim,
        batch_size,
    )
    y_true = _extract_true_labels(train_ds)
    probs = _run_classifier_inference(classifier, train_ds)
    id_records, real_ood_records = _select_id_ood_records(
        train_records, y_true, probs, id_threshold
    )
    logger.info(
        f"Selected {len(id_records)} ID and {len(real_ood_records)} "
        "high-confidence wrong OOD samples from real data"
    )

    # ---- synthetic OOD ----
    perturbations_cfg = generator_cfg.get("perturbations", {})
    synthetic_seqs = _generate_synthetic_sequences(
        train_records, synthetic_ood_multiplier, perturbations_cfg
    )
    synthetic_ood_records = _filter_synthetic_ood(
        classifier,
        synthetic_seqs,
        string_processor_config,
        classifier_out_dim,
        synthetic_ood_threshold,
        batch_size,
    )
    logger.info(f"Selected {len(synthetic_ood_records)} synthetic OOD samples")

    # ---- build train / validation records ----
    if val_csv_path:
        logger.info(f"Reading raw validation sequences from {val_csv_path}")
        val_source_records = _read_csv_records(val_csv_path)
        if not val_source_records:
            raise ValueError(f"No records found in {val_csv_path}")

        logger.info("Running classifier inference on real validation sequences")
        val_ds = _build_inference_dataset(
            val_csv_path,
            string_processor_config,
            classifier_out_dim,
            batch_size,
        )
        val_y_true = _extract_true_labels(val_ds)
        val_probs = _run_classifier_inference(classifier, val_ds)
        val_id_records, val_ood_records = _select_id_ood_records(
            val_source_records, val_y_true, val_probs, id_threshold
        )
        val_records = val_id_records + val_ood_records
        logger.info(
            f"Selected {len(val_id_records)} ID and {len(val_ood_records)} "
            "high-confidence wrong OOD samples from validation data"
        )

        train_records_out = id_records + real_ood_records + synthetic_ood_records
        rng = np.random.default_rng()
        rng.shuffle(train_records_out)
    else:
        all_records = id_records + real_ood_records + synthetic_ood_records
        rng = np.random.default_rng()
        rng.shuffle(all_records)

        val_fraction = generator_cfg.get("val_fraction", 0.1)
        n_val = int(len(all_records) * val_fraction)
        val_records = all_records[:n_val]
        train_records_out = all_records[n_val:]

    train_csv = str(output_dir_path / "reliability_train.csv")
    val_csv = str(output_dir_path / "reliability_val.csv")
    _write_csv(train_records_out, train_csv)
    _write_csv(val_records, val_csv)

    # ---- convert to NPZ ----
    train_npz = str(output_dir_path / "reliability_train.npz")
    val_npz = str(output_dir_path / "reliability_val.npz")
    _convert_to_npz(
        train_csv,
        train_npz,
        string_processor_config,
        reliability_out_dim,
        model_cfg,
        generator_cfg,
    )
    _convert_to_npz(
        val_csv,
        val_npz,
        string_processor_config,
        reliability_out_dim,
        model_cfg,
        generator_cfg,
    )

    return {
        "train": {
            "paths": [train_npz],
            "class": [],
            "label": [],
        },
        "validation": {
            "paths": [val_npz],
            "class": [],
            "label": [],
        },
    }
