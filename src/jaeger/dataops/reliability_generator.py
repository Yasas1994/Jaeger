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

import gc
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import tensorflow as tf

from jaeger.dataops.convert import convert_dataset
from jaeger.dataops.synthetic_perturbations import generate_synthetic_sequences
from jaeger.seqops.encode import CODON_ID, CODONS, process_string_train
from jaeger.utils.logging import get_logger


logger = get_logger(log_path=None, log_file=None, level=3)


def _log_memory(label: str) -> None:
    """Log resident set size after forcing garbage collection."""
    gc.collect()
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    logger.info(f"Memory [{label}]: RSS={rss_gb:.2f} GB")


def _resolve_memory_budget_mb(fraction: float = 0.5) -> int:
    """Return *fraction* of currently available RAM in megabytes."""
    available_bytes = psutil.virtual_memory().available
    return max(256, int(available_bytes * fraction / (1024 * 1024)))


def _read_csv_records_with_ids(
    path: str,
) -> tuple[list[tuple[int, str]], list[str]]:
    """Read a raw CSV file and return (label, sequence) tuples plus seq_ids.

    If the CSV has a third column (or more), the last column is treated as the
    sequence identifier. Otherwise a numeric row index is used.
    """
    df = pl.read_csv(path, has_header=False, infer_schema_length=0)
    n_cols = df.shape[1]
    records: list[tuple[int, str]] = []
    seq_ids: list[str] = []
    for i, row in enumerate(df.iter_rows()):
        label = int(row[0])
        seq = row[1]
        seq_id = row[-1] if n_cols >= 3 else str(i)
        records.append((label, seq))
        seq_ids.append(seq_id)
    return records, seq_ids


def _read_csv_records(path: str) -> list[tuple[int, str]]:
    """Read a label,sequence CSV file (no header)."""
    records, _ = _read_csv_records_with_ids(path)
    return records


def _write_csv(records: list[tuple[int, str]], path: str) -> None:
    """Write label,sequence records (no header)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for label, seq in records:
            f.write(f"{label},{seq}\n")


def _resolve_reliability_crop_size(
    generator_cfg: dict[str, Any],
    string_processor_config: dict[str, Any],
) -> int:
    """Return the effective nucleotide crop size for reliability data."""
    crop_size = generator_cfg.get("crop_size")
    if crop_size is None:
        crop_size = string_processor_config.get("crop_size")
    if crop_size is None:
        crop_sizes = string_processor_config.get("crop_sizes")
        crop_size = max(crop_sizes) if crop_sizes else 500

    units = generator_cfg.get("units", "nuc")
    if units == "codon":
        crop_size = crop_size * 3
    elif units != "nuc":
        raise ValueError("units must be 'nuc' or 'codon'")

    return crop_size


def _build_inference_dataset(
    csv_path: str,
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    batch_size: int,
    crop_size: int | None = None,
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
        crop_size=crop_size,
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
    """Return softmax probabilities for every sample in *dataset*.

    Uses a ``tf.function``-wrapped model call and accumulates results batch by
    batch. This is substantially faster than ``classifier.predict`` for the
    small classifier head used during reliability data generation.
    """
    _, probs = _run_classifier_inference_logits(classifier, dataset)
    return probs


def _run_classifier_inference_logits(
    classifier: tf.keras.Model,
    dataset: tf.data.Dataset,
    n_records: int | None = None,
    description: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, softmax probabilities) for samples in *dataset*.

    Uses ``classifier.predict()`` so Keras can optimize the inference loop
    (asynchronous dataset iteration, C++ iterator, graph caching). This is
    substantially faster than a Python ``for`` loop for the same model.
    At most *n_records* samples are processed when *n_records* is provided.

    The *description* argument is kept for API compatibility but is no longer
    used; ``classifier.predict()`` manages its own progress output via
    ``verbose=0``.
    """
    # classifier.predict returns a NumPy array for a single-output model and
    # a list/dict for multi-output models. jaeger_classifier has a single
    # logits output, so we always get an ndarray here.
    logits = classifier.predict(dataset, verbose=1)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if n_records is not None:
        logits = logits[:n_records]
    probs = tf.nn.softmax(logits).numpy()
    return logits, probs


def _select_id_ood_from_probs(
    probs: np.ndarray,
    records: list[tuple[int, str]],
    threshold: float,
    id_records: list[tuple[int, str]],
    ood_records: list[tuple[int, str]],
) -> None:
    """Select ID/OOD records from a full (N, num_classes) probability matrix."""
    y_true = np.array([label for label, _ in records], dtype=np.int32)
    conf = np.max(probs, axis=1)
    pred_class = np.argmax(probs, axis=1)

    for i, (label, seq) in enumerate(records):
        if conf[i] < threshold:
            continue
        if pred_class[i] == y_true[i]:
            id_records.append((1, seq))
        else:
            ood_records.append((0, seq))


def _run_classifier_inference_streamed(
    classifier: tf.keras.Model,
    dataset: tf.data.Dataset,
    records: list[tuple[int, str]],
    threshold: float,
    id_records: list[tuple[int, str]],
    ood_records: list[tuple[int, str]],
    preds_csv_path: str | None = None,
    num_classes: int | None = None,
    seq_ids: list[str] | None = None,
) -> None:
    """Run classifier inference on *dataset* and split records into ID/OOD.

    The classifier call is wrapped in ``tf.function`` and the full (logits,
    probabilities) matrices are materialised in CPU memory. This is much faster
    than the previous per-batch Python loop while still keeping the memory
    footprint small for typical reliability datasets (a few tens of MB).

    Probabilities and logits are written to *preds_csv_path* (if provided) and
    high-confidence records are appended to *id_records* / *ood_records*.

    The prediction CSV stores, for each sample, the sequence id, the original
    integer label, the class logits, and the class probabilities. This makes
    the file self-describing and allows ID/OOD selection to be reproduced from
    it without rerunning inference.
    """
    n_records = len(records)

    # If a prediction CSV already exists, use it directly instead of re-running
    # the classifier.
    if preds_csv_path is not None and Path(preds_csv_path).exists():
        logger.info(f"Prediction file {preds_csv_path} exists; using it directly")
        try:
            with open(preds_csv_path, "r", newline="") as fh:
                first_line = fh.readline()
            has_header = first_line.startswith("seq_id")
            df = pl.read_csv(
                preds_csv_path, has_header=has_header, infer_schema_length=0
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"Could not load existing predictions from {preds_csv_path}: {exc}. "
                "Recomputing."
            )
        else:
            n_rows, n_cols = df.shape
            if n_rows != n_records:
                logger.warning(
                    f"Existing prediction file has {n_rows} rows but "
                    f"{n_records} records were expected. Recomputing."
                )
            elif num_classes is not None and n_cols in (
                num_classes,
                num_classes + 2,
                2 * num_classes + 2,
            ):
                # Determine which columns are metadata vs probabilities/logits.
                if n_cols == 2 * num_classes + 2:
                    labels_loaded = df[:, 1].cast(pl.Int32).to_numpy()
                    logits_loaded = (
                        df[:, 2 : 2 + num_classes].cast(pl.Float32).to_numpy()
                    )
                    probs = df[:, 2 + num_classes :].cast(pl.Float32).to_numpy()
                    loaded_seq_ids = df[:, 0].to_list()
                elif n_cols == num_classes + 2:
                    labels_loaded = df[:, 1].cast(pl.Int32).to_numpy()
                    probs = df[:, 2:].cast(pl.Float32).to_numpy()
                    logits_loaded = None
                    loaded_seq_ids = df[:, 0].to_list()
                else:
                    # Legacy probability-only format.
                    labels_loaded = None
                    probs = df.cast(pl.Float32).to_numpy()
                    logits_loaded = None
                    loaded_seq_ids = None

                expected_labels = np.array(
                    [label for label, _ in records], dtype=np.int32
                )

                if labels_loaded is not None and not np.array_equal(
                    labels_loaded, expected_labels
                ):
                    logger.warning(
                        "Prediction file labels do not match records. Recomputing."
                    )
                elif (
                    seq_ids is not None
                    and loaded_seq_ids is not None
                    and loaded_seq_ids != seq_ids
                ):
                    logger.warning(
                        "Prediction file sequence IDs do not match records. Recomputing."
                    )
                else:
                    # Upgrade legacy files (or files with row-index IDs) to the
                    # current self-describing format with real sequence IDs.
                    if preds_csv_path is not None and (
                        loaded_seq_ids is None
                        or (seq_ids is not None and loaded_seq_ids != seq_ids)
                    ):
                        _write_predictions_csv(
                            preds_csv_path,
                            seq_ids
                            if seq_ids is not None
                            else [str(i) for i in range(n_records)],
                            expected_labels,
                            probs,
                            logits=logits_loaded,
                        )

                    _select_id_ood_from_probs(
                        probs, records, threshold, id_records, ood_records
                    )
                    return
            else:
                logger.warning(
                    f"Unexpected prediction CSV shape ({n_rows}, {n_cols}) for "
                    f"num_classes={num_classes}. Recomputing."
                )

    if preds_csv_path is not None:
        Path(preds_csv_path).parent.mkdir(parents=True, exist_ok=True)

    logits, probs = _run_classifier_inference_logits(
        classifier,
        dataset,
        n_records=n_records,
        description="Classifying sequences",
    )

    if preds_csv_path is not None:
        expected_labels = np.array([label for label, _ in records], dtype=np.int32)
        loaded_seq_ids = (
            seq_ids if seq_ids is not None else [str(i) for i in range(n_records)]
        )
        _write_predictions_csv(
            preds_csv_path,
            loaded_seq_ids,
            expected_labels,
            probs,
            logits=logits,
        )

    _select_id_ood_from_probs(probs, records, threshold, id_records, ood_records)


def _prediction_csv_header(num_classes: int) -> list[str]:
    """Return the header row for a predictions CSV."""
    header = ["seq_id", "label"]
    header.extend(f"logit_{i}" for i in range(num_classes))
    header.extend(f"prob_{i}" for i in range(num_classes))
    return header


def _write_predictions_csv(
    path: str,
    seq_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray | None = None,
) -> None:
    """Write a self-describing predictions CSV using Polars.

    Columns are: ``seq_id``, ``label``, ``logit_0`` ... ``logit_{C-1}``,
    followed by ``prob_0`` ... ``prob_{C-1}``.
    """
    data: dict[str, Any] = {
        "seq_id": pl.Series(seq_ids, dtype=pl.Utf8),
        "label": pl.Series(labels, dtype=pl.Int32),
    }
    if logits is not None:
        for i in range(logits.shape[1]):
            data[f"logit_{i}"] = pl.Series(logits[:, i], dtype=pl.Float32)
    for i in range(probs.shape[1]):
        data[f"prob_{i}"] = pl.Series(probs[:, i], dtype=pl.Float32)
    pl.DataFrame(data).write_csv(path, include_header=True)


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


def _filter_synthetic_ood(
    classifier: tf.keras.Model,
    synthetic_seqs: Iterable[str],
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    threshold: float,
    inference_batch_size: int,
    crop_size: int | None = None,
    chunk_size: int = 10_000,
) -> list[tuple[int, str]]:
    """Keep corrupted sequences that the classifier predicts with high confidence.

    All synthetic sequences are materialised on disk first, then inference is
    run once with ``classifier.predict``. This lets us count how many high-
    confidence synthetic OOD samples survive the threshold before deciding how
    many real (ID + high-confidence OOD) samples to keep.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        synthetic_csv = str(Path(tmpdir) / "synthetic_ood.csv")

        _log_memory("before writing synthetic sequences to disk")
        total = 0
        with open(synthetic_csv, "w") as fh:
            for seq in synthetic_seqs:
                fh.write(f"0,{seq}\n")
                total += 1
                if total % chunk_size == 0:
                    _log_memory(f"after writing {total} synthetic sequences to disk")
        _log_memory(f"after writing all {total} synthetic sequences to disk")
        if total == 0:
            return []

        ds = _build_inference_dataset(
            synthetic_csv,
            string_processor_config,
            classifier_out_dim,
            inference_batch_size,
            crop_size=crop_size,
        )
        probs = _run_classifier_inference(classifier, ds)
        conf = np.max(probs, axis=1)
        records = _read_csv_records(synthetic_csv)
        out = [(0, seq) for (_, seq), c in zip(records, conf) if c >= threshold]
        _log_memory(
            f"after synthetic OOD inference: {len(out)} / {len(records)} "
            f"sequences above threshold {threshold}"
        )

    return out


def _downsample_to_match(
    real_records: list[tuple[int, str]],
    synthetic_records: list[tuple[int, str]],
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Return *real_records* downsampled to the size of *synthetic_records*.

    If there are already fewer real records than synthetic records, return
    *real_records* unchanged. The draw is stratified by label so the ID/OOD
    ratio is preserved as closely as possible.
    """
    n_real = len(real_records)
    n_synth = len(synthetic_records)
    if n_real <= n_synth or n_synth == 0:
        return real_records

    labels = np.array([label for label, _ in real_records], dtype=np.int32)
    distinct_labels = np.unique(labels)
    kept_indices: list[int] = []
    for label in distinct_labels:
        idx = np.where(labels == label)[0]
        label_frac = len(idx) / n_real
        n_target = int(round(n_synth * label_frac))
        if n_target > 0:
            chosen = rng.choice(idx, size=n_target, replace=False)
            kept_indices.extend(chosen.tolist())

    # Fill any rounding gap while preserving label proportions.
    while len(kept_indices) < n_synth:
        remaining = [i for i in range(n_real) if i not in kept_indices]
        if not remaining:
            break
        kept_indices.append(int(rng.choice(remaining)))

    rng.shuffle(kept_indices)
    return [real_records[i] for i in kept_indices]


def _sample_records_for_synthetic_generation(
    records: list[tuple[int, str]],
    target_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    """Stratified sample of *records* used only for creating synthetic corruptions.

    Preserves the label distribution so mix perturbations still see every class.
    """
    n = len(records)
    if n <= target_size:
        return records

    labels = np.array([label for label, _ in records], dtype=np.int32)
    distinct_labels = np.unique(labels)
    kept_indices: list[int] = []
    for label in distinct_labels:
        idx = np.where(labels == label)[0]
        label_frac = len(idx) / n
        n_target = max(1, int(round(target_size * label_frac)))
        if n_target >= len(idx):
            kept_indices.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=n_target, replace=False)
            kept_indices.extend(chosen.tolist())

    # Trim rounding overshoot while keeping at least one sample per label.
    while len(kept_indices) > target_size:
        rng.shuffle(kept_indices)
        kept_indices.pop()

    rng.shuffle(kept_indices)
    return [records[i] for i in kept_indices]


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
    crop_size = _resolve_reliability_crop_size(generator_cfg, string_processor_config)

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
    inference_batch_size: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate reliability training data and return path metadata for the builder.

    Returns a dict shaped like ``builder._get_reliability_fragment_paths()``:
    ``{"train": {"paths": [...], ...}, "validation": {"paths": [...], ...}}``.
    """
    generator_cfg = generator_cfg or {}
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Inference can use a larger batch than training because no gradients are kept.
    inference_batch_size = inference_batch_size or max(batch_size, 512)

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

    # Prediction CSVs are named after the raw input CSVs.
    train_preds_name = Path(train_csv_path).stem + "_preds.csv"

    crop_size = _resolve_reliability_crop_size(generator_cfg, string_processor_config)
    logger.info(
        f"Using reliability crop size {crop_size} (nucleotides) for classifier inference"
    )

    logger.info(f"Reading raw training sequences from {train_csv_path}")
    train_records, train_seq_ids = _read_csv_records_with_ids(train_csv_path)
    if not train_records:
        raise ValueError(f"No records found in {train_csv_path}")

    # ---- ID / OOD from real data ----
    logger.info("Running classifier inference on real training sequences")
    train_ds = _build_inference_dataset(
        train_csv_path,
        string_processor_config,
        classifier_out_dim,
        inference_batch_size,
        crop_size=crop_size,
    )
    id_records: list[tuple[int, str]] = []
    real_ood_records: list[tuple[int, str]] = []
    _run_classifier_inference_streamed(
        classifier,
        train_ds,
        train_records,
        id_threshold,
        id_records,
        real_ood_records,
        preds_csv_path=str(output_dir_path / train_preds_name),
        num_classes=classifier_out_dim,
        seq_ids=train_seq_ids,
    )
    logger.info(f"Wrote training predictions to {output_dir_path / train_preds_name}")
    logger.info(
        f"Selected {len(id_records)} ID and {len(real_ood_records)} "
        "high-confidence wrong OOD samples from real data"
    )

    # ---- synthetic OOD ----
    perturbations_cfg = generator_cfg.get("perturbations", {})

    # Size synthetic generation / filtering chunks from available RAM.
    available_mb = _resolve_memory_budget_mb(fraction=0.25)
    seq_len = crop_size or 500
    synthetic_chunk_size = generator_cfg.get(
        "synthetic_chunk_size",
        max(
            1000,
            min(10_000, int((available_mb * 1024 * 1024 * 0.5) / max(1, seq_len))),
        ),
    )
    generation_chunk_size = generator_cfg.get("synthetic_generation_chunk_size", 10_000)
    generation_workers = generator_cfg.get("synthetic_generation_workers")

    # Optionally sample the source records for synthetic generation. With millions
    # of training fragments the full list is huge and dominates generation time;
    # a stratified sample preserves class balance and keeps the same expected
    # number of synthetic sequences by scaling the multiplier.
    source_sample_size = generator_cfg.get("synthetic_source_sample_size")
    rng = np.random.default_rng()
    if source_sample_size is not None and source_sample_size < len(train_records):
        synthetic_source_records = _sample_records_for_synthetic_generation(
            train_records, source_sample_size, rng
        )
        adjusted_multiplier = synthetic_ood_multiplier * (
            len(train_records) / len(synthetic_source_records)
        )
        logger.info(
            f"Sampled {len(synthetic_source_records)} source records for synthetic "
            f"generation (multiplier adjusted {synthetic_ood_multiplier:.4f} -> "
            f"{adjusted_multiplier:.4f})"
        )
    else:
        synthetic_source_records = train_records
        adjusted_multiplier = synthetic_ood_multiplier

    _log_memory("before synthetic generation")
    synthetic_seqs = generate_synthetic_sequences(
        synthetic_source_records,
        adjusted_multiplier,
        perturbations_cfg,
        crop_size=crop_size,
        generation_chunk_size=generation_chunk_size,
        n_workers=generation_workers,
    )
    synthetic_ood_records = _filter_synthetic_ood(
        classifier,
        synthetic_seqs,
        string_processor_config,
        classifier_out_dim,
        synthetic_ood_threshold,
        inference_batch_size,
        crop_size=crop_size,
        chunk_size=synthetic_chunk_size,
    )
    _log_memory("after synthetic OOD filtering")
    logger.info(f"Selected {len(synthetic_ood_records)} synthetic OOD samples")

    # Balance real (ID + high-confidence OOD) samples against surviving
    # synthetic OOD samples so the reliability model does not drown in real
    # sequences when only a fraction of synthetic corruptions are kept.
    rng = np.random.default_rng()
    real_train_records = id_records + real_ood_records
    n_real_train_before = len(real_train_records)
    real_train_records = _downsample_to_match(
        real_train_records, synthetic_ood_records, rng
    )
    if len(real_train_records) < n_real_train_before:
        logger.info(
            f"Downsampled real training records from {n_real_train_before} to "
            f"{len(real_train_records)} to match {len(synthetic_ood_records)} "
            "synthetic OOD samples"
        )

    # ---- build train / validation records ----
    if val_csv_path:
        val_preds_name = Path(val_csv_path).stem + "_preds.csv"

        logger.info(f"Reading raw validation sequences from {val_csv_path}")
        val_source_records, val_seq_ids = _read_csv_records_with_ids(val_csv_path)
        if not val_source_records:
            raise ValueError(f"No records found in {val_csv_path}")

        logger.info("Running classifier inference on real validation sequences")
        val_ds = _build_inference_dataset(
            val_csv_path,
            string_processor_config,
            classifier_out_dim,
            inference_batch_size,
            crop_size=crop_size,
        )
        val_id_records: list[tuple[int, str]] = []
        val_ood_records: list[tuple[int, str]] = []
        _run_classifier_inference_streamed(
            classifier,
            val_ds,
            val_source_records,
            id_threshold,
            val_id_records,
            val_ood_records,
            preds_csv_path=str(output_dir_path / val_preds_name),
            num_classes=classifier_out_dim,
            seq_ids=val_seq_ids,
        )
        logger.info(
            f"Wrote validation predictions to {output_dir_path / val_preds_name}"
        )
        logger.info(
            f"Selected {len(val_id_records)} ID and {len(val_ood_records)} "
            "high-confidence wrong OOD samples from validation data"
        )

        if source_sample_size is not None and source_sample_size < len(
            val_source_records
        ):
            val_synthetic_source_records = _sample_records_for_synthetic_generation(
                val_source_records, source_sample_size, rng
            )
            val_adjusted_multiplier = synthetic_ood_multiplier * (
                len(val_source_records) / len(val_synthetic_source_records)
            )
        else:
            val_synthetic_source_records = val_source_records
            val_adjusted_multiplier = synthetic_ood_multiplier

        logger.info("Generating synthetic OOD samples from validation sequences")
        val_synthetic_seqs = generate_synthetic_sequences(
            val_synthetic_source_records,
            val_adjusted_multiplier,
            perturbations_cfg,
            crop_size=crop_size,
            generation_chunk_size=generation_chunk_size,
            n_workers=generation_workers,
        )
        val_synthetic_ood_records = _filter_synthetic_ood(
            classifier,
            val_synthetic_seqs,
            string_processor_config,
            classifier_out_dim,
            synthetic_ood_threshold,
            inference_batch_size,
            crop_size=crop_size,
            chunk_size=synthetic_chunk_size,
        )
        logger.info(
            f"Selected {len(val_synthetic_ood_records)} synthetic OOD samples "
            "from validation data"
        )
        val_real_records = val_id_records + val_ood_records
        n_val_real_before = len(val_real_records)
        val_real_records = _downsample_to_match(
            val_real_records, val_synthetic_ood_records, rng
        )
        if len(val_real_records) < n_val_real_before:
            logger.info(
                f"Downsampled real validation records from {n_val_real_before} to "
                f"{len(val_real_records)} to match {len(val_synthetic_ood_records)} "
                "synthetic OOD samples"
            )

        val_records = val_real_records + val_synthetic_ood_records
        train_records_out = real_train_records + synthetic_ood_records
        rng.shuffle(train_records_out)
    else:
        all_records = real_train_records + synthetic_ood_records
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

    sp_cfg = model_cfg.get("string_processor", {})
    max_memory_mb = generator_cfg.get("max_memory_mb")
    if max_memory_mb is None:
        # Use half of currently available RAM for the conversion stage.
        max_memory_mb = _resolve_memory_budget_mb(fraction=0.5)

    convert_dataset(
        input_path=train_csv,
        output_path=train_npz,
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
        max_memory_mb=max_memory_mb,
    )
    convert_dataset(
        input_path=val_csv,
        output_path=val_npz,
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
        max_memory_mb=max_memory_mb,
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
