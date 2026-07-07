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

import random
import tempfile
from collections.abc import Iterable
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import tensorflow as tf

from jaeger.dataops.convert import convert_dataset
from jaeger.seqops.encode import CODON_ID, CODONS, process_string_train
from jaeger.seqops.synthetic import (
    apply_dinuc_shuffle,
    apply_kmer_shuffle,
    apply_mix,
    apply_shuffle,
    apply_subseq_repeat_window,
    apply_tandem_repeat_window,
)
from jaeger.utils.logging import get_logger
from jaeger.utils.misc import track_ms

logger = get_logger(log_path=None, log_file=None, level=3)


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

    The classifier call is wrapped in ``tf.function`` so the graph is traced
    once and reused for every batch. Each batch is moved to CPU memory
    immediately to keep GPU memory bounded. At most *n_records* samples are
    processed when *n_records* is provided.

    If *description* is given, a Rich progress bar is shown over the batches.
    """

    # Cache the traced function on the classifier so repeated calls (e.g. many
    # synthetic-OOD chunks) do not pay the trace cost each time.
    infer_fn = getattr(classifier, "_jaeger_reliability_infer_fn", None)
    if infer_fn is None:

        @tf.function
        def _infer_fn(x):
            return classifier(x, training=False)

        classifier._jaeger_reliability_infer_fn = _infer_fn
        infer_fn = _infer_fn

    logits_list: list[np.ndarray] = []
    record_idx = 0
    batch_iter = dataset
    if description:
        batch_iter = track_ms(batch_iter, description=description, disable=False)
    for x, _ in batch_iter:
        batch_logits = infer_fn(x)
        if n_records is not None:
            batch_n = min(batch_logits.shape[0], n_records - record_idx)
            if batch_n <= 0:
                break
            logits_list.append(batch_logits[:batch_n].numpy())
            record_idx += batch_n
            if record_idx >= n_records:
                break
        else:
            logits_list.append(batch_logits.numpy())

    logits = np.concatenate(logits_list, axis=0)
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
        modes = shuffle_dict.get("mode", "random")
        if isinstance(modes, str):
            modes = [modes]
        for mode in modes:
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

    # ---- mix / chimera ----
    mix_value = perturbations_cfg.get("mix", False)
    if _is_enabled(mix_value):
        mix_dict = mix_value if isinstance(mix_value, dict) else {}
        specs.append(
            {
                "name": "mix",
                "fn": apply_mix,
                "n_segments": mix_dict.get("n_segments", 2),
                "kwargs": {},
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


def _make_mix_chimera(
    records: list[tuple[int, str]],
    n_segments: int,
    crop_size: int | None = None,
) -> str:
    """Build a chimera from *n_segments* sequences belonging to distinct classes."""
    distinct_labels = list({label for label, _ in records})
    if len(distinct_labels) < n_segments:
        raise ValueError(
            f"mix perturbation requires at least {n_segments} distinct classes, "
            f"found {len(distinct_labels)}"
        )

    selected_labels = random.sample(distinct_labels, k=n_segments)
    label_to_seqs: dict[int, list[str]] = {}
    for label, seq in records:
        label_to_seqs.setdefault(label, []).append(seq)

    selected_seqs = [random.choice(label_to_seqs[label]) for label in selected_labels]
    return apply_mix(selected_seqs, output_length=crop_size)


# Global shared by worker processes forked for synthetic sequence generation.
_WORKER_RECORDS: list[tuple[int, str]] | None = None


def _init_synthetic_worker(records: list[tuple[int, str]], seed: int) -> None:
    """Seed RNGs and share the source records with a worker process."""
    global _WORKER_RECORDS
    _WORKER_RECORDS = records
    random.seed(seed)
    np.random.seed(seed)


def _generate_synthetic_chunk(
    spec_name: str,
    fn_name: str,
    kwargs: dict[str, Any],
    count: int,
    crop_size: int | None,
    n_segments: int | None,
    seed: int,
) -> list[str]:
    """Generate a chunk of synthetic sequences in a worker process."""
    random.seed(seed)
    np.random.seed(seed)
    records = _WORKER_RECORDS
    if records is None:
        raise RuntimeError("synthetic worker not initialized with records")

    out: list[str] = []
    n_records = len(records)
    if spec_name == "mix":
        for _ in range(count):
            out.append(_make_mix_chimera(records, n_segments, crop_size))
    else:
        fn = globals()[fn_name]
        for i in range(count):
            _, seq = records[i % n_records]
            out.append(fn(seq, **kwargs))
    return out


def _generate_synthetic_chunk_wrapper(
    args: tuple[str, str, dict[str, Any], int, int | None, int | None, int],
) -> list[str]:
    """Unpack arguments for :func:`_generate_synthetic_chunk`.

    Needed because ``Pool.imap_unordered`` passes a single argument.
    """
    return _generate_synthetic_chunk(*args)


def _generate_synthetic_sequences(
    records: list[tuple[int, str]],
    multiplier: float,
    perturbations_cfg: dict[str, Any],
    crop_size: int | None = None,
) -> Iterable[str]:
    """Yield corrupted sequences from *records* according to *perturbations_cfg*.

    Returning a generator instead of a single list keeps peak memory bounded
    when the caller processes synthetic sequences in chunks.
    """
    specs = _normalize_perturbation_cfg(perturbations_cfg)
    if not specs:
        return

    counts = _compute_perturbation_counts(records, multiplier, specs, perturbations_cfg)

    n_workers = max(1, min(cpu_count(), max(counts, default=0)))
    use_pool = n_workers > 1 and any(c >= n_workers * 2 for c in counts)

    if use_pool:
        base_seed = random.randint(0, 2**31 - 1)
        with Pool(
            processes=n_workers,
            initializer=_init_synthetic_worker,
            initargs=(records, base_seed),
        ) as pool:
            for spec, count in zip(specs, counts):
                if count <= 0:
                    continue
                spec_name = spec["name"]
                if spec_name == "mix":
                    fn_name = ""
                    kwargs: dict[str, Any] = {}
                    n_segments = spec["n_segments"]
                else:
                    fn_name = spec["fn"].__name__
                    kwargs = spec["kwargs"]
                    n_segments = None

                worker_chunk_size = max(1, count // n_workers)
                chunks = []
                remaining = count
                for i in range(n_workers):
                    if remaining <= 0:
                        break
                    c = min(worker_chunk_size, remaining)
                    chunks.append(c)
                    remaining -= c

                args = [
                    (
                        spec_name,
                        fn_name,
                        kwargs,
                        c,
                        crop_size,
                        n_segments,
                        base_seed + i,
                    )
                    for i, c in enumerate(chunks)
                ]
                # Stream results as they complete instead of materialising the
                # whole spec count in memory with starmap.
                for r in pool.imap_unordered(
                    _generate_synthetic_chunk_wrapper, args, chunksize=1
                ):
                    for seq in r:
                        yield seq
    else:
        for spec, count in zip(specs, counts):
            if count <= 0:
                continue
            if spec["name"] == "mix":
                n_segments = spec["n_segments"]
                for _ in range(count):
                    yield _make_mix_chimera(records, n_segments, crop_size)
            else:
                fn = spec["fn"]
                kwargs = spec["kwargs"]
                n_records = len(records)
                for i in range(count):
                    _, seq = records[i % n_records]
                    yield fn(seq, **kwargs)


def _filter_synthetic_ood_chunk(
    classifier: tf.keras.Model,
    csv_path: str,
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    threshold: float,
    inference_batch_size: int,
    crop_size: int | None = None,
) -> list[tuple[int, str]]:
    """Run inference on one chunk of synthetic sequences and keep high-conf OOD."""
    ds = _build_inference_dataset(
        csv_path,
        string_processor_config,
        classifier_out_dim,
        inference_batch_size,
        crop_size=crop_size,
    )
    probs = _run_classifier_inference(classifier, ds)
    conf = np.max(probs, axis=1)
    records = _read_csv_records(csv_path)
    return [(0, seq) for (_, seq), c in zip(records, conf) if c >= threshold]


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

    Processes *synthetic_seqs* in bounded chunks. Each chunk is written to a
    temporary CSV on disk, so the parent process never holds more than
    *chunk_size* synthetic sequences in memory at once. Kept high-confidence
    records are also accumulated in a temporary CSV and only read into a list
    at the end.
    """
    out: list[tuple[int, str]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        batch_csv = str(Path(tmpdir) / "synthetic_ood_batch.csv")
        kept_csv = str(Path(tmpdir) / "kept_synthetic_ood.csv")

        batch_fh = open(batch_csv, "w")
        batch_count = 0
        try:
            for seq in synthetic_seqs:
                batch_fh.write(f"0,{seq}\n")
                batch_count += 1
                if batch_count >= chunk_size:
                    batch_fh.close()
                    _flush_synthetic_ood_batch(
                        classifier,
                        batch_csv,
                        kept_csv,
                        string_processor_config,
                        classifier_out_dim,
                        threshold,
                        inference_batch_size,
                        crop_size=crop_size,
                    )
                    batch_fh = open(batch_csv, "w")
                    batch_count = 0
        finally:
            if not batch_fh.closed:
                batch_fh.close()

        if batch_count > 0:
            _flush_synthetic_ood_batch(
                classifier,
                batch_csv,
                kept_csv,
                string_processor_config,
                classifier_out_dim,
                threshold,
                inference_batch_size,
                crop_size=crop_size,
            )

        if Path(kept_csv).exists():
            out = _read_csv_records(kept_csv)

    return out


def _flush_synthetic_ood_batch(
    classifier: tf.keras.Model,
    batch_csv: str,
    kept_csv: str,
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    threshold: float,
    inference_batch_size: int,
    crop_size: int | None = None,
) -> None:
    """Run inference on *batch_csv* and append kept records to *kept_csv*."""
    kept = _filter_synthetic_ood_chunk(
        classifier,
        batch_csv,
        string_processor_config,
        classifier_out_dim,
        threshold,
        inference_batch_size,
        crop_size=crop_size,
    )
    with open(kept_csv, "a") as kept_fh:
        for _, seq in kept:
            kept_fh.write(f"0,{seq}\n")


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
            min(100_000, int((available_mb * 1024 * 1024 * 0.5) / max(1, seq_len))),
        ),
    )

    synthetic_seqs = _generate_synthetic_sequences(
        train_records,
        synthetic_ood_multiplier,
        perturbations_cfg,
        crop_size=crop_size,
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
    logger.info(f"Selected {len(synthetic_ood_records)} synthetic OOD samples")

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

        logger.info("Generating synthetic OOD samples from validation sequences")
        val_synthetic_seqs = _generate_synthetic_sequences(
            val_source_records,
            synthetic_ood_multiplier,
            perturbations_cfg,
            crop_size=crop_size,
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
        val_records = val_id_records + val_ood_records + val_synthetic_ood_records

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
