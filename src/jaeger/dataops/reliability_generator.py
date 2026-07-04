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

logger = get_logger(log_path=None, log_file=None, level=3)


def _resolve_memory_budget_mb(fraction: float = 0.5) -> int:
    """Return *fraction* of currently available RAM in megabytes."""
    available_bytes = psutil.virtual_memory().available
    return max(256, int(available_bytes * fraction / (1024 * 1024)))


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
    """Return softmax probabilities for every sample in *dataset*."""
    logits = classifier.predict(dataset, verbose=1)
    return tf.nn.softmax(logits).numpy()


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
) -> None:
    """Stream classifier inference over *dataset*, selecting ID/OOD per batch.

    Probabilities are appended to *preds_csv_path* (if provided) and
    high-confidence records are appended to *id_records* / *ood_records*.
    This avoids materialising the full (N, num_classes) softmax matrix.

    The prediction CSV stores, for each sample, the row index, the original
    integer label, and the class probabilities. This makes the file
    self-describing and allows ID/OOD selection to be reproduced from it
    without rerunning inference.
    """
    n_records = len(records)

    # If a prediction CSV already exists, use it directly instead of re-running
    # the classifier.
    if preds_csv_path is not None and Path(preds_csv_path).exists():
        logger.info(f"Prediction file {preds_csv_path} exists; using it directly")
        try:
            raw = np.loadtxt(preds_csv_path, delimiter=",", dtype=np.float32)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"Could not load existing predictions from {preds_csv_path}: {exc}. "
                "Recomputing."
            )
        else:
            if raw.shape[0] != n_records:
                logger.warning(
                    f"Existing prediction file has {raw.shape[0]} rows but "
                    f"{n_records} records were expected. Recomputing."
                )
            elif num_classes is not None and raw.shape[1] == num_classes + 2:
                # New self-describing format: seq_id, original_label, probs...
                seq_ids = raw[:, 0].astype(np.int32)
                labels_loaded = raw[:, 1].astype(np.int32)
                probs = raw[:, 2:]
                expected_labels = np.array(
                    [label for label, _ in records], dtype=np.int32
                )
                if not np.array_equal(seq_ids, np.arange(n_records, dtype=np.int32)):
                    logger.warning(
                        "Prediction file row order does not match records. Recomputing."
                    )
                elif not np.array_equal(labels_loaded, expected_labels):
                    logger.warning(
                        "Prediction file labels do not match records. Recomputing."
                    )
                else:
                    _select_id_ood_from_probs(
                        probs, records, threshold, id_records, ood_records
                    )
                    return
            elif num_classes is not None and raw.shape[1] == num_classes:
                # Legacy format: probabilities only.
                logger.info("Legacy prediction CSV detected; using row order.")
                _select_id_ood_from_probs(
                    raw, records, threshold, id_records, ood_records
                )
                return
            else:
                logger.warning(
                    f"Unexpected prediction CSV shape {raw.shape} for "
                    f"num_classes={num_classes}. Recomputing."
                )

    if preds_csv_path is not None:
        Path(preds_csv_path).parent.mkdir(parents=True, exist_ok=True)

    record_idx = 0

    for x, y in dataset:
        logits = classifier(x, training=False)
        probs = tf.nn.softmax(logits).numpy()
        batch_n = min(probs.shape[0], n_records - record_idx)
        if batch_n <= 0:
            break

        probs_batch = probs[:batch_n]

        # Integer labels from one-hot encoding.
        y_true = (
            y.numpy()[:batch_n]
            if y.shape.rank == 1
            else np.argmax(y.numpy(), axis=1)[:batch_n]
        )

        if preds_csv_path is not None:
            seq_ids = np.arange(
                record_idx, record_idx + batch_n, dtype=np.int32
            ).reshape(-1, 1)
            labels_out = y_true.astype(np.int32).reshape(-1, 1)
            out = np.concatenate([seq_ids, labels_out, probs_batch], axis=1)
            with open(preds_csv_path, "ab") as fh:
                np.savetxt(fh, out, delimiter=",")

        conf = np.max(probs_batch, axis=1)
        pred_class = np.argmax(probs_batch, axis=1)

        for i in range(batch_n):
            if conf[i] < threshold:
                continue
            _, seq = records[record_idx + i]
            if pred_class[i] == y_true[i]:
                id_records.append((1, seq))
            else:
                ood_records.append((0, seq))

        record_idx += batch_n
        if record_idx >= n_records:
            break


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
                results = pool.starmap(_generate_synthetic_chunk, args)
                for r in results:
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
    seqs: list[str],
    tmp_csv: str,
    string_processor_config: dict[str, Any],
    classifier_out_dim: int,
    threshold: float,
    inference_batch_size: int,
    crop_size: int | None = None,
) -> list[tuple[int, str]]:
    """Run inference on one chunk of synthetic sequences and keep high-conf OOD."""
    _write_csv([(0, s) for s in seqs], tmp_csv)
    ds = _build_inference_dataset(
        tmp_csv,
        string_processor_config,
        classifier_out_dim,
        inference_batch_size,
        crop_size=crop_size,
    )
    probs = _run_classifier_inference(classifier, ds)
    conf = np.max(probs, axis=1)
    return [(0, seq) for seq, c in zip(seqs, conf) if c >= threshold]


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

    Processes *synthetic_seqs* in bounded chunks so the full set of synthetic
    sequences is never materialised as a single list or inference dataset.
    """
    out: list[tuple[int, str]] = []
    buffer: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = str(Path(tmpdir) / "synthetic_ood_chunk.csv")
        for seq in synthetic_seqs:
            buffer.append(seq)
            if len(buffer) >= chunk_size:
                out.extend(
                    _filter_synthetic_ood_chunk(
                        classifier,
                        buffer,
                        tmp_csv,
                        string_processor_config,
                        classifier_out_dim,
                        threshold,
                        inference_batch_size,
                        crop_size=crop_size,
                    )
                )
                buffer = []
        if buffer:
            out.extend(
                _filter_synthetic_ood_chunk(
                    classifier,
                    buffer,
                    tmp_csv,
                    string_processor_config,
                    classifier_out_dim,
                    threshold,
                    inference_batch_size,
                    crop_size=crop_size,
                )
            )

    return out


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
    train_records = _read_csv_records(train_csv_path)
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
        val_source_records = _read_csv_records(val_csv_path)
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
