"""Out-of-distribution (OOD) sequence generation.

Generates synthetic OOD CSV datasets from in-distribution (ID) CSV files using
the shared perturbation engine in ``jaeger.dataops.synthetic_perturbations``.
The output is compatible with ``jaeger utils encode-npz`` and
``jaeger utils combine-reliability``.
"""

from __future__ import annotations

from jaeger.utils.logging import get_logger

logger = get_logger(log_file=None, log_path=None, level=3)


def generate_ood_csv_core(**kwargs):
    """Generate an OOD-only CSV from an ID CSV using synthetic perturbations.

    This uses the same perturbation engine as the reliability data generation
    pipeline, so the resulting OOD CSV is compatible with
    ``jaeger utils encode-npz`` and ``jaeger utils combine-reliability``.
    """
    from pathlib import Path

    import numpy as np

    from jaeger.dataops.synthetic_perturbations import generate_synthetic_sequences
    from jaeger.seqops.crop import codons_to_nucleotides

    input_path = kwargs.get("input")
    output_path = kwargs.get("output")
    crop_size = kwargs.get("crop_size")
    crop_units = kwargs.get("crop_units", "codon")
    multiplier = kwargs.get("multiplier", 1.0)
    shuffle_proportion = kwargs.get("shuffle_proportion", 0.75)
    pre_shuffle = kwargs.get("pre_shuffle", False)
    shuffle_modes = list(kwargs.get("shuffle_mode", ("dinuc", "random")))
    kmer_k = kwargs.get("kmer_k", 2)
    subseq_repeat = kwargs.get("subseq_repeat", True)
    tandem_repeat = kwargs.get("tandem_repeat", True)
    n_stretch = kwargs.get("n_stretch", True)
    gc_shift = kwargs.get("gc_shift", False)
    gc_shift_rate_range = tuple(kwargs.get("gc_shift_rate_range", (0.05, 0.20)))
    pad_truncate = kwargs.get("pad_truncate", False)
    pad_length_range = tuple(kwargs.get("pad_length_range", (500, 1900)))
    iid_random = kwargs.get("iid_random", False)
    iid_random_gc_range = tuple(kwargs.get("iid_random_gc_range", (0.25, 0.75)))
    mix = kwargs.get("mix", True)
    window_fraction = kwargs.get("window_fraction", 0.25)
    motif_length_range = tuple(kwargs.get("motif_length_range", (3, 10)))
    num_repeats = kwargs.get("num_repeats", 20)
    n_fraction_range = tuple(kwargs.get("n_fraction_range", (0.3, 1.0)))
    max_stretches = kwargs.get("max_stretches", 3)
    generation_workers = kwargs.get("generation_workers")
    generation_chunk_size = kwargs.get("generation_chunk_size", 10000)
    source_sample_size = kwargs.get("source_sample_size")

    # Resolve crop size to nucleotides
    if crop_size is not None and crop_units == "codon":
        crop_size_nt = codons_to_nucleotides(crop_size)
    else:
        crop_size_nt = crop_size

    # Read input CSV
    records: list[tuple[int, str]] = []
    with open(input_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            label = int(parts[0])
            seq = parts[1]
            records.append((label, seq))

    if not records:
        raise ValueError(f"No records found in {input_path}")

    # Optionally subsample source records
    rng = np.random.default_rng()
    if source_sample_size is not None and source_sample_size < len(records):
        labels = np.array([label for label, _ in records], dtype=np.int32)
        distinct_labels = np.unique(labels)
        kept_indices: list[int] = []
        for label in distinct_labels:
            idx = np.where(labels == label)[0]
            label_frac = len(idx) / len(records)
            n_target = max(1, int(round(source_sample_size * label_frac)))
            if n_target >= len(idx):
                kept_indices.extend(idx.tolist())
            else:
                chosen = rng.choice(idx, size=n_target, replace=False)
                kept_indices.extend(chosen.tolist())
        while len(kept_indices) > source_sample_size:
            rng.shuffle(kept_indices)
            kept_indices.pop()
        rng.shuffle(kept_indices)
        records = [records[i] for i in kept_indices]
        logger.info(f"Sampled {len(records)} source records for OOD generation")

    # Build perturbations config
    perturbations_cfg = {
        "shuffle_before_perturbation": pre_shuffle,
        "shuffle_proportion": shuffle_proportion,
        "shuffle": {
            "enabled": True,
            "mode": shuffle_modes,
            "k": kmer_k,
        },
        "subseq_repeat": {
            "enabled": subseq_repeat,
            "window_fraction": window_fraction,
        },
        "tandem_repeat": {
            "enabled": tandem_repeat,
            "motif_length_range": list(motif_length_range),
            "window_fraction": window_fraction,
            "num_repeats": num_repeats,
        },
        "n_stretch": {
            "enabled": n_stretch,
            "n_fraction_range": list(n_fraction_range),
            "max_stretches": max_stretches,
        },
        "gc_shift": {
            "enabled": gc_shift,
            "rate_range": list(gc_shift_rate_range),
        },
        "pad_truncate": {
            "enabled": pad_truncate,
            "length_range": list(pad_length_range),
        },
        "iid_random": {
            "enabled": iid_random,
            "gc_range": list(iid_random_gc_range),
        },
        "mix": mix,
    }

    logger.info(
        f"Generating OOD sequences from {len(records)} input records "
        f"(multiplier={multiplier}, crop_size={crop_size_nt} nt)"
    )

    synthetic_seqs = generate_synthetic_sequences(
        records,
        multiplier,
        perturbations_cfg,
        crop_size=crop_size_nt,
        generation_chunk_size=generation_chunk_size,
        n_workers=generation_workers,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w") as fh:
        for seq in synthetic_seqs:
            fh.write(f"0,{seq}\n")
            total += 1

    logger.info(f"Wrote {total} OOD sequences to {output_path}")
