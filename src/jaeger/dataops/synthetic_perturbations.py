"""TF-free synthetic sequence perturbations.

This module contains only CPU-side string operations used to generate
corrupted/out-of-distribution sequences for reliability training. It deliberately
does not import TensorFlow so that it can be launched as a stand-alone subprocess
after the parent process has initialized CUDA.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from jaeger.seqops.synthetic import (
    apply_dinuc_shuffle,
    apply_gc_shift,
    apply_random_seq,
    apply_kmer_shuffle,
    apply_mix,
    apply_n_stretch,
    apply_pad_truncate,
    apply_shuffle,
    apply_subseq_repeat_window,
    apply_tandem_repeat_window,
)


def _normalize_perturbation_cfg(
    perturbations_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert flexible user config into a normalized list of perturbation specs."""
    specs: list[dict[str, Any]] = []
    global_pre_shuffle = perturbations_cfg.get("shuffle_before_perturbation", False)

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
                "pre_shuffle": subseq_dict.get(
                    "shuffle_before_perturbation", global_pre_shuffle
                ),
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
                "pre_shuffle": tandem_dict.get(
                    "shuffle_before_perturbation", global_pre_shuffle
                ),
            }
        )

    # ---- N stretch (ambiguous-base corruption) ----
    # Opt-in: unlike the other perturbations this one is disabled unless the
    # config explicitly enables it, so existing configs keep their behaviour.
    n_stretch_value = perturbations_cfg.get("n_stretch", False)
    if _is_enabled(n_stretch_value):
        n_stretch_dict = n_stretch_value if isinstance(n_stretch_value, dict) else {}
        specs.append(
            {
                "name": "n_stretch",
                "fn": apply_n_stretch,
                "kwargs": {
                    "n_fraction_range": tuple(
                        n_stretch_dict.get("n_fraction_range", [0.3, 1.0])
                    ),
                    "max_stretches": n_stretch_dict.get("max_stretches", 3),
                    "point_n_share": n_stretch_dict.get("point_n_share", 0.2),
                },
                "pre_shuffle": n_stretch_dict.get(
                    "shuffle_before_perturbation", global_pre_shuffle
                ),
            }
        )

    # ---- GC shift (composition corruption) ----
    # Opt-in: disabled unless the config explicitly enables it.
    gc_shift_value = perturbations_cfg.get("gc_shift", False)
    if _is_enabled(gc_shift_value):
        gc_shift_dict = gc_shift_value if isinstance(gc_shift_value, dict) else {}
        specs.append(
            {
                "name": "gc_shift",
                "fn": apply_gc_shift,
                "kwargs": {
                    "rate_range": tuple(gc_shift_dict.get("rate_range", (0.05, 0.20))),
                },
                "pre_shuffle": gc_shift_dict.get(
                    "shuffle_before_perturbation", global_pre_shuffle
                ),
            }
        )

    # ---- pad + truncate (short-contig padding corruption) ----
    # Opt-in: disabled unless the config explicitly enables it.
    pad_truncate_value = perturbations_cfg.get("pad_truncate", False)
    if _is_enabled(pad_truncate_value):
        pad_truncate_dict = (
            pad_truncate_value if isinstance(pad_truncate_value, dict) else {}
        )
        specs.append(
            {
                "name": "pad_truncate",
                "fn": apply_pad_truncate,
                "kwargs": {
                    "length_range": tuple(
                        pad_truncate_dict.get("length_range", (500, 1900))
                    ),
                    "pad_char": pad_truncate_dict.get("pad_char", "M"),
                    "output_length": pad_truncate_dict.get("output_length"),
                },
                "pre_shuffle": pad_truncate_dict.get(
                    "shuffle_before_perturbation", global_pre_shuffle
                ),
            }
        )

    # ---- iid random sequence (structureless corruption) ----
    # Opt-in: disabled unless the config explicitly enables it.
    iid_random_value = perturbations_cfg.get("iid_random", False)
    if _is_enabled(iid_random_value):
        iid_random_dict = iid_random_value if isinstance(iid_random_value, dict) else {}
        specs.append(
            {
                "name": "iid_random",
                "fn": apply_random_seq,
                "kwargs": {
                    "gc_range": tuple(iid_random_dict.get("gc_range", (0.25, 0.75))),
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


def _resolve_category_proportions(
    proportions_cfg: dict[str, Any],
    category_names: list[str],
) -> dict[str, float]:
    """Normalize user-provided category proportions to sum to 1.0."""
    raw = {name: float(proportions_cfg.get(name, 0.0)) for name in category_names}
    total = sum(raw.values())
    if total <= 0:
        return {name: 1.0 / len(category_names) for name in category_names}
    return {name: v / total for name, v in raw.items()}


def _compute_perturbation_counts(
    records: list[tuple[int, str]],
    multiplier: float,
    specs: list[dict[str, Any]],
    perturbations_cfg: dict[str, Any],
) -> list[int]:
    """Return the number of synthetic samples to create for each perturbation spec.

    Explicit ``count`` or ``multiplier`` per perturbation take precedence. The
    remaining budget is distributed either equally across implicit specs (default,
    legacy behaviour) or according to category proportions when ``proportions`` or
    ``shuffle_proportion`` are configured.
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
        return counts

    allocated = sum(counts[i] for i in explicit_indices)
    remaining = max(0, global_count - allocated)

    use_category_proportions = (
        "proportions" in perturbations_cfg or "shuffle_proportion" in perturbations_cfg
    )

    if use_category_proportions:
        # Group implicit specs by perturbation category.
        implicit_categories: dict[str, list[int]] = {}
        for i in implicit_indices:
            implicit_categories.setdefault(specs[i]["name"], []).append(i)
        category_names = list(implicit_categories.keys())

        if "proportions" in perturbations_cfg:
            proportions = _resolve_category_proportions(
                perturbations_cfg["proportions"], category_names
            )
        else:
            shuffle_prop = float(perturbations_cfg["shuffle_proportion"])
            other_names = [name for name in category_names if name != "shuffle"]
            if other_names:
                other_prop = (1.0 - shuffle_prop) / len(other_names)
                proportions = {name: other_prop for name in other_names}
            else:
                proportions = {}
            proportions["shuffle"] = shuffle_prop

        # Allocate remaining budget to categories.
        category_remaining: dict[str, int] = {}
        for name in category_names:
            category_remaining[name] = int(remaining * proportions[name])
        distrib_remainder = remaining - sum(category_remaining.values())
        sorted_names = sorted(
            category_names, key=lambda x: proportions[x], reverse=True
        )
        for i in range(distrib_remainder):
            category_remaining[sorted_names[i % len(sorted_names)]] += 1

        # Split each category's budget equally across its implicit specs.
        for name, indices in implicit_categories.items():
            cat_count = category_remaining[name]
            per_spec = cat_count // len(indices)
            for i in indices:
                counts[i] = per_spec
            leftover = cat_count - sum(counts[i] for i in indices)
            for i in range(leftover):
                counts[indices[i % len(indices)]] += 1
    else:
        # Legacy equal split across implicit specs.
        per_implicit = remaining // len(implicit_indices)
        for i in implicit_indices:
            counts[i] = per_implicit
        leftover = remaining - sum(counts[i] for i in implicit_indices)
        for i in range(leftover):
            counts[implicit_indices[i % len(implicit_indices)]] += 1

    return counts


def _build_label_index(
    records: list[tuple[int, str]],
) -> tuple[dict[int, list[str]], list[int]]:
    """Return a label -> sequences map and the list of distinct labels."""
    label_to_seqs: dict[int, list[str]] = {}
    for label, seq in records:
        label_to_seqs.setdefault(label, []).append(seq)
    distinct_labels = list(label_to_seqs.keys())
    return label_to_seqs, distinct_labels


def _make_mix_chimera(
    label_to_seqs: dict[int, list[str]],
    distinct_labels: list[int],
    n_segments: int,
    crop_size: int | None = None,
) -> str:
    """Build a chimera from *n_segments* sequences belonging to distinct classes."""
    if len(distinct_labels) < n_segments:
        raise ValueError(
            f"mix perturbation requires at least {n_segments} distinct classes, "
            f"found {len(distinct_labels)}"
        )

    selected_labels = random.sample(distinct_labels, k=n_segments)
    selected_seqs = [random.choice(label_to_seqs[label]) for label in selected_labels]
    return apply_mix(selected_seqs, output_length=crop_size)


def _generate_chunk_serial(
    records: list[tuple[int, str]],
    spec: dict[str, Any],
    count: int,
    crop_size: int | None,
    seed: int,
) -> list[str]:
    """Generate *count* sequences for a single spec."""
    random.seed(seed)
    np.random.seed(seed)
    out: list[str] = []
    spec_name = spec["name"]
    n_records = len(records)
    if spec_name == "mix":
        label_to_seqs, distinct_labels = _build_label_index(records)
        n_segments = spec["n_segments"]
        for _ in range(count):
            out.append(
                _make_mix_chimera(label_to_seqs, distinct_labels, n_segments, crop_size)
            )
    else:
        fn = spec["fn"]
        kwargs = spec["kwargs"]
        if spec_name == "pad_truncate" and kwargs.get("output_length") is None:
            # pad back to the record/crop length so masked positions fill the
            # canvas exactly like inference-time padding of short contigs
            kwargs = {**kwargs, "output_length": crop_size}
        pre_shuffle = spec.get("pre_shuffle", False)
        for i in range(count):
            _, seq = records[i % n_records]
            if pre_shuffle:
                seq = apply_shuffle(seq)
            out.append(fn(seq, **kwargs))
    return out


def _dump_records_to_temp(records: list[tuple[int, str]]) -> str:
    """Write records to a temporary pickle file and return the path."""
    import pickle

    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="jaeger_synthetic_records_")
    with open(fd, "wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _load_records_from_temp(path: str) -> list[tuple[int, str]]:
    """Load records from a temporary pickle file."""
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _cleanup_temp(path: str | None) -> None:
    """Remove a temporary file if it exists."""
    if path and Path(path).exists():
        Path(path).unlink()


def _write_chunk(path: str, sequences: list[str]) -> None:
    """Write sequences to *path*, one per line."""
    with open(path, "w") as fh:
        for seq in sequences:
            fh.write(f"{seq}\n")


def _read_chunk(path: str) -> list[str]:
    """Read sequences from *path*, one per line."""
    with open(path, "r") as fh:
        return [line.rstrip("\n") for line in fh]


def _run_subprocess_worker(
    records_path: str,
    output_path: str,
    spec_name: str,
    fn_name: str,
    kwargs: dict[str, Any],
    count: int,
    crop_size: int | None,
    n_segments: int | None,
    seed: int,
    pre_shuffle: bool = False,
) -> str:
    """Launch a stand-alone subprocess worker and return its output path."""
    cmd = [
        sys.executable,
        "-m",
        "jaeger.dataops.synthetic_perturbations",
        "--worker",
        "--records-path",
        records_path,
        "--output-path",
        output_path,
        "--spec-name",
        spec_name,
        "--fn-name",
        fn_name,
        "--kwargs-json",
        json.dumps(kwargs),
        "--count",
        str(count),
        "--seed",
        str(seed),
    ]
    if crop_size is not None:
        cmd.extend(["--crop-size", str(crop_size)])
    if n_segments is not None:
        cmd.extend(["--n-segments", str(n_segments)])
    if pre_shuffle:
        cmd.append("--pre-shuffle")

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def generate_synthetic_sequences(
    records: list[tuple[int, str]],
    multiplier: float,
    perturbations_cfg: dict[str, Any],
    crop_size: int | None = None,
    generation_chunk_size: int = 10_000,
    n_workers: int | None = None,
) -> Iterable[str]:
    """Yield corrupted sequences from *records* according to *perturbations_cfg*."""
    specs = _normalize_perturbation_cfg(perturbations_cfg)
    if not specs:
        return

    counts = _compute_perturbation_counts(records, multiplier, specs, perturbations_cfg)

    if n_workers is None:
        # Serial generation avoids per-chunk subprocess overhead. With a sampled
        # source set (see reliability_generator.py) it is usually the fastest
        # and most memory-stable option after TF/CUDA is loaded.
        n_workers = 1
    n_workers = max(1, min(n_workers, cpu_count(), max(counts, default=0)))
    use_pool = n_workers > 1 and any(c >= n_workers * 2 for c in counts)

    base_seed = random.randint(0, 2**31 - 1)

    if use_pool:
        temp_records_path = _dump_records_to_temp(records)
        try:
            tmpdir = tempfile.mkdtemp(prefix="jaeger_synthetic_chunks_")
            try:
                tasks: list[
                    tuple[
                        str,
                        str,
                        str,
                        dict[str, Any],
                        int,
                        int | None,
                        int | None,
                        int,
                        bool,
                    ]
                ] = []
                task_index = 0
                for spec, count in zip(specs, counts):
                    if count <= 0:
                        continue
                    spec_name = spec["name"]
                    fn_name = "" if spec_name == "mix" else spec["fn"].__name__
                    kwargs: dict[str, Any] = (
                        {} if spec_name == "mix" else spec["kwargs"]
                    )
                    n_segments = spec.get("n_segments")
                    pre_shuffle = spec.get("pre_shuffle", False)
                    seed_offset = 0
                    for start in range(0, count, generation_chunk_size):
                        sub_count = min(generation_chunk_size, count - start)
                        output_path = str(Path(tmpdir) / f"chunk_{task_index:08d}.txt")
                        tasks.append(
                            (
                                temp_records_path,
                                output_path,
                                spec_name,
                                fn_name,
                                kwargs,
                                sub_count,
                                crop_size,
                                n_segments,
                                base_seed + seed_offset,
                                pre_shuffle,
                            )
                        )
                        seed_offset += 1
                        task_index += 1

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(_run_subprocess_worker, *task): task[1]
                        for task in tasks
                    }
                    for future in as_completed(futures):
                        output_path = future.result()
                        for seq in _read_chunk(output_path):
                            yield seq
                        _cleanup_temp(output_path)
            finally:
                if Path(tmpdir).exists():
                    for p in Path(tmpdir).iterdir():
                        p.unlink()
                    Path(tmpdir).rmdir()
        finally:
            _cleanup_temp(temp_records_path)
    else:
        seed_offset = 0
        for spec, count in zip(specs, counts):
            if count <= 0:
                continue
            for start in range(0, count, generation_chunk_size):
                sub_count = min(generation_chunk_size, count - start)
                for seq in _generate_chunk_serial(
                    records, spec, sub_count, crop_size, base_seed + seed_offset
                ):
                    yield seq
                seed_offset += 1


def _worker_main(args: argparse.Namespace) -> None:
    """Entry point for stand-alone subprocess workers."""
    records = _load_records_from_temp(args.records_path)
    spec: dict[str, Any] = {
        "name": args.spec_name,
        "kwargs": json.loads(args.kwargs_json),
        "pre_shuffle": args.pre_shuffle,
    }
    if args.spec_name == "mix":
        spec["n_segments"] = args.n_segments
    else:
        spec["fn"] = globals()[args.fn_name]

    sequences = _generate_chunk_serial(
        records,
        spec,
        args.count,
        args.crop_size,
        args.seed,
    )
    _write_chunk(args.output_path, sequences)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic perturbed sequences for Jaeger reliability training."
    )
    parser.add_argument(
        "--worker", action="store_true", help="Run in stand-alone worker mode."
    )
    parser.add_argument("--records-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--spec-name", type=str)
    parser.add_argument("--fn-name", type=str, default="")
    parser.add_argument("--kwargs-json", type=str, default="{}")
    parser.add_argument("--count", type=int)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--n-segments", type=int, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--pre-shuffle", action="store_true", help="Shuffle sequence before perturbing."
    )
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
    else:
        parser.error("Only --worker mode is supported from the command line.")


if __name__ == "__main__":
    _main()
