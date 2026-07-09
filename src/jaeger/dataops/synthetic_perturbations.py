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
    apply_kmer_shuffle,
    apply_mix,
    apply_shuffle,
    apply_subseq_repeat_window,
    apply_tandem_repeat_window,
)


def _normalize_perturbation_cfg(
    perturbations_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert flexible user config into a normalized list of perturbation specs."""
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
    """Return the number of synthetic samples to create for each perturbation spec."""
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
        for i in range(count):
            _, seq = records[i % n_records]
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
                        str, str, str, dict[str, Any], int, int | None, int | None, int
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
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
    else:
        parser.error("Only --worker mode is supported from the command line.")


if __name__ == "__main__":
    _main()
