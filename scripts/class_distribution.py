#!/usr/bin/env python3
"""Report the per-class fragment distribution of Jaeger npz datasets.

Handles both flat npz files (``labels`` array) and sharded npz files
(``labels_00000``, ``labels_00001``, ... produced by ``jaeger.dataops.convert``).
Prints per-class counts and percentages for each input file, pairwise count
ratios when multiple files are given (e.g. train vs val prior comparison),
and inverse-frequency class weights ready to paste into a training config as
``training.classifier_class_weights``.

Usage:
    python scripts/class_distribution.py train.npz [val.npz ...]
    python scripts/class_distribution.py train.npz --names bacteria phage eukarya archaea virus plasmid
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

DEFAULT_CLASS_NAMES = ["bacteria", "phage", "eukarya", "archaea", "virus", "plasmid"]


def load_labels(path: str) -> np.ndarray:
    """Load the label array from a flat or sharded Jaeger npz."""
    npz = np.load(path, mmap_mode="r")
    keys = npz.files
    if "labels" in keys:
        return np.asarray(npz["labels"]).astype(np.int64, copy=False)
    shards = sorted(k for k in keys if k.startswith("labels_"))
    if not shards:
        sys.exit(f"{path}: no 'labels' or 'labels_*' arrays found")
    return np.concatenate(
        [np.asarray(npz[k]) for k in shards]
    ).astype(np.int64, copy=False)


def report(path: str, names: list[str]) -> tuple[np.ndarray, int]:
    labels = load_labels(path)
    counts = np.bincount(labels, minlength=len(names))
    total = int(counts.sum())
    width = max(len(n) for n in names)
    print(f"\n{path}")
    print(f"  {'class':>{width}} {'count':>12} {'percent':>8}")
    for name, count in zip(names, counts):
        print(f"  {name:>{width}} {int(count):>12,} {100 * count / total:>7.2f}%")
    print(f"  {'TOTAL':>{width}} {total:>12,}")
    return counts, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="+", help="Jaeger npz dataset file(s)")
    parser.add_argument(
        "--names",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="Class names in label order (default: %(default)s)",
    )
    args = parser.parse_args()

    results = [report(p, args.names) for p in args.npz]

    if len(results) > 1:
        ref_counts, _ = results[0]
        width = max(len(n) for n in args.names)
        print(f"\ncount ratio vs {args.npz[0]}:")
        for p, (counts, _) in zip(args.npz[1:], results[1:]):
            ratios = [
                f"{args.names[i]}: {ref_counts[i] / max(counts[i], 1):.1f}x"
                for i in range(len(args.names))
            ]
            print(f"  {p}\n    " + ", ".join(ratios))
        _ = width  # keep ruff from flagging an unused variable in edits

    counts, total = results[0]
    weights = total / (len(args.names) * np.maximum(counts, 1))
    print(
        f"\ninverse-frequency class weights from {args.npz[0]} "
        "(paste into the training config):"
    )
    print("classifier_class_weights:")
    for i, w in enumerate(weights):
        print(f"  {i}: {w:.3f}")


if __name__ == "__main__":
    main()
