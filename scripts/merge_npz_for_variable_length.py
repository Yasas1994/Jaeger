#!/usr/bin/env python3
"""Merge multiple fixed-length NPZs into one mixed-length NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for inp in args.inputs:
        data = np.load(inp)
        features.extend(list(data["features"]))
        labels.extend(list(data["labels"]))

    lengths = np.array([f.shape[0] for f in features], dtype=np.int32)
    max_len = int(lengths.max())

    # Pad variable-length samples to the maximum length so the output is a
    # dense array that can be loaded with allow_pickle=False.
    out_shape = (len(features), max_len) + features[0].shape[1:]
    padded = np.zeros(out_shape, dtype=features[0].dtype)
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f

    labels_arr = np.array(labels)

    np.savez(
        args.output,
        features=padded,
        labels=labels_arr,
        lengths=lengths,
    )
    print(f"Wrote mixed-length NPZ to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
