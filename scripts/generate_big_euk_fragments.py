#!/usr/bin/env python3
"""Generate fixed-length, non-overlapping fragments from a eukaryote FASTA."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pyfastx


def fragment_fasta(
    input_fasta: Path,
    output_csv: Path,
    label: int,
    frag_len: int,
    stride: int,
    min_len: int,
) -> int:
    """Write `label,sequence` CSV rows for every valid fragment."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    open_fn = gzip.open if str(input_fasta).endswith(".gz") else open
    n_written = 0
    with (
        open_fn(input_fasta, "rt") as fh_in,
        open(output_csv, "w", newline="") as fh_out,
    ):
        for _name, seq in pyfastx.Fasta(fh_in.name, build_index=False):
            seq = str(seq).upper()
            L = len(seq)
            if L < frag_len:
                continue
            for start in range(0, L - frag_len + 1, stride):
                frag = seq[start : start + frag_len]
                if len(frag) < min_len:
                    continue
                fh_out.write(f"{label},{frag}\n")
                n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2000 bp fragments from a eukaryote FASTA for Jaeger training."
    )
    parser.add_argument("-i", "--input-fasta", required=True, type=Path)
    parser.add_argument("-o", "--output-csv", required=True, type=Path)
    parser.add_argument("--label", type=int, default=2, help="Eukarya label")
    parser.add_argument("--frag-len", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=2000)
    parser.add_argument("--min-len", type=int, default=2000)
    args = parser.parse_args()

    n = fragment_fasta(
        args.input_fasta,
        args.output_csv,
        args.label,
        args.frag_len,
        args.stride,
        args.min_len,
    )
    print(f"Wrote {n} fragments to {args.output_csv}")


if __name__ == "__main__":
    main()
