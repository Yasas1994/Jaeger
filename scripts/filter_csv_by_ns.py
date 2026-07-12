#!/usr/bin/env python3
"""Remove rows whose sequence contains long runs of ambiguous nucleotides (N).

Reads a Jaeger fragment CSV (``label,sequence[,metadata...]``) and writes the
rows whose sequence has no run of more than ``--max-run`` consecutive N/n
characters to the output CSV. Rows are copied verbatim; only the sequence
field (second column) is inspected. Prints per-class kept/removed counts.

Usage:
    python scripts/filter_csv_by_ns.py train_data_2000.csv -o train_data_2000_filtered.csv
    python scripts/filter_csv_by_ns.py train_data_2000.csv -o out.csv --max-run 5
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Input CSV (label,sequence[,metadata...])")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--max-run",
        type=int,
        default=5,
        help="Maximum allowed consecutive Ns; rows with longer runs are removed "
        "(default: %(default)s)",
    )
    args = parser.parse_args()

    pattern = re.compile(rb"[Nn]{%d,}" % (args.max_run + 1))

    kept: defaultdict[int, int] = defaultdict(int)
    removed: defaultdict[int, int] = defaultdict(int)
    unparsed = 0

    with open(args.input_csv, "rb") as fin, open(args.output, "wb") as fout:
        for line in fin:
            parts = line.split(b",", 2)
            if len(parts) < 2:
                unparsed += 1
                fout.write(line)
                continue
            try:
                label = int(parts[0])
            except ValueError:
                # Header or malformed row: pass through verbatim.
                unparsed += 1
                fout.write(line)
                continue
            if pattern.search(parts[1]):
                removed[label] += 1
            else:
                kept[label] += 1
                fout.write(line)

    labels = sorted(set(kept) | set(removed))
    width = max(5, *(len(str(k)) for k in labels))
    print(f"input:  {args.input_csv}")
    print(f"output: {args.output}")
    print(f"removed rows with > {args.max_run} consecutive Ns\n")
    print(f"{'class':>{width}} {'kept':>12} {'removed':>10} {'removed %':>10}")
    for label in labels:
        total = kept[label] + removed[label]
        frac = 100 * removed[label] / total if total else 0.0
        print(f"{label:>{width}} {kept[label]:>12,} {removed[label]:>10,} {frac:>9.3f}%")
    total_kept = sum(kept.values())
    total_removed = sum(removed.values())
    total = total_kept + total_removed
    print(
        f"{'TOTAL':>{width}} {total_kept:>12,} {total_removed:>10,} "
        f"{100 * total_removed / total if total else 0.0:>9.3f}%"
    )
    if unparsed:
        print(f"\npassed through {unparsed} unparsed row(s) verbatim", file=sys.stderr)


if __name__ == "__main__":
    main()
