#!/usr/bin/env python3
"""Parse raw FASTA + labels and create train/val CSVs for arbitrary fragment lengths."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

LABEL_MAP = {"chromosome": 0, "virus": 1, "plasmid": 2}


def load_labels(tsv_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            seq_id = parts[0]
            label_str = parts[2]
            labels[seq_id] = LABEL_MAP[label_str]
    return labels


def parse_fasta(fasta_path: Path):
    """Yield (seq_id, sequence) tuples."""
    seq_id = None
    seq_parts: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(seq_parts)
                seq_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if seq_id is not None:
            yield seq_id, "".join(seq_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[500, 1000, 2000],
        help="Fragment lengths to generate (default: 500 1000 2000).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(Path(args.tsv))

    records = []
    for seq_id, seq in parse_fasta(Path(args.fasta)):
        if seq_id not in labels:
            continue
        records.append((labels[seq_id], seq_id, seq))

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_val = int(len(records) * args.val_frac)
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"Total records: {len(records)}")
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    for length in args.lengths:
        for split_name, split_records in (
            ("train", train_records),
            ("val", val_records),
        ):
            out_path = out_dir / f"{split_name}_{length}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                rows_written = 0
                for label, _seq_id, seq in split_records:
                    cropped = seq[:length]
                    if len(cropped) == length:
                        writer.writerow([label, cropped])
                        rows_written += 1
            print(f"Wrote {out_path}: {rows_written} rows")


if __name__ == "__main__":
    main()
