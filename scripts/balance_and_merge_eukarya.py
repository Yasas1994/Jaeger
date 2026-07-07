#!/usr/bin/env python3
"""Replace some existing eukarya fragments with new big-euk fragments.

The overall eukarya class count stays equal to the original count, so the
eukarya fraction of the training set does not grow when a large eukaryotic
genome is added.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def count_label(csv_path: Path, label: int) -> int:
    """Count rows whose first column equals *label*."""
    n = 0
    with csv_path.open() as fh:
        for line in fh:
            first, _rest = line.split(",", 1)
            if int(first) == label:
                n += 1
    return n


def downsample_and_merge(
    existing_csv: Path,
    new_euk_csv: Path,
    output_csv: Path,
    euk_label: int = 2,
    seed: int = 42,
) -> dict[str, int]:
    """Keep all non-eukarya rows and probabilistically drop existing euk rows.

    The keep probability is chosen so that the final eukarya count equals the
    original eukarya count (new fragments replace dropped old fragments).
    """
    existing_euk = count_label(existing_csv, euk_label)
    new_euk = count_label(new_euk_csv, euk_label)

    if new_euk > existing_euk:
        raise ValueError(
            f"New eukarya fragments ({new_euk}) exceed existing ({existing_euk}). "
            "Use a smaller genome/larger stride, deduplicate more, or allow the fraction to grow."
        )

    keep_p = (existing_euk - new_euk) / existing_euk if existing_euk else 0.0
    rng = random.Random(seed)

    counters = {"kept_existing_euk": 0, "new_euk": new_euk, "non_euk": 0}
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with existing_csv.open() as fin, output_csv.open("w") as fout:
        for line in fin:
            first, _rest = line.split(",", 1)
            if int(first) == euk_label:
                if rng.random() < keep_p:
                    fout.write(line)
                    counters["kept_existing_euk"] += 1
            else:
                fout.write(line)
                counters["non_euk"] += 1

    with new_euk_csv.open() as fin, output_csv.open("a") as fout:
        fout.write(fin.read())

    return counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge new eukarya fragments while keeping the class count fixed."
    )
    parser.add_argument(
        "--existing", required=True, type=Path, help="Original training CSV"
    )
    parser.add_argument(
        "--new-euk", required=True, type=Path, help="New eukarya fragments CSV"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output balanced CSV"
    )
    parser.add_argument(
        "--euk-label", type=int, default=2, help="Integer label used for eukarya"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counters = downsample_and_merge(
        args.existing,
        args.new_euk,
        args.output,
        euk_label=args.euk_label,
        seed=args.seed,
    )
    print(
        f"kept_existing_euk={counters['kept_existing_euk']} "
        f"new_euk={counters['new_euk']} "
        f"non_euk={counters['non_euk']}"
    )


if __name__ == "__main__":
    main()
