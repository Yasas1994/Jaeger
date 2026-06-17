#!/usr/bin/env python3
"""Discover trained Jaeger models and run per-model evaluation."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def discover_experiments(experiments_root: Path, data_root: Path):
    """Yield (experiment_name, graph_dir, val_npz) triples."""
    for exp_dir in experiments_root.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("experiment_"):
            continue
        graph_dirs = list(exp_dir.glob("*_graph"))
        if not graph_dirs:
            continue
        graph_dir = graph_dirs[0]

        name = exp_dir.name.replace("experiment_", "")
        m = re.search(r"(\d+)bp", name)
        length = m.group(1) if m else None
        input_type = "translated" if "_trans" in name else "nucleotide"
        npz_name = f"val_shuffled_{input_type}_{length}.npz"
        val_npz = data_root / npz_name
        if not val_npz.exists():
            val_npz = data_root / f"val_shuffled_{input_type}.npz"
        yield exp_dir.name, graph_dir, val_npz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation_metrics.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = []
    fieldnames = None
    for exp_name, graph_dir, val_npz in discover_experiments(
        args.experiments_root, args.data_root
    ):
        if not val_npz.exists():
            print(f"Warning: missing {val_npz} for {exp_name}", file=sys.stderr)
            continue

        tmp_csv = Path(f"/tmp/eval_{exp_name}.csv")
        tmp_cm = Path(f"/tmp/eval_{exp_name}_cm.npy")
        subprocess.run(
            [
                "python", "scripts/evaluate_saved_model.py",
                "--model-dir", str(graph_dir),
                "--npz", str(val_npz),
                "--batch-size", str(args.batch_size),
                "--output-csv", str(tmp_csv),
                "--output-cm", str(tmp_cm),
            ],
            check=True,
        )

        with open(tmp_csv, newline="") as f:
            reader = csv.DictReader(f)
            for record in reader:
                record["experiment"] = exp_name
                record["length_bp"] = re.search(r"(\d+)bp", exp_name).group(1)
                record["input_type"] = (
                    "translated" if "_trans" in exp_name else "nucleotide"
                )
                record["cm_path"] = str(tmp_cm)
                rows.append(record)
                if fieldnames is None:
                    fieldnames = reader.fieldnames

    if not rows:
        print("No experiments evaluated.", file=sys.stderr)
        return 1

    ordered = ["experiment", "length_bp", "input_type", "cm_path"] + fieldnames
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote aggregated metrics to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
