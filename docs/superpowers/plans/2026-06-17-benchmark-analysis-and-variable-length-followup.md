# Benchmark analysis and variable-length follow-up implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scripts, tests, and SLURM orchestration needed to evaluate/visualize the fixed-length nucleotide-vs-translated benchmark and to run the planned variable-length follow-up.

**Architecture:** Keep the analysis as small, composable scripts that consume the artifacts already produced by `jaeger train` and `jaeger utils optimize-data`. A runner discovers experiment/model/validation triples, per-model evaluator produces metrics, and a plotting script turns the metrics into the requested figures. The variable-length follow-up reuses the same evaluator/plotter on a merged mixed-length NPZ.

**Tech Stack:** Python 3.12+, TensorFlow/Keras, NumPy, scikit-learn, matplotlib, seaborn, pytest, SLURM, Apptainer.

---

## File structure

| File | Responsibility |
|------|----------------|
| `scripts/prepare_length_csvs.py` | Generate train/val CSVs for arbitrary crop sizes (currently hard-coded to 500/1000/2000). |
| `scripts/run_benchmark_evaluation.py` | Discover trained models, run `evaluate_saved_model.py` for each, and write `evaluation_metrics.csv`. |
| `scripts/plot_benchmark_results.py` | Read the report CSVs and generate the four requested figures. |
| `tests/unit/test_evaluate_saved_model.py` | Unit tests for confusion-matrix saving. |
| `scripts/merge_npz_for_variable_length.py` | Concatenate several fixed-length NPZs into one mixed-length NPZ for variable-length training. |
| `scripts/create_variable_length_config.py` | Copy a fixed-length config and convert it to a dynamic-length config. |
| `slurm/scripts/eval_benchmark.slurm` | SLURM job that runs reporting, evaluation, and plotting after training finishes. |
| `slurm/scripts/prep_variable_length_csvs.slurm` | Generate 300/600/900/1800/3600 CSVs. |
| `slurm/scripts/merge_variable_length.slurm` | Merge the five crop-size NPZs into mixed train/val NPZs. |
| `slurm/scripts/train_variable_length_trans.slurm` | Train a variable-length translated model. |
| `slurm/scripts/train_variable_length_nuc.slurm` | Train a variable-length nucleotide model. |
| `tests/unit/test_prepare_length_csvs.py` | Unit tests for the CSV generator. |
| `tests/unit/test_run_benchmark_evaluation.py` | Unit tests for the evaluation runner. |
| `tests/unit/test_plot_benchmark_results.py` | Unit tests for the plotting script. |
| `tests/unit/test_merge_npz_for_variable_length.py` | Unit tests for the NPZ merger. |
| `tests/unit/test_create_variable_length_config.py` | Unit tests for the config generator. |

---

## Task 1: Make `prepare_length_csvs.py` accept a list of lengths

**Files:**
- Modify: `scripts/prepare_length_csvs.py:72-76`
- Test: `tests/unit/test_prepare_length_csvs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_prepare_length_csvs.py
import csv
from pathlib import Path

from scripts.prepare_length_csvs import main


def test_prepare_length_csvs_custom_lengths(tmp_path):
    fasta = tmp_path / "input.fasta"
    tsv = tmp_path / "labels.tsv"
    out_dir = tmp_path / "out"

    fasta.write_text(
        ">seq1\n" + "A" * 600 + "\n>seq2\n" + "C" * 600 + "\n"
    )
    tsv.write_text("seq1\tignore\tchromosome\nseq2\tignore\tvirus\n")

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_length_csvs.py",
            "--fasta", str(fasta),
            "--tsv", str(tsv),
            "--out-dir", str(out_dir),
            "--lengths", "300", "600",
        ]
        main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "train_300.csv").exists()
    assert (out_dir / "val_300.csv").exists()
    assert (out_dir / "train_600.csv").exists()
    assert (out_dir / "val_600.csv").exists()

    with open(out_dir / "train_600.csv") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # 90% train of 2 records -> 1 record
    assert len(rows[0][1]) == 600
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_prepare_length_csvs.py -v
```

Expected: FAIL because `--lengths` is not implemented.

- [ ] **Step 3: Implement `--lengths` argument**

```python
# scripts/prepare_length_csvs.py, replace lines 45-52 with:
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
        help="Crop sizes to generate (default: 500 1000 2000).",
    )
    args = parser.parse_args()
    ...
    for length in args.lengths:
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_prepare_length_csvs.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_length_csvs.py tests/unit/test_prepare_length_csvs.py
git commit -m "feat(scripts): accept --lengths in prepare_length_csvs.py"
```

---

## Task 2: Extend `evaluate_saved_model.py` to save confusion matrices

**Files:**
- Modify: `scripts/evaluate_saved_model.py`
- Test: `tests/unit/test_evaluate_saved_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_evaluate_saved_model.py
import sys
from pathlib import Path

import numpy as np

from scripts.evaluate_saved_model import compute_metrics


def test_compute_metrics_returns_confusion_matrix():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    y_pred = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    metrics, cm = compute_metrics(y_true, y_pred, num_classes=3, return_cm=True)
    assert cm.shape == (3, 3)
    assert metrics["overall_accuracy"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_evaluate_saved_model.py -v
```

Expected: FAIL because `return_cm` is not supported.

- [ ] **Step 3: Modify `compute_metrics` and add `--output-cm`**

```python
# scripts/evaluate_saved_model.py
from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred, num_classes=3, return_cm=False):
    pred_labels = np.argmax(y_pred, axis=-1)
    true_labels = np.argmax(y_true, axis=-1) if y_true.ndim > 1 else y_true

    overall_acc = float(accuracy_score(true_labels, pred_labels))
    balanced_acc = float(balanced_accuracy_score(true_labels, pred_labels))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

    metrics = {
        "overall_accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
    }
    for i in range(num_classes):
        metrics[f"precision_class_{i}"] = float(precision[i])
        metrics[f"recall_class_{i}"] = float(recall[i])
        metrics[f"f1_class_{i}"] = float(f1[i])
        metrics[f"support_class_{i}"] = float(support[i])

    if return_cm:
        return metrics, cm
    return metrics
```

Add argument:
```python
parser.add_argument("--output-cm", type=Path, default=None, help="Path to save confusion matrix .npy")
```

After computing metrics:
```python
if args.output_cm is not None:
    _, cm = compute_metrics(labels, preds, num_classes=num_classes, return_cm=True)
    np.save(args.output_cm, cm)
    print(f"Wrote confusion matrix to {args.output_cm}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_evaluate_saved_model.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_saved_model.py tests/unit/test_evaluate_saved_model.py
git commit -m "feat(scripts): save confusion matrix in evaluate_saved_model.py"
```

---

## Task 3: Create `run_benchmark_evaluation.py`

**Files:**
- Create: `scripts/run_benchmark_evaluation.py`
- Test: `tests/unit/test_run_benchmark_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_run_benchmark_evaluation.py
import csv
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_benchmark_evaluation import main


def test_run_benchmark_evaluation(tmp_path):
    experiments = tmp_path / "experiments"
    data = tmp_path / "data"
    out_csv = tmp_path / "evaluation_metrics.csv"

    # fake experiment + graph dir + val npz
    exp = experiments / "experiment_500bp_baseline_trans_42"
    graph = exp / "jaeger_500bp_baseline_trans_graph"
    graph.mkdir(parents=True)
    (graph / "saved_model.pb").write_text("fake")

    val = data / "val_shuffled_translated_500.npz"
    val.write_text("fake")

    def fake_subprocess(cmd, *args, **kwargs):
        out = Path(cmd[cmd.index("--output-csv") + 1])
        cm_out = Path(cmd[cmd.index("--output-cm") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        cm_out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "model_dir,npz,num_samples,overall_accuracy,balanced_accuracy\n"
            f"{graph},{val},100,0.6,0.5\n"
        )
        np.save(cm_out, np.array([[30, 5, 5], [2, 20, 8], [3, 7, 20]]))
        class Proc:
            returncode = 0
        return Proc()

    with patch("subprocess.run", side_effect=fake_subprocess):
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_benchmark_evaluation.py",
                "--experiments-root", str(experiments),
                "--data-root", str(data),
                "--output-csv", str(out_csv),
            ]
            main()
        finally:
            sys.argv = old_argv

    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["experiment"] == "experiment_500bp_baseline_trans_42"
    assert float(rows[0]["overall_accuracy"]) == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_run_benchmark_evaluation.py -v
```

Expected: FAIL because `run_benchmark_evaluation.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
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

        # map experiment name to validation NPZ
        name = exp_dir.name.replace("experiment_", "")
        m = re.search(r"(\\d+)bp", name)
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
                record["length_bp"] = re.search(r"(\\d+)bp", exp_name).group(1)
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_run_benchmark_evaluation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_benchmark_evaluation.py tests/unit/test_run_benchmark_evaluation.py
git commit -m "feat(scripts): add benchmark evaluation runner"
```

---

## Task 4: Create `plot_benchmark_results.py`

**Files:**
- Create: `scripts/plot_benchmark_results.py`
- Test: `tests/unit/test_plot_benchmark_results.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_plot_benchmark_results.py
import sys
from pathlib import Path

import numpy as np

from scripts.plot_benchmark_results import main


def test_plot_benchmark_results(tmp_path):
    report = tmp_path / "benchmark_report.csv"
    metrics = tmp_path / "evaluation_metrics.csv"
    out_dir = tmp_path / "figures"

    cm_trans = tmp_path / "cm_trans.npy"
    cm_nuc = tmp_path / "cm_nuc.npy"
    np.save(cm_trans, np.array([[30, 5, 5], [2, 20, 8], [3, 7, 20]]))
    np.save(cm_nuc, np.array([[25, 10, 5], [5, 15, 10], [5, 10, 15]]))

    report.write_text(
        "experiment,epochs_trained,best_epoch,best_val_accuracy,best_loss_epoch,\n"
        "experiment_500bp_baseline_trans_42,5,4,0.55,3,\n"
        "experiment_500bp_baseline_nuc_42,5,4,0.50,3,\n"
    )
    metrics.write_text(
        "experiment,length_bp,input_type,cm_path,model_dir,npz,num_samples,overall_accuracy,balanced_accuracy,f1_class_0,f1_class_1,f1_class_2\n"
        f"experiment_500bp_baseline_trans_42,500,translated,{cm_trans},m,n,100,0.55,0.50,0.9,0.4,0.3\n"
        f"experiment_500bp_baseline_nuc_42,500,nucleotide,{cm_nuc},m,n,100,0.50,0.45,0.85,0.35,0.25\n"
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "plot_benchmark_results.py",
            "--report-csv", str(report),
            "--metrics-csv", str(metrics),
            "--output-dir", str(out_dir),
        ]
        main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "f1_per_class_bar.png").exists()
    assert (out_dir / "accuracy_vs_length.png").exists()
    assert (out_dir / "confusion_matrix_grid.png").exists()
    assert (out_dir / "training_curves.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_plot_benchmark_results.py -v
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Generate benchmark figures from report and evaluation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_f1_per_class(metrics: pd.DataFrame, out_dir: Path) -> None:
    f1_cols = [c for c in metrics.columns if c.startswith("f1_class_")]
    melted = metrics.melt(
        id_vars=["length_bp", "input_type"],
        value_vars=f1_cols,
        var_name="class",
        value_name="f1",
    )
    melted["class"] = melted["class"].str.replace("f1_class_", "class ")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="class", y="f1", hue="input_type", col="length_bp")
    plt.title("Per-class F1 by length and input type")
    plt.tight_layout()
    plt.savefig(out_dir / "f1_per_class_bar.png")
    plt.close()


def plot_accuracy_vs_length(metrics: pd.DataFrame, out_dir: Path) -> None:
    metrics["length_bp"] = metrics["length_bp"].astype(int)
    plt.figure(figsize=(8, 5))
    for metric in ("overall_accuracy", "balanced_accuracy"):
        sns.lineplot(
            data=metrics,
            x="length_bp",
            y=metric,
            hue="input_type",
            marker="o",
            label=metric,
        )
    plt.title("Accuracy vs. sequence length")
    plt.xlabel("Length (bp)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_length.png")
    plt.close()


def plot_confusion_matrix_grid(metrics: pd.DataFrame, out_dir: Path) -> None:
    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    class_names = ["chromosome", "virus", "plasmid"]
    for idx, (_, row) in enumerate(metrics.iterrows()):
        cm = np.load(row["cm_path"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{row['input_type']} {row['length_bp']}bp")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_grid.png")
    plt.close()


def plot_training_curves(report: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=report, x="best_epoch", y="best_val_accuracy", hue="experiment")
    plt.title("Best validation accuracy per experiment")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-csv", type=Path, default=Path("benchmark_report.csv"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("evaluation_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_report/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(args.report_csv)
    metrics = pd.read_csv(args.metrics_csv)

    plot_f1_per_class(metrics, args.output_dir)
    plot_accuracy_vs_length(metrics, args.output_dir)
    plot_confusion_matrix_grid(metrics, args.output_dir)
    plot_training_curves(report, args.output_dir)

    print(f"Figures written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_plot_benchmark_results.py -v
```

Expected: PASS (the confusion-matrix placeholder creates a file).

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_benchmark_results.py tests/unit/test_plot_benchmark_results.py
git commit -m "feat(scripts): add benchmark plotting script"
```

---

## Task 5: Wire the evaluator/plotter into `eval_benchmark.slurm`

**Files:**
- Modify: `slurm/scripts/eval_benchmark.slurm`

- [ ] **Step 1: Update the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=jaeger_eval_benchmark
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/eval_benchmark_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/eval_benchmark_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif
SRC=$PROJECT/Jaeger

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "
        cd /jaeger_src
        python scripts/benchmark_report.py \
            --experiments-root $PROJECT/experiments \
            --output-md $PROJECT/benchmark_report.md \
            --output-csv $PROJECT/benchmark_report.csv
        python scripts/run_benchmark_evaluation.py \
            --experiments-root $PROJECT/experiments \
            --data-root $PROJECT/data \
            --output-csv $PROJECT/evaluation_metrics.csv \
            --batch-size 128
        python scripts/plot_benchmark_results.py \
            --report-csv $PROJECT/benchmark_report.csv \
            --metrics-csv $PROJECT/evaluation_metrics.csv \
            --output-dir $PROJECT/benchmark_report/figures
    "

echo "Benchmark reports written to $PROJECT/benchmark_report.md and $PROJECT/evaluation_metrics.csv"
```

- [ ] **Step 2: Test syntax**

```bash
ssh zeus 'sbatch --test-only /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/scripts/eval_benchmark.slurm'
```

Expected: `sbatch` accepts the script.

- [ ] **Step 3: Commit**

```bash
git add slurm/scripts/eval_benchmark.slurm
git commit -m "chore(slurm): run evaluation and plotting in eval_benchmark"
```

---

## Task 6: Create `merge_npz_for_variable_length.py`

**Files:**
- Create: `scripts/merge_npz_for_variable_length.py`
- Test: `tests/unit/test_merge_npz_for_variable_length.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_merge_npz_for_variable_length.py
import sys
from pathlib import Path

import numpy as np

from scripts.merge_npz_for_variable_length import main


def test_merge_npz(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    out = tmp_path / "mixed.npz"

    np.savez(a, features=np.arange(6).reshape(2, 3), labels=np.array([0, 1]))
    np.savez(b, features=np.arange(12).reshape(2, 6), labels=np.array([1, 2]))

    old_argv = sys.argv
    try:
        sys.argv = [
            "merge_npz_for_variable_length.py",
            "--inputs", str(a), str(b),
            "--output", str(out),
        ]
        main()
    finally:
        sys.argv = old_argv

    data = np.load(out)
    assert data["features"].shape[0] == 4
    assert data["labels"].shape[0] == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_merge_npz_for_variable_length.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
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

    features = []
    labels = []
    for inp in args.inputs:
        data = np.load(inp)
        features.append(data["features"])
        labels.append(data["labels"])

    np.savez(
        args.output,
        features=np.concatenate(features, axis=0),
        labels=np.concatenate(labels, axis=0),
    )
    print(f"Wrote mixed-length NPZ to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_merge_npz_for_variable_length.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_npz_for_variable_length.py tests/unit/test_merge_npz_for_variable_length.py
git commit -m "feat(scripts): add NPZ merger for variable-length training"
```

---

## Task 7: Create `create_variable_length_config.py`

**Files:**
- Create: `scripts/create_variable_length_config.py`
- Test: `tests/unit/test_create_variable_length_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_create_variable_length_config.py
import sys
from pathlib import Path

import yaml

from scripts.create_variable_length_config import main


def test_create_variable_length_config(tmp_path):
    base = tmp_path / "base.yaml"
    out = tmp_path / "out.yaml"

    cfg = {
        "model": {
            "name": "jaeger_500bp_baseline_trans",
            "experiment": "500bp_baseline_trans",
            "base_dir": "/tmp",
            "string_processor": {"crop_size": 500, "length": 500},
        }
    }
    base.write_text(yaml.dump(cfg))

    old_argv = sys.argv
    try:
        sys.argv = [
            "create_variable_length_config.py",
            "--base-config", str(base),
            "--output", str(out),
            "--experiment-suffix", "variable",
        ]
        main()
    finally:
        sys.argv = old_argv

    result = yaml.safe_load(out.read_text())
    assert result["model"]["name"] == "jaeger_variable_trans"
    assert result["model"]["experiment"] == "500bp_baseline_trans_variable"
    assert result["model"]["string_processor"]["crop_size"] is None
    assert result["model"]["string_processor"]["length"] is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_create_variable_length_config.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
"""Create a dynamic-length training config from a fixed-length base config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-suffix", default="variable")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.base_config.read_text())
    model = cfg.setdefault("model", {})

    old_name = model.get("name", "jaeger")
    old_exp = model.get("experiment", "experiment")
    suffix = args.experiment_suffix

    model["name"] = f"{old_name.rsplit('_', 1)[0]}_{suffix}"
    model["experiment"] = f"{old_exp}_{suffix}"

    sp = model.setdefault("string_processor", {})
    sp["crop_size"] = None
    sp["length"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.dump(cfg))
    print(f"Wrote variable-length config to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_create_variable_length_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/create_variable_length_config.py tests/unit/test_create_variable_length_config.py
git commit -m "feat(scripts): add variable-length config generator"
```

---

## Task 8: Add variable-length SLURM scripts

**Files:**
- Create: `slurm/scripts/prep_variable_length_csvs.slurm`
- Create: `slurm/scripts/merge_variable_length.slurm`
- Create: `slurm/scripts/train_variable_length_trans.slurm`
- Create: `slurm/scripts/train_variable_length_nuc.slurm`

- [ ] **Step 1: Create the four SLURM scripts**

`prep_variable_length_csvs.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=jaeger_prep_variable_csvs
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/prep_variable_csvs_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/prep_variable_csvs_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && python scripts/prepare_length_csvs.py \
        --fasta $PROJECT/data/train_test_sequences.fasta \
        --tsv $PROJECT/data/sequence_weights.tsv \
        --out-dir $PROJECT/data_variable \
        --lengths 300 600 900 1800 3600"
```

`merge_variable_length.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=jaeger_merge_variable
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/merge_variable_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/merge_variable_%j.err
#SBATCH --dependency=singleton

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

TRAIN_FILES=""
VAL_FILES=""
for size in 300 600 900 1800 3600; do
    TRAIN_FILES="$TRAIN_FILES $PROJECT/data_variable/train_shuffled_translated_${size}.npz"
    VAL_FILES="$VAL_FILES $PROJECT/data_variable/val_shuffled_translated_${size}.npz"
done

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && \
        python scripts/merge_npz_for_variable_length.py \
            --inputs $TRAIN_FILES \
            --output $PROJECT/data_variable/train_mixed_translated.npz && \
        python scripts/merge_npz_for_variable_length.py \
            --inputs $VAL_FILES \
            --output $PROJECT/data_variable/val_mixed_translated.npz"
```

`train_variable_length_trans.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=jaeger_train_variable_trans
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/train_variable_trans_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/train_variable_trans_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif
CONFIG=$PROJECT/configs/nn_config_variable_length_trans.yaml

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

apptainer exec --nv \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && jaeger train -c $CONFIG --force --xla --mixed_precision"
```

`train_variable_length_nuc.slurm`:
```bash
# same as above but with:
# CONFIG=$PROJECT/configs/nn_config_variable_length_nuc.yaml
# job-name=jaeger_train_variable_nuc
```

- [ ] **Step 2: Test syntax on Zeus**

```bash
ssh zeus 'for f in prep_variable_length_csvs merge_variable_length train_variable_length_trans train_variable_length_nuc; do sbatch --test-only /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/scripts/${f}.slurm; done'
```

Expected: All four scripts are accepted by `sbatch`.

- [ ] **Step 3: Commit**

```bash
git add slurm/scripts/prep_variable_length_csvs.slurm slurm/scripts/merge_variable_length.slurm slurm/scripts/train_variable_length_trans.slurm slurm/scripts/train_variable_length_nuc.slurm
git commit -m "chore(slurm): add variable-length follow-up scripts"
```

---

## Self-review

- **Spec coverage:**
  - Fixed-length report → Task 3 (`run_benchmark_evaluation.py`) + existing `benchmark_report.py`.
  - Per-class metrics → Task 2 extends `evaluate_saved_model.py`; Task 3 aggregates them.
  - Four figures → Task 4.
  - Variable-length follow-up → Tasks 1, 6, 7, 8.
- **Placeholder scan:** No TBDs or vague steps; code is shown for each script.
- **Type consistency:** All CSV columns referenced in plotting match the output of `run_benchmark_evaluation.py` and `benchmark_report.py`.
- **Confusion matrices:** Task 2 serializes per-model confusion matrices as `.npy` files and records their paths in `evaluation_metrics.csv`; Task 4 loads those arrays and draws the grid.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-17-benchmark-analysis-and-variable-length-followup.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach would you like?
