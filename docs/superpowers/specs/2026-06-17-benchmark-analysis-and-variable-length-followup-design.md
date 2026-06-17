# Benchmark analysis and variable-length follow-up

## Context

We are running a matched-capacity benchmark on Zeus comparing **nucleotide (2-strand one-hot)** and **translated (six-frame codon embedding)** Jaeger fragment classifiers at **500, 1000, and 2000 bp**. The pipeline currently generates the final NPZ files; six training jobs are queued to run on L40S GPUs with `--xla --mixed_precision`.

After the fixed-length benchmark completes, we want a compact but informative report and a **variable-length follow-up** experiment using the winning representation.

## Goals

1. Compare nucleotide vs. translated inputs fairly (same capacity, same head, same training hyperparameters).
2. Report **per-class precision/recall/F1** (virus, plasmid, chromosome) in addition to overall accuracy.
3. Produce four standard figures:
   - per-class F1 bar plot by length and input type
   - accuracy-vs-length line plot
   - confusion-matrix grid for all six models
   - training curves (val loss/accuracy)
4. If a clear winner emerges, run a follow-up training the **winning representation** on mixed crop sizes (300, 600, 900, 1800, 3600 bp) with per-batch padding, then compare its length scaling to the fixed-length models.

## Fixed-length benchmark analysis

### Inputs

| Asset | Location |
|-------|----------|
| Training logs | `experiments/experiment_*/checkpoints/classifier/training.log` |
| Saved models | `experiments/experiment_*/jaeger_*_graph` |
| Validation NPZs | `data/val_shuffled_*_{500,1000,2000}.npz` |

### Scripts

1. **`scripts/benchmark_report.py`** (exists)
   - Reads each experiment's `training.log`.
   - Computes epochs trained, best epoch, best val accuracy, best val loss, final metrics.
   - Writes `benchmark_report.md` and `benchmark_report.csv`.

2. **`scripts/evaluate_saved_model.py`** (exists)
   - Loads a SavedModel using Jaeger's custom object map.
   - Runs inference on the full validation NPZ.
   - Computes overall accuracy, balanced accuracy, per-class precision/recall/F1/support, and confusion matrix.
   - Writes a per-model CSV row; we concatenate the six rows into `evaluation_metrics.csv`.

3. **`scripts/plot_benchmark_results.py`** (new)
   - Reads `benchmark_report.csv` and `evaluation_metrics.csv`.
   - Generates four PNGs under `benchmark_report/figures/`:
     - `f1_per_class_bar.png`
     - `accuracy_vs_length.png`
     - `confusion_matrix_grid.png`
     - `training_curves.png`

### Outputs

- `benchmark_report.md`
- `benchmark_report.csv`
- `evaluation_metrics.csv`
- `benchmark_report/figures/*.png`

## Variable-length follow-up

### Scope

- Use the representation that wins the fixed-length benchmark.
- Generate crops of **300, 600, 900, 1800, and 3600 bp** from the same source FASTA and label file.
- Train **one model** on a mixed dataset containing all five crop sizes.
- Rely on the existing `padded_batch` logic so each batch is padded to its longest sequence.
- Evaluate the variable-length model separately on each crop-size validation NPZ and compare to the corresponding fixed-length model.

### Scripts

1. Extend `scripts/prepare_length_csvs.py` (or add a small wrapper) to accept a list of lengths.
2. Generate CSVs and NPZs for the five new crop sizes.
3. Add `configs/nn_config_variable_length.yaml` based on the winning config but with dynamic length (`crop_size: null` / `length: null`).
4. Train one variable-length model.
5. Run `evaluate_saved_model.py` for each crop-size validation NPZ.
6. Update `plot_benchmark_results.py` to add a "variable-length" series to the accuracy-vs-length plot.

### Outputs

- Updated `benchmark_report.md` with a "Variable-length follow-up" section.
- `accuracy_vs_length_variable.png` showing fixed-length vs. variable-length performance.

## Success criteria

- The fixed-length report clearly shows which representation performs better at each length.
- Per-class metrics reveal whether the advantage is uniform or class-specific.
- The variable-length follow-up determines whether a single mixed-length model matches or beats fixed-length specialists.

## Open questions

- None at design time.
