# Running Jaeger

This guide covers the main prediction workflow, output interpretation, prophage extraction, and available utility commands.

For inference speedups and model quantization, see the [Performance optimizations](optimizations.md) page.

---

## Table of contents

- [Basic prediction](#basic-prediction)
- [Choosing a model](#choosing-a-model)
- [Batch size and memory](#batch-size-and-memory)
- [Output files](#output-files)
- [Understanding the output table](#understanding-the-output-table)
- [Prophage extraction](#prophage-extraction)
- [Command-line reference](#command-line-reference)
- [Python integration](#python-integration)

---

## Basic prediction

The simplest way to run Jaeger:

```bash
jaeger predict -i contigs.fasta -o output_dir
```

Run on CPU explicitly:
```bash
jaeger predict -i contigs.fasta -o output_dir --cpu
```

Run with Apptainer/Singularity:
```bash
apptainer run --nv jaeger.sif jaeger predict -i contigs.fasta -o output_dir
```

### Common options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input FASTA file (required) | — |
| `-o, --output` | Output directory (required) | — |
| `--batch` | Parallel batch size | 96 |
| `--cpu` | Force CPU execution | False |
| `--workers` | Number of CPU threads | 4 |
| `--fsize` | Sliding window length | 2000 |
| `--stride` | Step size between windows | 2000 |
| `-f, --overwrite` | Overwrite existing output | False |

---

## Choosing a model

Jaeger ships with a `default` model. Additional models can be downloaded and selected via the `-m` flag.

```bash
# List all available models
jaeger download --list

# Download a specific model
jaeger download --model_name jaeger_57341_1.5M_fragment --path ~/jaeger_models

# Use the downloaded model
jaeger predict -i contigs.fasta -o output_dir -m jaeger_57341_1.5M_fragment
```

To register a custom model path:
```bash
jaeger register-models --path ~/my_custom_models
```

---

## Batch size and memory

The `--batch` option controls how many sequences are processed in parallel. If you encounter out-of-memory (OOM) errors, reduce this value.

| GPU memory | Suggested `--batch` |
|------------|---------------------|
| 4 GB | 32–64 |
| 8 GB | 64–96 |
| 16 GB | 96–128 |
| 24+ GB | 128–256 |

You can also limit GPU memory allocation:
```bash
jaeger predict -i contigs.fasta -o output_dir --mem 8
```

---

## Output files

After running Jaeger, the output directory contains:

```
output_dir/
├── <input>_jaeger.tsv           # Main predictions table
├── <input>_phages_jaeger.tsv    # Subset of phage-only predictions
└── <input>_jaeger.log           # Runtime log
```

When prophage extraction is enabled (`-p`):

```
output_dir/
├── <input>_jaeger.tsv
├── <input>_phages_jaeger.tsv
├── <input>_jaeger.log
└── <sample_name>_prophages/
    ├── prophages_jaeger.tsv     # Prophage coordinates
    └── plots/
        └── *.pdf                # Circular genome visualizations
```

---

## Understanding the output table

The main TSV file (`<input>_jaeger.tsv`) contains one row per contig:

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `contig_id` | FASTA header | — |
| `length` | Sequence length | ≥ `fsize` |
| `prediction` | Final call | `phage` or `bacteria` |
| `entropy` | Softmax entropy | 0 (confident) → 2 (uncertain) |
| `reliability_score` | Model confidence | 0 (uncertain) → 1 (confident) |
| `host_contam` | Potential host contamination | `True` / `False` |
| `prophage_contam` | Potential prophage contamination | `True` / `False` |
| `G+C` | GC content | 0–1 |
| `N%` | Fraction of N bases | 0–1 |
| `prediction_2` | Secondary prediction | e.g., `bacteria` |
| `#_bacteria_windows` | Windows classified as bacteria | — |
| `bacteria_score` | Mean logit for bacteria windows | — |
| `bacteria_var` | Variance of bacteria logits | — |
| `#_phage_windows` | Windows classified as phage | — |
| `phage_score` | Mean logit for phage windows | — |
| `phage_var` | Variance of phage logits | — |
| `#_eukarya_windows` | Windows classified as eukarya | — |
| `eukarya_score` | Mean logit for eukarya windows | — |
| `eukarya_var` | Variance of eukarya logits | — |
| `#_archaea_windows` | Windows classified as archaea | — |
| `archaea_score` | Mean logit for archaea windows | — |
| `archaea_var` | Variance of archaea logits | — |
| `window_summary` | Pattern of V (phage) / n (non-phage) windows | e.g., `5V1n2V` |
| `terminal_repeats` | Detected terminal repeat type | `DTR`, `ITR`, or `null` |
| `repeat_length` | Length of terminal repeat | bp |

### Filtering tips

- High-confidence phages: `prediction == "phage"` and `reliability_score > 0.5`
- Uncertain calls: `entropy > 1.0` or `reliability_score < 0.3`
- Potential prophages: `prophage_contam == True`

---

## Prophage extraction

Enable prophage detection with the `-p` flag:

```bash
jaeger predict -p -i genome.fna -o output_dir
```

Tune sensitivity with `-s` (0–4, higher = more sensitive):
```bash
jaeger predict -p -i genome.fna -o output_dir -s 2.0
```

Set minimum contig length for prophage scanning:
```bash
jaeger predict -p -i genome.fna -o output_dir --lc 100000
```

The `plots/` directory contains circular genome visualizations with prophage regions highlighted.

---

## Command-line reference

### Main commands

```
jaeger [COMMAND] [OPTIONS]
```

| Command | Purpose |
|---------|---------|
| `jaeger predict` | Run phage detection on a FASTA file |
| `jaeger health` | Check installation and hardware |
| `jaeger download` | Download pre-trained models |
| `jaeger register-models` | Register custom model directories |
| `jaeger train` | Train new models from scratch |
| `jaeger utils` | Auxiliary tools (see below) |
| `jaeger taxonomy` | Experimental taxonomy prediction |

### `jaeger predict` full help

````
Usage: jaeger predict [OPTIONS]

  Runs Jaeger on a dataset

Options:
  -i, --input PATH         Path to input file  [required]
  -o, --output TEXT        Path to output directory  [required]
  --fsize INTEGER          Length of the sliding window  [default: 2000]
  --stride INTEGER         The gap between two the sliding windows
                           (stride==fsize)  [default: 2000]
  -m, --model TEXT         Select a deep-learning model to use.  [default:
                           default]
  --model_path TEXT        Give the path to a model. overrides --model
  --config PATH            Path to Jaeger config file (e.g., when using
                           Apptainer or Docker)
  -p, --prophage           Extract and report prophage-like regions
  -s, --sensitivity FLOAT  Sensitivity of the prophage extraction algorithm
                           (0-4)  [default: 1.5]
  --lc INTEGER             Minimum contig length for prophage extraction
                           [default: 500000]
  --rc FLOAT               Minimum reliability score required to accept
                           predictions  [default: 0.1]
  --pc INTEGER             Minimum phage score required to accept predictions
                           [default: 3]
  --batch INTEGER          Parallel batch size, lower if GPU runs out of
                           memory  [default: 96]
  --workers INTEGER        Number of threads to use  [default: 4]
  --getalllogits           Writes window-wise scores to a .npy file
  --getsequences           Writes the putative phage sequences to a .fasta
                           file
  --cpu                    Ignore available GPUs and explicitly run on CPU
  --physicalid INTEGER     Set default GPU device ID for multi-GPU systems
                           [default: 0]
  --mem INTEGER            GPU memory limit  [default: 4]
  --getalllabels           Get predicted labels for Non-Viral contigs
  -v, --verbose            Verbosity level: -vv debug, -v info  [default: 1]
  -f, --overwrite          Overwrite existing files
  --help                   Show this message and exit.
````

### Utility commands (`jaeger utils`)

| Subcommand | Purpose |
|------------|---------|
| `jaeger utils dataset` | Generate non-redundant fragment databases for training |
| `jaeger utils fragment` | Simulate metagenome assemblies from genomes |
| `jaeger utils convert` | Convert between CSV and FASTA formats |
| `jaeger utils mask` | Mask or mutate positions in FASTA files |
| `jaeger utils ood-data` | Generate out-of-distribution (shuffled) sequences |
| `jaeger utils combine-models` | Combine multiple models into an ensemble |
| `jaeger utils stats` | Calculate statistics from Jaeger output |

---

## Python integration

> **Note:** The Python API (`jaeger.api`) is currently experimental and not available in the latest release. For programmatic access, use the CLI via `subprocess`:
>
> ```python
> import subprocess
> subprocess.run([
>     "jaeger", "predict",
>     "-i", "input.fasta",
>     "-o", "output_dir",
>     "--batch", "128"
> ])
> ```
