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
- [Taxonomic assignment](#taxonomic-assignment)
- [Interpreting predictions](#interpreting-predictions-jaeger-interpret)
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
| `--stride` | Step size between windows | 1500 |
| `--min-len` | Minimum contig length to classify | `fsize` |
| `--dynamic-stride` | Evenly distribute windows on short contigs | False |
| `--no-dustmask` | Skip low-complexity masking | False |
| `--chunk-size` | Process input in chunks of N contigs | 0 (off) |
| `--taxonomy` | Assign taxonomy to viral contigs | False |
| `-f, --overwrite` | Overwrite existing output | False |

### Short contigs

Contigs shorter than `--fsize` are skipped by default. Pass `--min-len` to also classify contigs down to that length — they are padded (with `M` bases) up to the window size and classified as a single window:

```bash
jaeger predict -i contigs.fasta -o output_dir --min-len 1500
```

With `--dynamic-stride`, windows on short contigs are distributed evenly so the final window reaches the contig end.

---

## Choosing a model

The bundled `default` model, together with `experimental_1` and `experimental_2`, use the legacy prediction workflow and are **deprecated** — Jaeger prints a warning when you use them. They will be removed in a future release. New users should download a modern SavedModel from the model catalog.

```bash
# List all available models (newest first, with upload dates)
jaeger download --list

# Download the latest model
jaeger download --path ~/jaeger_models

# ...or download a specific model
jaeger download --model_name jaeger_91c5575c_1.3M_fragment --path ~/jaeger_models

# Use the downloaded model
jaeger predict -i contigs.fasta -o output_dir -m jaeger_91c5575c_1.3M_fragment
```

When `--model_name` is omitted, `jaeger download` automatically picks the newest model in the catalog.

`jaeger health` also warns you when no modern model is registered.

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

When `--window-scores` is passed, per-window logits and metadata are also written:

```
output_dir/
├── <input>_jaeger.tsv
├── <input>_phages_jaeger.tsv
├── <input>_window_scores.npz   # Per-window predictions + metadata
└── <input>_jaeger.log
```

The `.npz` file contains the arrays `headers`, `lengths`, `predictions`, `gc_skews`, and `gcs`. `predictions`, `gc_skews`, and `gcs` are stored as object arrays with one entry per contig.

When prophage extraction is enabled (`-p`):

```
output_dir/
├── <input>_jaeger.tsv
├── <input>_phages_jaeger.tsv
├── <input>_jaeger.log
└── <sample_name>_prophages/
    ├── prophages_jaeger.tsv     # Prophage coordinates
    └── plots/
        └── *.pdf                # Genome visualizations (circular and/or linear)
```

When taxonomy assignment is enabled (`--taxonomy`), an additional table is written:

```
├── <input>_taxonomy.tsv         # Per-contig taxonomy predictions + confidence
```

---

## Understanding the output table

The main TSV file (`<input>_jaeger.tsv`) contains one row per contig:

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `contig_id` | FASTA header | — |
| `length` | Sequence length | ≥ `min-len` |
| `prediction` | Final call | one of the model's classes (e.g. `phage`, `virus`, `bacteria`, `archaea`, `eukarya`, `plasmid`), or `uncertain` when the reliability score is below `--uc` (default 0.5) |
| `entropy` | Softmax entropy | 0 (confident) → high (uncertain) |
| `reliability_score` | Reliability of the input | 0 (out-of-distribution) → 1 (in-distribution); `unavailable` if the model has no reliability head |
| `host_contam` | Potential host contamination | `True` / `False` |
| `prophage_contam` | Potential prophage contamination | `True` / `False` |
| `G+C` | GC content | 0–1 |
| `N%` | Fraction of N bases | 0–1 |
| `#_<class>_windows` | Windows classified as `<class>` | one set per model class |
| `<class>_score` | Mean logit for `<class>` windows | — |
| `<class>_var` | Variance of `<class>` logits | — |
| `window_summary` | Pattern of per-window calls | e.g., `5V1n2V` |
| `terminal_repeats` | Detected terminal repeat type | `DTR`, `ITR`, or `null` |
| `repeat_length` | Length of terminal repeat | bp |

The exact per-class columns depend on the model's classes — modern six-class models report `phage`, `virus` (eukaryotic virus), `bacteria`, `archaea`, `eukarya`, and `plasmid`.

### Filtering tips

- High-confidence phages: `prediction == "phage"` and `reliability_score > 0.5`
- Eukaryotic viruses: `prediction == "virus"` and `reliability_score > 0.5`
- Out-of-distribution inputs: `prediction == "uncertain"` (reliability score below `--uc`, default 0.5); lower `--uc` to accept more marginal inputs
- Potential prophages: `prophage_contam == True`
- Models without a reliability head report `reliability_score` as `unavailable`; in this case calls are filtered using the class score and `--pc` only

---

## Taxonomic assignment

Modern models can additionally assign a taxonomy to each contig by searching the per-window embeddings against a reference database with FAISS. Pass `--taxonomy`:

```bash
jaeger predict -i contigs.fasta -o output_dir --taxonomy
```

The taxonomy database (`taxonomy_db/`) is detected automatically inside the model directory; if it is missing, Jaeger prints a warning and skips taxonomy prediction. Not supported with quantized or ONNX inference engines.

Results are written to `<input>_taxonomy.tsv` with one row per contig:

| Column | Description |
|--------|-------------|
| `header` | Contig id |
| `length` | Contig length |
| `taxid` / `name` / `rank` | Predicted taxon (majority-vote LCA of window-level neighbors) |
| `support_frac` | Fraction of votes supporting the call — a per-contig confidence signal (low values flag discordant, unreliable calls) |
| `lineage` | Full ranked lineage string |
| `prediction`, `entropy`, `reliability_score`, `<class>_score` | Classifier summary from the main prediction table |

Use `--taxonomy-k` to change the number of nearest neighbors voting on each window (default 5). For GPU-accelerated search, install `jaeger-bio[taxonomy-gpu]` (mutually exclusive with the `taxonomy` extra, which installs CPU FAISS).

---

## Interpreting predictions (`jaeger interpret`)

`jaeger interpret` computes gradient-based saliency maps showing which regions of a contig drive its classification. Unlike `predict` (which runs the frozen SavedModel graph), it rebuilds the Keras model from the recipe and weights bundled with the model:

```bash
jaeger interpret -i contigs.fasta -o saliency_out -m jaeger_91c5575c_1.3M_fragment
```

Key options:

| Option | Description | Default |
|--------|-------------|---------|
| `--flavour` | Attribution method: `raw`, `gradxinput`, or `integrated` | `raw` |
| `--class-index` | Class to compute saliency for (default: predicted class per window) | — |
| `--all-classes` | Compute saliency for every class | False |
| `--whole-sequence` | Treat each contig as a single window instead of sliding windows | False |
| `--ig-steps` | Interpolation steps for `integrated` | 50 |
| `--store-gradients` | Also store raw per-dimension gradients | False |

Results are written as an `.npz` file with per-position saliency for the forward and reverse strands, which can be visualized together with gene annotations (e.g. GFF3) in downstream notebooks.

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

The `plots/` directory contains genome visualizations with prophage regions highlighted. By default, circular plots are generated. Use `--plot-type` to select linear plots, both, or none.

### Plot types

| `--plot-type` | Description |
|---------------|-------------|
| `circular` | Circular genome map (default) |
| `linear` | Horizontal 4-panel linear plot |
| `both` | Generate both circular and linear plots |
| `none` | Skip plot generation (report only) |

```bash
# Generate linear plots instead of circular
jaeger predict -p -i genome.fna -o output_dir --plot-type linear

# Generate both circular and linear plots
jaeger predict -p -i genome.fna -o output_dir --plot-type both

# Skip plots, generate TSV report only
jaeger predict -p -i genome.fna -o output_dir --plot-type none
```

---

## Command-line reference

### Main commands

```
jaeger [COMMAND] [OPTIONS]
```

| Command | Purpose |
|---------|---------|
| `jaeger predict` | Run phage/virus detection on a FASTA file |
| `jaeger interpret` | Compute gradient-based saliency maps |
| `jaeger health` | Check installation and hardware |
| `jaeger download` | Download pre-trained models (latest by default) |
| `jaeger register-models` | Register custom model directories |
| `jaeger train` | Train new models from scratch |
| `jaeger utils` | Auxiliary tools (see [Utility commands](utils.md)) |
| `jaeger taxonomy` | Experimental taxonomy prediction |

### `jaeger predict` full help

````
Usage: jaeger predict [OPTIONS]

  Runs Jaeger on a dataset

Options:
  -i, --input PATH                Path to input file  [required]
  -o, --output TEXT               Path to output directory  [required]
  --fsize INTEGER                 Length of the sliding window  [default: 2000]
  --stride INTEGER                The gap between two the sliding windows
                                  [default: 1500]
  --dynamic-stride                For short contigs, distribute windows evenly
                                  so the final window reaches the contig end.
  --min-len INTEGER               Minimum contig length to classify (default:
                                  --fsize). Contigs between --min-len and
                                  --fsize are classified as a single whole-
                                  contig window.
  --dustmask / --no-dustmask      Mask low-complexity regions with
                                  pydustmasker before fragmenting.
                                  [default: dustmask]
  -m, --model TEXT                Select a deep-learning model to use.
                                  [default: default]
  --model_path TEXT               Give the path to a model. overrides --model
  --config PATH                   Path to Jaeger config file (e.g., when using
                                  Apptainer or Docker)
  -p, --prophage                  Extract and report prophage-like regions
  -s, --sensitivity FLOAT         Sensitivity of the prophage extraction
                                  algorithm (0-4)  [default: 1.5]
  --lc INTEGER                    Minimum contig length for prophage
                                  extraction  [default: 500000]
  --plot-type [circular|linear|both|none]
                                  Prophage plot type  [default: none]
  --rc FLOAT                      Minimum reliability score required to accept
                                  predictions  [default: 0.1]
  --uc FLOAT                      Reliability score below which contigs are
                                  labeled 'uncertain' instead of the argmax
                                  class  [default: 0.5]
  --taxonomy                      Predict contig taxonomy using the taxonomy
                                  database bundled with the model
  --taxonomy-k INTEGER RANGE      Number of nearest database neighbors voting
                                  on each window's taxonomy  [default: 5]
  --pc FLOAT                      Minimum phage probability required to accept
                                  predictions (0-1)  [default: 0.5]
  --batch INTEGER                 Parallel batch size, lower if GPU runs out
                                  of memory  [default: 96]
  --workers INTEGER               Number of threads to use  [default: 4]
  --chunk-size INTEGER            Process the input in chunks of N contigs to
                                  limit memory (0 = disabled)  [default: 0]
  --window-scores                 Writes window-wise prediction scores and
                                  per-window metadata to a .npz file
  --getsequences                  Writes the putative phage sequences to a
                                  .fasta file
  --cpu                           Ignore available GPUs and explicitly run on
                                  CPU
  --physicalid INTEGER            Set default GPU device ID for multi-GPU
                                  systems  [default: 0]
  --mem INTEGER                   GPU memory limit  [default: 4]
  --getalllabels                  Get predicted labels for Non-Viral contigs
  --save-embedding                Save per-window embedding vectors to
                                  <sample>_embedding.npz
  --save-nmd                      Save per-window NMD (novelty) vectors to
                                  <sample>_nmd.npz
  -v, --verbose                   Verbosity level: -vv debug, -v info
                                  [default: 1]
  -f, --overwrite                 Overwrite existing files
  --help                          Show this message and exit.
````

### Utility commands (`jaeger utils`)

See the [Utility commands](utils.md) page.

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
