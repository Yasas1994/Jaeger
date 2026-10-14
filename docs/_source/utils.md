# Utility commands

`jaeger utils` groups auxiliary commands for building training datasets, generating OOD data, converting models, and summarizing predictions.

```
jaeger utils [COMMAND] --help
```

| Subcommand | Purpose |
|------------|---------|
| `build-dataset` | Generate a non-redundant fragment database for training |
| `sample-fragments` | Simulate a metagenome assembly from whole genomes |
| `convert-sequences` | Convert training data between CSV and FASTA |
| `encode-npz` | Convert CSV training data to the optimized NPZ format |
| `inspect-npz` | Print a summary of an NPZ dataset |
| `generate-ood` | Generate out-of-distribution (OOD) sequences for reliability training |
| `combine-reliability` | Merge an ID NPZ and an OOD NPZ into one reliability dataset |
| `mask-sequences` | Gradually mask or mutate random positions in FASTA files |
| `combine-models` | Combine multiple model graphs into an ensemble |
| `convert-model` | Convert a SavedModel to TFLite / ONNX / TensorRT |
| `quantize` | Quantize a model for faster inference |
| `compute-receptive-field` | Display the receptive field of a model config |
| `summarize-predictions` | Calculate statistics from Jaeger output |

---

## Building a reliability dataset

The reliability head is trained on a mix of in-distribution (ID) sequences and synthetic out-of-distribution (OOD) sequences. The recommended workflow is:

### 1. Generate OOD sequences from your training CSV

```bash
jaeger utils generate-ood \
    -i train.csv \
    -o train_ood.csv \
    --crop-size 3331 \
    --multiplier 1.0 \
    --shuffle-proportion 0.75
```

`--shuffle-proportion 0.75` generates 75% of OOD sequences by shuffling (dinucleotide + random); the rest is split across the enabled perturbations (`--subseq-repeat`, `--tandem-repeat`, `--n-stretch`, `--mix`, `--gc-shift`, `--pad-truncate`, `--iid-random`). Use `--pre-shuffle` to shuffle a sequence before applying a non-shuffle perturbation, and `--pad-truncate` to mimic the short-contig padding the model sees at inference time.

### 2. Encode both datasets to NPZ

```bash
jaeger utils encode-npz -i train.csv     -o train_id.npz  --format translated --crop-size 3331
jaeger utils encode-npz -i train_ood.csv -o train_ood.npz --format translated --crop-size 3331
```

Use `--max-memory-mb` to stream large datasets shard-by-shard instead of building them in memory, and `--target-shards N` when the NPZ will later be combined shard-wise with another sharded NPZ.

### 3. Combine ID and OOD into the final reliability NPZ

```bash
jaeger utils combine-reliability \
    --id-npz train_id.npz \
    --ood-npz train_ood.npz \
    -o reliability_train.npz \
    --balance-ratio 1.0
```

`--balance-ratio 1.0` downsamples the ID side so the combined dataset has equal ID and OOD counts. Repeat with your validation CSVs to build `reliability_val.npz`, then point the training config's `reliability` section at both files.
