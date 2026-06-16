# Training and Fine-Tuning Models

Jaeger provides a complete training pipeline for building custom phage detection models from scratch or fine-tuning existing ones on new data.

---

## Table of contents

- [Overview](#overview)
- [Training workflow](#training-workflow)
- [Preparing training data](#preparing-training-data)
- [Configuration file](#configuration-file)
- [Data format optimization](#data-format-optimization)
- [Running training](#running-training)
- [Fine-tuning](#fine-tuning)
- [Self-supervised pretraining](#self-supervised-pretraining)
- [Creating ensembles](#creating-ensembles)
- [Command reference](#command-reference)

---

## Overview

Jaeger's architecture consists of three components:

1. **Representation learner** — A 1D convolutional network with residual blocks that learns sequence embeddings from translated DNA (6-frame codon embeddings).
2. **Classification head** — Predicts the class of each window (bacteria, phage, eukarya, archaea, plasmid, virus).
3. **Reliability head** — An out-of-distribution detector that estimates prediction confidence.

All three can be trained jointly or independently.

---

## Training workflow

```
1. Prepare FASTA files per class
        ↓
2. Generate fragments (jaeger utils dataset / fragment)
        ↓
3. Convert to CSV format (jaeger utils convert)
        ↓
4. Create training config YAML
        ↓
5. Run training (jaeger train -c config.yaml)
        ↓
6. Save model and register (jaeger register-models)
```

---

## Preparing training data

### Step 1: Collect reference sequences

Organize your reference genomes into separate FASTA files by class:

```
data/
├── bacteria.fasta
├── phage.fasta
├── eukarya.fasta
├── archaea.fasta
├── plasmid.fasta
└── virus.fasta
```

### Step 2: Generate training fragments

Use `jaeger utils dataset` to create non-redundant fragment databases:

```bash
# Fragment bacteria genomes into 2048 bp pieces with 60% identity filtering
jaeger utils dataset \
  -i bacteria.fasta \
  -o bacteria_fragments.csv \
  --itype fasta \
  --outtype csv \
  --fraglen 2048 \
  --overlap 1024 \
  --maxiden 0.6 \
  --maxcov 0.6 \
  --class 0 \
  --seq_col 1 \
  --class_col 0
```

Parameters explained:

| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| `--fraglen` | Maximum fragment length | 2048 |
| `--overlap` | Overlap between adjacent fragments | 1024 |
| `--maxiden` | Maximum identity between any two fragments | 0.6 |
| `--maxcov` | Maximum coverage between any two fragments | 0.6 |
| `--class` | Numeric class label | 0=bacteria, 1=phage, etc. |
| `--valperc` | Fraction for validation set | 0.1 |
| `--trainperc` | Fraction for training set | 0.8 |
| `--testperc` | Fraction for test set | 0.1 |

### Step 3: Simulate metagenome fragments

For more realistic training data, simulate variable-length fragments:

```bash
jaeger utils fragment \
  -i phage.fasta \
  -o phage_fragments.fasta \
  --minlen 1000 \
  --maxlen 5000 \
  --overlap 0
```

Then convert to CSV:
```bash
jaeger utils convert \
  -i phage_fragments.fasta \
  -o phage_fragments.csv \
  --itype fasta
```

### Step 4: Generate OOD data for reliability training

The reliability head needs out-of-distribution examples. Use shuffled sequences:

```bash
jaeger utils ood-data \
  -i combined_train.csv \
  -o ood_train.csv \
  --itype csv \
  --otype csv \
  --dinuc
```

---

## Data format optimization

By default, Jaeger loads training data from CSV files and preprocesses sequences on-the-fly (live codon translation, n-gram extraction, etc.). This CPU-bound preprocessing can become a bottleneck, leaving the GPU underutilized.

Jaeger supports two training data formats via the `data_format` config option:

| Format | Speedup vs CSV | Best for | Notes |
|--------|----------------|----------|-------|
| `csv` | 1.0× (baseline) | Small datasets, quick experiments | Live preprocessing every epoch |
| `numpy` | **~9×** | Maximum throughput, direct loading | Preprocessed once, loaded from `.npz` |

**Hardware-dependent speeds:**
- Local RTX 3500 Ada: NumPy ~10K batches/sec, CSV ~130 batches/sec
- Zeus L40S (node030): NumPy ~2.9K batches/sec, CSV ~317 batches/sec

### When to optimize

Consider converting to NumPy when:
- GPU utilization is low (< 20%)
- Epoch times are dominated by data loading
- You have enough RAM to hold the preprocessed dataset

For example, a 3.1M sample dataset (~15 GB preprocessed) easily fits in most training servers' RAM and loads **30× faster** as NumPy.

### Converting CSV to the NumPy format

Use the `jaeger utils optimize-data` command. The `--format` option controls the type of preprocessing applied during conversion, not the training `data_format`, which is always `numpy` for `.npz` inputs:

```bash
# Translated representation (default for most Jaeger models)
jaeger utils optimize-data \
  -i train_shuffled.csv \
  -o train_shuffled_translated.npz \
  --format translated \
  --crop-size 500 \
  --num-classes 3

# Nucleotide/one-hot representation
jaeger utils optimize-data \
  -i train_shuffled.csv \
  -o train_shuffled_nucleotide.npz \
  --format nucleotide \
  --crop-size 500 \
  --num-classes 3

# Both translated and nucleotide representations in one file
jaeger utils optimize-data \
  -i train_shuffled.csv \
  -o train_shuffled_both.npz \
  --format both \
  --crop-size 500 \
  --num-classes 3
```

Convert both training and validation sets:

```bash
for split in train val; do
  jaeger utils optimize-data \
    -i ${split}_shuffled.csv \
    -o ${split}_shuffled.npz \
    --format translated \
    --crop-size 500 \
    --num-classes 3
done
```

### Configuring training to use the NumPy format

Add `data_format: numpy` to the `string_processor` section of your config:

```yaml
model:
  string_processor:
    data_format: numpy   # csv | numpy
    seq_onehot: false
    codon: CODON
    codon_id: CODON_ID
    crop_size: 500
    buffer_size: 500000
    shuffle: true
    # ... other fields
```

Then point `fragment_classifier_data` to the converted `.npz` files:

```yaml
fragment_classifier_data:
  train:
    - class: ["chromosome", "virus", "plasmid"]
      path:
        - "{{ training.data_dir }}/train_shuffled.npz"
      label: [0, 1, 2]
  validation:
    - class: ["chromosome", "virus", "plasmid"]
      path:
        - "{{ training.data_dir }}/val_shuffled.npz"
      label: [0, 1, 2]
```

**Notes on augmentation and preprocessing:**
- Whether runtime shuffling, mutation, or masking can be applied depends on how the NPZ was produced (e.g., integer sequences allow mutation/shuffle, one-hot tensors do not) and on the config's `seq_onehot` and `nucleotide_onehot_map` settings.
- For `numpy` inputs, live preprocessing is skipped; the converted tensors are loaded directly. If you need augmentations such as frame shuffling, configure them in the `string_processor` and ensure the NPZ stores the appropriate representation (e.g., integer indices rather than one-hot tensors).

---

## Configuration file

Training is controlled by a YAML configuration file. A template is provided at `train_config/nn_config.yaml`.

### Minimal example

```yaml
model:
  name: "my_jaeger_model"
  experiment: 1
  seed: 42
  classifier_out_dim: 6
  reliability_out_dim: 1
  base_dir: "/path/to/experiments"
  class_label_map:
    - class: "bacteria"
      label: 0
    - class: "phage"
      label: 1
    - class: "eukarya"
      label: 2
    - class: "archaea"
      label: 3
    - class: "plasmid"
      label: 4
    - class: "virus"
      label: 5

  embedding:
    use_embedding_layer: true
    input_type: "translated"
    strands: 2
    frames: 6
    length: null
    input_shape: [6, null]
    embedding_size: 192

training:
  data_dir: "/path/to/training/data"
  experiment_root: "exp_001"
  epochs: 100
  batch_size: 64
  learning_rate: 0.001

fragment_classifier_data:
  train:
    - class: ["bacteria", "phage", "eukarya", "archaea", "plasmid", "virus"]
      path:
        - "{{ training.data_dir }}/train_data.csv"
      label: [0, 1, 2, 3, 4, 5]
  validation:
    - class: ["bacteria", "phage", "eukarya", "archaea", "plasmid", "virus"]
      path:
        - "{{ training.data_dir }}/validation_data.csv"
      label: [0, 1, 2, 3, 4, 5]

fragment_reliability_data:
  train:
    - class: [indist, ood]
      path:
        - "{{ training.data_dir }}/train_data_ood.csv"
      label: [1, 0]
  validation:
    - class: [indist, ood]
      path:
        - "{{ training.data_dir }}/validation_data_ood.csv"
      label: [1, 0]
```

### Key configuration sections

| Section | Purpose |
|---------|---------|
| `model` | Architecture, embedding, class labels |
| `model.string_processor` | Preprocessing settings, **data format** (`csv`/`numpy`) |
| `representation_learner` | CNN layers, residual blocks, attention |
| `classifier` | Classification head architecture |
| `reliability` | Reliability (OOD) head architecture |
| `training` | Optimizer, batch size, epochs, callbacks |
| `fragment_classifier_data` | Paths to classification training data |
| `fragment_reliability_data` | Paths to reliability training data |

---

## Running training

### From scratch

```bash
jaeger train -c train_config/nn_config.yaml
```

### With mixed precision (faster on modern GPUs)

```bash
jaeger train -c train_config/nn_config.yaml --mixed_precision
```

### Resume from checkpoint

```bash
jaeger train -c train_config/nn_config.yaml --from_last_checkpoint
```

### Save model without training

If you already have checkpoints and just want to export a SavedModel:

```bash
jaeger train -c train_config/nn_config.yaml --only_save
```

---

## Fine-tuning

Fine-tuning allows you to adapt a pre-trained model to new data without training from scratch.

### Freeze the representation learner, train only heads

```bash
jaeger train -c fine_tune_config.yaml --only_heads
```

### Train only the classification head

```bash
jaeger train -c fine_tune_config.yaml --only_classification_head
```

### Train only the reliability head

```bash
jaeger train -c fine_tune_config.yaml --only_reliability_head
```

### Tips for fine-tuning

- Use a lower learning rate (e.g., 1e-4 vs 1e-3) to avoid catastrophic forgetting.
- Start from the pre-trained model's checkpoints by setting the correct paths in your config.
- Use `--from_last_checkpoint` to resume interrupted fine-tuning runs.

---

## Self-supervised pretraining

You can pretrain the representation learner with self-supervised learning before supervised classification:

```bash
jaeger train -c pretrain_config.yaml --self_supervised_pretraining
```

This is useful when you have large amounts of unlabeled sequence data.

---

## Creating ensembles

Combine multiple trained models into an ensemble for improved robustness:

```bash
jaeger utils combine-models \
  -i /path/to/model1 \
  -i /path/to/model2 \
  -i /path/to/model3 \
  -o /path/to/ensemble \
  -c mean
```

Aggregation methods:

| Method | Description |
|--------|-------------|
| `mv` | Majority voting |
| `sum` | Sum of logits |
| `mean` | Mean of logits (recommended) |
| `none` | No aggregation (returns all outputs) |

---

## Command reference

### `jaeger train`

```
Usage: jaeger train [OPTIONS]

Options:
  -c, --config PATH              Training config YAML  [required]
  --only_classification_head     Train only classification head
  --only_reliability_head        Train only reliability head
  --self_supervised_pretraining  Self-supervised pretraining
  --only_heads                   Train both heads, freeze representation
  --from_last_checkpoint         Resume from last checkpoint
  --force                        Delete existing checkpoints and restart
  --save_model                   Save model from last checkpoint
  --only_save                    Save model without training
  --mixed_precision              Use mixed-precision floats
  --meta PATH                    Write container metadata
  -v, --verbose                  Verbosity: -vv debug, -v info
  --help                         Show this message and exit.
```

### `jaeger register-models`

After training, register your model so `jaeger predict` can find it:

```bash
jaeger register-models --path /path/to/my_model
```

---

## Data augmentation tips

### Sequence masking

Gradually mask random positions to improve robustness:

```bash
jaeger utils mask \
  -i train.fasta \
  -o train_masked.fasta \
  --minperc 0.0 \
  --maxperc 0.3 \
  --step 0.05
```

### Sequence mutation

Introduce random mutations instead of masking:

```bash
jaeger utils mask \
  -i train.fasta \
  -o train_mutated.fasta \
  --mutate \
  --minperc 0.0 \
  --maxperc 0.1 \
  --step 0.01
```
