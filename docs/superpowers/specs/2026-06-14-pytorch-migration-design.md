# PyTorch Migration Design

## Context

Jaeger's training stack currently uses TensorFlow/Keras. Empirical testing shows TensorFlow is too slow for architectures that use attention mechanisms and energy-based training regimes. This design migrates the training pipeline — and the inference path that consumes trained models — to PyTorch in a single release.

## Goals

- Replace TensorFlow training with a plain PyTorch training stack.
- Migrate `jaeger predict` to support PyTorch-trained models natively.
- Maintain the existing YAML config format so current `train_config/` files remain valid.
- Validate parity with the TensorFlow baseline before merging.
- Run on the Zeus cluster using the existing Singularity + SLURM workflow.

## Non-goals

- Loading existing TensorFlow/Keras checkpoints into PyTorch. This is a clean break; models are trained from scratch.
- Supporting TensorFlow and PyTorch training backends simultaneously. The `pytorch_migration` branch becomes PyTorch-only for training.
- Loading old TensorFlow SavedModels in the PyTorch release. The release is a clean break for inference as well; users who need legacy models can stay on the previous Jaeger version.

## Clarifying decisions

| Question | Decision |
|---|---|
| Load existing TF checkpoints? | No — train from scratch. |
| PyTorch framework | Plain PyTorch (not Lightning or HF). |
| Data loading | Native `torch.utils.data.Dataset` + `DataLoader`. |
| Inference | Migrate to PyTorch in the same release. |
| Next version | `1.27.1` after merge. |

## Section 1 — High-level architecture & module layout

The migration creates a new, self-contained PyTorch stack under `src/jaeger/` while leaving the existing TensorFlow code untouched on the `pytorch_migration` branch. The TF code is deleted only at the very end of the branch, once PyTorch training and inference are validated.

```
src/jaeger/
├── nnlib/
│   └── pytorch/                 # NEW: PyTorch model library
│       ├── __init__.py
│       ├── layers.py            # PyTorch equivalents of custom Keras layers
│       ├── models.py            # RepModel, ClassifierHead, ReliabilityHead, JaegerModel
│       ├── losses.py            # ArcFace, hierarchical loss, classification losses
│       ├── metrics.py           # Per-class precision/recall/specificity
│       ├── builder.py           # Build nn.Modules from YAML config
│       └── checkpoints.py       # Save/load .pt checkpoints + metadata
├── data/
│   └── pytorch/                 # NEW: PyTorch datasets & loaders
│       ├── __init__.py
│       ├── dataset_csv.py       # CSV Dataset with padding & augmentations
│       ├── dataset_numpy.py     # numpy_full / numpy_raw / numpy_raw_variable
│       ├── dataset_tfrecord.py  # TFRecord Dataset (deferred)
│       ├── transforms.py        # codon translation, mutation, frame shuffle
│       └── collate.py           # padding collate functions
├── training/
│   └── pytorch/                 # NEW: training loops
│       ├── __init__.py
│       ├── trainer.py           # Classification / reliability / pretrain loops
│       ├── engine.py            # epoch loop, validation, mixed precision
│       ├── distributed.py       # DDP setup/teardown
│       └── callbacks.py         # ModelCheckpoint, EarlyStopping, LR scheduling
├── inference/
│   └── pytorch/                 # NEW: PyTorch inference backend
│       ├── __init__.py
│       ├── model.py             # Load PyTorch checkpoint + config
│       └── engine.py            # Batch inference, windowing, aggregation
├── commands/
│   ├── train.py                 # MODIFIED: dispatch to PyTorch trainer
│   └── predict.py               # MODIFIED: add PyTorch backend option
```

Key principles:
- The YAML config format stays the same so existing `train_config/` files remain valid.
- `DynamicModelBuilder` is replaced by `jaeger.nnlib.pytorch.builder.ModelBuilder`, but it reads the same config.
- Training entry point `jaeger train` detects the backend from config or CLI flag; on the `pytorch_migration` branch it defaults to PyTorch.
- Existing TF training and inference files (`nnlib/v1/`, `nnlib/v2/`, current `commands/train.py`, current `commands/predict.py`, etc.) remain during development for reference and are removed before merge. The PyTorch release does not load legacy TF models.

## Section 2 — Model architecture

The current Keras builder assembles four pieces:

1. **Representation learner** — embedding + stack of residual/attention blocks → `(embedding, nmd, [gate])`
2. **Classification head** — dense layers → class logits
3. **Reliability head** — dense layers → confidence score
4. **Projection head** (optional) — for ArcFace self-supervised pretraining

In PyTorch, each becomes an `nn.Module`:

- `Embedding` — maps codon indices or one-hot nucleotides to `(B, 6, L, D)` or `(B, 6, L, 3, D)` depending on config.
- `ResidualBlock`, `AxialAttention`, `CrossFrameAttention`, `TransformerEncoder` — replicate the Keras custom layers in `nnlib/pytorch/layers.py`.
- `RepresentationModel` — composes embedding + blocks + pooling → returns `(embedding, nmd, gate)` tuple.
- `ClassificationHead` / `ReliabilityHead` / `ProjectionHead` — small MLPs.
- `JaegerModel` — combines all heads and returns the same output dict as the Keras model: `{"prediction": ..., "embedding": ..., "nmd": ..., "gate": ..., "reliability": ...}`.

Key design choices:
- **Masking**: Keras uses `Masking` layers and `supports_masking`. In PyTorch we pass a `mask` tensor explicitly and use it in `MaskedBatchNorm`, `MaskedConv1D`, pooling, and attention.
- **Variable length**: PyTorch handles padded batches naturally; we use a collate function to pad sequences and pass the mask.
- **Mixed precision**: Use `torch.cuda.amp.autocast` + `GradScaler` (or `torch.amp` if using newer PyTorch).
- **Weight init**: Keep the same initializers where possible (orthogonal for embeddings/dense, glorot for conv) to make parity tests fair.

The builder reads the same YAML config and constructs the module graph, preserving hyperparameters (filters, kernel sizes, attention heads, dropout, regularization).

## Section 3 — Data pipeline

Current `tf.data` supports five formats: `csv`, `tfrecord`, `numpy_raw`, `numpy_raw_variable`, `numpy_full`. The PyTorch pipeline replaces this with `torch.utils.data.Dataset` + `DataLoader`.

Priority order:

1. **Phase B1 — `numpy_full`** (highest priority)
   - Fastest loading, no runtime augmentations.
   - `Dataset` loads the `.npz` once and returns preprocessed tensors.
   - `DataLoader` handles shuffling and batching; no padding needed because all sequences are already cropped.

2. **Phase B2 — `numpy_raw`**
   - Returns int8 sequences + labels.
   - Apply codon translation, n-gram extraction, frame shuffle, mutation in a `transform` callable.
   - Collate function pads to the batch max length and produces the mask.

3. **Phase B3 — `csv`**
   - Parse CSV lines, apply the same transforms as `numpy_raw`.
   - Slower, kept for backward compatibility and small experiments.

4. **Phase B4 — `tfrecord` / `numpy_raw_variable`** (deferred if time-constrained)
   - TFRecord is lower priority now that NumPy formats exist.
   - Variable-length can be handled with a custom collate similar to `numpy_raw`.

Augmentations (from `process_string_train`) move to `src/jaeger/data/pytorch/transforms.py`:
- codon translation
- frame shuffling
- random mutation
- masking
- n-gram encoding

Each `Dataset` returns a tuple `(input_tensor, label_tensor, mask_tensor)`. The collate function stacks inputs and labels and combines masks.

For multi-GPU, use `DistributedSampler` with `DataLoader`.

## Section 4 — Training loop

The current training flow in `commands/train.py` is:

1. Build models inside a distribution strategy scope.
2. Compile classifier, reliability, and optionally pretrain branches.
3. Call `.fit()` on each branch sequentially.
4. Save weights + SavedModel graphs.

In PyTorch, this becomes an explicit training engine:

- `Trainer` class in `src/jaeger/training/pytorch/trainer.py` owns:
  - model
  - optimizer(s)
  - loss function(s)
  - metric trackers
  - device / DDP wrapper
  - AMP gradient scaler

- Three training modes, matching the Keras flow:
  1. **Self-supervised pretraining** — train `rep_model` + `projection_head` with ArcFace loss.
  2. **Classifier training** — train `rep_model` + `classification_head`.
  3. **Reliability training** — freeze `rep_model`, train `reliability_head`.

- `Engine` in `src/jaeger/training/pytorch/engine.py` implements one epoch:
  - training step with `autocast`, loss backward, gradient clipping, optimizer step
  - validation step
  - metric aggregation
  - logging

- `callbacks.py` implements:
  - `ModelCheckpoint` — save best/last `.pt` checkpoints
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `TensorBoardLogger` / CSV logger

- Multi-GPU on Zeus:
  - `torchrun --nproc_per_node=2 ... jaeger train ...`
  - `DistributedDataParallel` wrapper
  - `DistributedSampler` for each DataLoader

- Resume logic:
  - Checkpoint file contains model state_dict, optimizer state_dict, epoch, and config.
  - `--from_last_checkpoint` loads the latest checkpoint and continues.

- CLI flags preserved:
  - `--mixed_precision`, `--xla` (replaced with AMP), `--from_last_checkpoint`, `--force`, `--only_classification_head`, `--only_reliability_head`, `--only_heads`, `--save_model`, `--only_save`, `--self_supervised_pretraining`.

## Section 5 — Inference migration

Current `jaeger predict` supports multiple backends (TF SavedModel, ONNX, TFLite, TensorRT). For the PyTorch release, we add a first-class PyTorch backend.

- Save format:
  - `model.pt` — model state_dict
  - `model_config.yaml` — architecture config (same as training config)
  - `class_label_map.yaml` — class mapping
  - Optional: `model.pt` can be a TorchScript module if tracing works for the variable-length inputs.

- `src/jaeger/inference/pytorch/model.py`:
  - Load config, instantiate `JaegerModel`, load state_dict.
  - Move to device (CUDA if available).
  - Set to eval mode.

- `src/jaeger/inference/pytorch/engine.py`:
  - Replicate the windowing + batching logic from the current `predict.py`.
  - Run forward pass, collect `prediction`, `embedding`, `reliability`.
  - Support `--precision fp16` via `autocast`.
  - Support `--batch` and `--workers`.

- CLI integration:
  - `jaeger predict` expects PyTorch models (look for `model.pt` + `model_config.yaml`).
  - Legacy TensorFlow SavedModels are not loaded by this release.

- Quantization/optimization:
  - Not in the first PyTorch release unless explicitly required.
  - Can be added later via `torch.compile` or ONNX export from PyTorch.

## Section 6 — Checkpointing, AMP, distributed training, and dependencies

- **Checkpoints**: PyTorch checkpoints are dictionaries containing:
  - `model_state_dict`
  - `optimizer_state_dict`
  - `epoch`
  - `config`
  - `metrics`
  - `branch` (classifier / reliability / projection)
  - File naming: `epoch:{epoch}-loss:{val_loss:.4f}.pt`

- **Mixed precision**: Use `torch.amp.autocast(device_type="cuda")` with `GradScaler`. The Keras `mixed_float16` policy maps directly.

- **Distributed training**:
  - Launch via `torchrun` / `torch.distributed.launch`.
  - `DistributedDataParallel` wraps the model.
  - `DistributedSampler` for each DataLoader.
  - Only the rank-0 process saves checkpoints and logs.

- **Dependencies**:
  - Replace `tensorflow[and-cuda]` with `torch` and `torchvision`.
  - Keep `numpy`, `pyyaml`, `click`, `rich`, `scipy`, `pandas`, `polars`, `pyfastx`.
  - Add `tensorboard` if using TensorBoard logging.
  - Remove TF-only dependencies from the PyTorch branch's training path.

- **Zeus cluster**:
  - Update SLURM scripts to use `torchrun --nproc_per_node=2`.
  - Ensure Singularity container has PyTorch + CUDA installed.
  - Bind `Jaeger/` source directory at runtime as per cluster rules.

## Section 7 — Testing & validation strategy

To make sure the PyTorch models match or exceed TF quality:

1. **Unit tests** for each new PyTorch module:
   - `nnlib/pytorch/layers.py` — forward pass shape tests, mask behavior.
   - `nnlib/pytorch/builder.py` — build models from existing configs.
   - `data/pytorch/dataset_*.py` — dataset length, batch shapes, augmentation correctness.
   - `training/pytorch/engine.py` — one training step, checkpoint save/load.

2. **Parity tests**:
   - Build the same architecture in TF and PyTorch with fixed random seeds.
   - Compare forward-pass outputs on identical synthetic input to within tolerance.
   - Compare a few training steps on a tiny dataset to verify gradients and weight updates are similar.

3. **Smoke tests**:
   - `jaeger train -c train_config/nn_config_500bp_baseline.yaml` runs for a few epochs.
   - `jaeger predict` with a PyTorch-trained model on a small FASTA.

4. **Benchmarks**:
   - Measure batches/sec on Zeus `gpu` partition for attention-based configs.
   - Compare against TF baseline on the same hardware.

5. **CI**:
   - Add a GitHub Actions job that installs the PyTorch branch and runs unit + smoke tests.
   - Keep the TF tests running until the branch is merged.

## Section 8 — Branch, release, and migration plan

- **Branch**: `pytorch_migration` created from `main`.
- **Version policy**: Bump to `1.27.1` when the branch is merged. During development, keep the current version.
- **Phase milestones**:
  1. Model architecture + parity tests
  2. `numpy_full` data loader + classifier training loop
  3. Reliability + pretraining loops
  4. CSV / `numpy_raw` loaders
  5. PyTorch inference backend
  6. SLURM script updates + Zeus benchmarks
  7. Delete TF training/inference code, update docs, final QA
- **Merge criteria**:
  - All unit and smoke tests pass.
  - Parity tests show acceptable agreement with TF on the same config.
  - Zeus benchmark shows faster training for attention/energy-based configs.
  - `jaeger predict` works end-to-end with PyTorch-trained models.
  - Bioconda recipe and install script updated to PyTorch dependencies.
- **Release**: After merge, tag and release as normal using `.github/scripts/bump-version.sh`.

## Open questions / risks

| Risk | Mitigation |
|---|---|
| Custom Keras layers (masked conv, batch norm, attention) are complex to port exactly. | Write parity tests layer-by-layer; fix mismatches before end-to-end training. |
| Mixed precision behavior differs between TF and PyTorch. | Test AMP separately; make it opt-in at first. |
| Zeus Singularity container may need a new PyTorch image. | Build and test the container early in Phase 6. |
| Variable-length padding/collate may be slower than `tf.data`. | Benchmark `numpy_full` first; optimize collate if needed. |
| Removing TF may break other commands (e.g., `quantize`, `convert_graph`, `taxonomy`). | Audit all `commands/` before deleting TF; keep TF-only commands if they have no PyTorch equivalent yet. |
