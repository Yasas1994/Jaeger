> I'm using the writing-plans skill to create the implementation plan.

# Multi-scale receptive-field experiment implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `parallel_branches` config primitive to the Jaeger builder, generate a multi-scale representation-learner config, and run a classifier-only experiment on Zeus.

**Architecture:** Extend `DynamicModelBuilder._build_block` to recognise a new pseudo-layer `parallel_branches` that builds several independent sub-branches on the same feature map and merges their pooled outputs. A generator script emits the YAML for four branches (RF 31/63/127/191) plus an optional single-RF baseline, and a Zeus SLURM script trains each with `--only_classification_head --xla`.

**Tech Stack:** Python 3.13, TensorFlow/Keras, YAML, SLURM/Apptainer on Zeus.

---

## Files

| File | Responsibility |
|------|----------------|
| `src/jaeger/nnlib/builder.py` | Add `parallel_branches` handling in `_build_block`. |
| `tests/pytest/test_builder_parallel_branches.py` | New test: build a minimal model with `parallel_branches` and assert output shape. |
| `receptive_field_experiments/generate_multiscale_config.py` | Generate the multi-scale YAML config and the RF 127 baseline YAML. |
| `slurm/scripts/multiscale_rf_zeus.slurm` | Zeus job for the multi-scale experiment. |
| `slurm/scripts/multiscale_rf_baseline_zeus.slurm` | Zeus job for the RF 127 single-RF baseline. |

---

### Task 1: Add `parallel_branches` support to the builder

**Files:**
- Modify: `src/jaeger/nnlib/builder.py`

**Change location:** Inside `_build_block`, after the existing special-case checks (`block_size`, `return_nmd`, `nmd`) and before the generic `x = layer_class(...)` call.

- [ ] **Step 1: Insert the `parallel_branches` branch handler**

```python
# Inside DynamicModelBuilder._build_block, after the "if layer_name == 'nmd':" block
# and before "x = layer_class(**cfg_layer)(x)", add:

if layer_name == "parallel_branches":
    merge_method = cfg_layer.get("merge", "concat").lower()
    branch_cfgs = cfg_layer.get("branches", [])
    if not branch_cfgs:
        raise ValueError("parallel_branches requires at least one branch")

    branch_outputs = []
    for b_idx, branch_cfg in enumerate(branch_cfgs):
        branch_out = self._build_block(
            x,
            branch_cfg,
            prefix=f"{prefix}_branch_{b_idx}",
            nmd_merge=None,
        )
        # Branches must return a single tensor (vectors when pooling is set).
        if isinstance(branch_out, (list, tuple)):
            branch_out = branch_out[0]
        branch_outputs.append(branch_out)

    if merge_method == "average":
        x = tf.keras.layers.Average(name=f"{prefix}_merge_avg")(branch_outputs)
    elif merge_method == "sum":
        x = tf.keras.layers.Add(name=f"{prefix}_merge_sum")(branch_outputs)
    elif merge_method == "max":
        x = tf.keras.layers.Maximum(name=f"{prefix}_merge_max")(branch_outputs)
    elif merge_method == "concat":
        x = tf.keras.layers.Concatenate(
            axis=-1, name=f"{prefix}_merge_concat"
        )(branch_outputs)
    else:
        raise ValueError(f"Unknown parallel_branches merge method: {merge_method}")

    if merge_method == "concat":
        previous_channels = sum(int(out.shape[-1]) for out in branch_outputs)
    else:
        previous_channels = int(branch_outputs[0].shape[-1])
    continue
```

- [ ] **Step 2: Run ruff/format checks**

```bash
ruff check src/jaeger/nnlib/builder.py
ruff format src/jaeger/nnlib/builder.py
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/jaeger/nnlib/builder.py
git commit -m "feat(builder): add parallel_branches pseudo-layer for multi-scale features"
```

---

### Task 2: Add a builder test for `parallel_branches`

**Files:**
- Create: `tests/pytest/test_builder_parallel_branches.py`

- [ ] **Step 1: Write the test**

```python
import pytest
import yaml
from jaeger.nnlib.builder import DynamicModelBuilder

MINIMAL_CONFIG = """
model:
  name: test_multiscale
  experiment: test
  seed: 42
  classifier_out_dim: 6
  base_dir: /tmp/jaeger_test_multiscale
  class_label_map:
    - class: bacteria
      label: 0
    - class: phage
      label: 1
    - class: eukarya
      label: 2
    - class: archaea
      label: 3
    - class: plasmid
      label: 4
    - class: virus
      label: 5
  activation: gelu
  embedding:
    use_embedding_layer: true
    input_type: translated
    strands: 2
    frames: 6
    length: null
    input_shape: [6, null]
    embedding_size: 128
    embedding_regularizer: l2
    embedding_regularizer_w: 0.00001
  string_processor:
    data_format: numpy
    seq_onehot: false
    codon: CODON
    codon_id: CODON_ID
    crop_sizes: [500]
    validation_crop_sizes: [500]
    buffer_size: 5000
    shuffle: true
    reshuffle_each_iteration: true
    mutate: false
    mutation_rate: 0.05
    shuffle_frames: false
    masking: false
    classifier_labels: [0, 1, 2, 3, 4, 5]
    classifier_labels_map: [0, 1, 2, 3, 4, 5]
  representation_learner:
    hidden_layers:
      - name: masked_conv1d
        config:
          filters: 128
          kernel_size: 7
          strides: 1
          dilation_rate: 1
          use_bias: true
          activation: null
          kernel_regularizer: l2
          kernel_regularizer_w: 0.00001
      - name: masked_layernorm
      - name: activation
        config:
          activation: gelu
      - name: parallel_branches
        config:
          merge: concat
          branches:
            - hidden_layers:
                - name: residual_block
                  config:
                    use_1x1conv: false
                    block_size: 1
                    filters: 128
                    kernel_size: 5
                    dilation_rate: 6
                    use_bias: true
                    kernel_regularizer: l2
                    kernel_regularizer_w: 0.00001
                    norm_type: masked_layernorm
              pooling: max
            - hidden_layers:
                - name: residual_block
                  config:
                    use_1x1conv: false
                    block_size: 1
                    filters: 128
                    kernel_size: 5
                    dilation_rate: 14
                    use_bias: true
                    kernel_regularizer: l2
                    kernel_regularizer_w: 0.00001
                    norm_type: masked_layernorm
              pooling: max
    pooling: null
  classifier:
    input_shape: 256
    hidden_layers:
      - name: dropout
        config: { rate: 0.1 }
      - name: dense
        config:
          units: 6
          activation: null
          dtype: float32
          use_bias: true

training:
  data_dir: /tmp/jaeger_test_data
  experiment_root: experiments/experiment_{{ model.experiment }}_{{ model.seed }}
  classifier_dir: '{{ model.base_dir }}/{{ training.experiment_root }}/checkpoints/classifier'
  classifier_epochs: 1
  classifier_train_steps: 2
  classifier_validation_steps: 1
  projection_epochs: 0
  reliability_epochs: 0
  batch_size: 64
  optimizer: adamw
  optimizer_params:
    learning_rate: 0.0003
    clipnorm: 1
  loss_classifier: categorical_crossentropy
  loss_params_classifier:
    from_logits: true
    label_smoothing: 0.1
  classifier_class_weights:
    0: 1.0
    1: 1.0
    2: 1.0
    3: 1.0
    4: 1.0
    5: 1.0
  metrics_classifier:
    - name: categorical_accuracy
      params: null
  callbacks:
    classifier: []
    projection: []
    reliability: []
  model_saving:
    path: '{{ model.base_dir }}/{{ training.experiment_root }}/model'
    save_weights: false
    save_exec_graph: false
  fragment_classifier_data:
    train: []
    validation: []
"""


def test_parallel_branches_builds_and_output_shape(tmp_path):
    cfg = yaml.safe_load(MINIMAL_CONFIG)
    base = tmp_path / "multiscale_test"
    cfg["model"]["base_dir"] = str(base)
    cfg["training"]["classifier_dir"] = str(base / "checkpoints" / "classifier")
    builder = DynamicModelBuilder(cfg)
    models = builder.build_fragment_classifier()
    rep_model = models["rep_model"]
    assert rep_model.output.shape[-1] == 256  # two 128-D branches concatenated
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/pytest/test_builder_parallel_branches.py -v
```

Expected: `test_parallel_branches_builds_and_output_shape` passes.

- [ ] **Step 3: Commit**

```bash
git add tests/pytest/test_builder_parallel_branches.py
git commit -m "test(builder): add parallel_branches build test"
```

---

### Task 3: Create the multi-scale config generator

**Files:**
- Create: `receptive_field_experiments/generate_multiscale_config.py`

- [ ] **Step 1: Write the generator**

```python
"""Generate multi-scale and baseline configs for Zeus."""

from pathlib import Path

import yaml

TEMPLATE = Path("train_config/nn_config_1500bp_nmd_merge_6_class.yaml")
OUT_DIR = Path("receptive_field_experiments")
BASE_DIR = Path("/mnt/beegfs/bioinf/wijesekara/jaeger/receptive_field_experiments")
DATA_DIR = Path("/mnt/beegfs/bioinf/wijesekara/jaeger/data/jaeger_train_data/numpy")

BRANCHES = [
    ("rf_031", 6),
    ("rf_063", 14),
    ("rf_127", 30),
    ("rf_191", 46),
]


def _residual_block(dilation):
    return {
        "name": "residual_block",
        "config": {
            "use_1x1conv": False,
            "block_size": 1,
            "filters": 128,
            "kernel_size": 5,
            "dilation_rate": dilation,
            "use_bias": True,
            "kernel_regularizer": "l2",
            "kernel_regularizer_w": 0.00001,
            "norm_type": "masked_layernorm",
            "alpha_init": 0.01,
        },
    }


def _shared_stem():
    return [
        {
            "name": "masked_conv1d",
            "config": {
                "filters": 128,
                "kernel_size": 7,
                "strides": 1,
                "dilation_rate": 1,
                "use_bias": True,
                "activation": None,
                "kernel_regularizer": "l2",
                "kernel_regularizer_w": 0.00001,
            },
        },
        {"name": "masked_layernorm"},
        {"name": "activation", "config": {"activation": "gelu"}},
    ]


def _multiscale_representation_learner():
    return {
        "hidden_layers": _shared_stem()
        + [
            {
                "name": "parallel_branches",
                "config": {
                    "merge": "concat",
                    "branches": [
                        {
                            "hidden_layers": [_residual_block(d)],
                            "pooling": "max",
                        }
                        for _, d in BRANCHES
                    ],
                },
            }
        ],
        "pooling": None,
    }


def _single_rf_representation_learner(dilation):
    return {
        "hidden_layers": _shared_stem()
        + [_residual_block(dilation), {"name": "nmd"}],
        "pooling": "max",
    }


def _configure_training(cfg):
    cfg["training"]["classifier_epochs"] = 10
    cfg["training"]["classifier_train_steps"] = 10000
    cfg["training"]["classifier_validation_steps"] = 1000
    cfg["training"]["projection_epochs"] = 0
    cfg["training"]["reliability_epochs"] = 0
    cfg["training"]["data_dir"] = str(DATA_DIR)
    cfg["model"]["string_processor"]["mutate"] = False
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(TEMPLATE.read_text())

    # Multi-scale config
    cfg = yaml.safe_load(yaml.dump(template, sort_keys=False))
    cfg["model"].pop("reliability_model", None)
    cfg["model"]["name"] = "jaeger"
    cfg["model"]["experiment"] = "multiscale_rf"
    cfg["model"]["base_dir"] = str(BASE_DIR)
    cfg["model"]["representation_learner"] = _multiscale_representation_learner()
    cfg["model"]["classifier"]["input_shape"] = 512
    cfg["model"]["projection"]["input_shape"] = 512
    cfg = _configure_training(cfg)
    (OUT_DIR / "multiscale_rf.yaml").write_text(
        yaml.dump(cfg, sort_keys=False, default_flow_style=False)
    )

    # Single-RF baseline (RF 127)
    cfg = yaml.safe_load(yaml.dump(template, sort_keys=False))
    cfg["model"].pop("reliability_model", None)
    cfg["model"]["name"] = "jaeger"
    cfg["model"]["experiment"] = "single_rf_127_baseline"
    cfg["model"]["base_dir"] = str(BASE_DIR)
    cfg["model"]["representation_learner"] = _single_rf_representation_learner(30)
    cfg["model"]["classifier"]["input_shape"] = 128
    cfg["model"]["projection"]["input_shape"] = 128
    cfg = _configure_training(cfg)
    (OUT_DIR / "single_rf_127_baseline.yaml").write_text(
        yaml.dump(cfg, sort_keys=False, default_flow_style=False)
    )

    print("Wrote multiscale_rf.yaml and single_rf_127_baseline.yaml")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator locally**

```bash
python receptive_field_experiments/generate_multiscale_config.py
```

Expected: prints `Wrote multiscale_rf.yaml and single_rf_127_baseline.yaml`.

- [ ] **Step 3: Verify RF with CLI**

```bash
jaeger utils receptive_field -c receptive_field_experiments/multiscale_rf.yaml
jaeger utils receptive_field -c receptive_field_experiments/single_rf_127_baseline.yaml
```

Expected: shows the per-branch RFs (31, 63, 127, 191) for multi-scale and RF 127 for baseline.

- [ ] **Step 4: Commit**

```bash
git add receptive_field_experiments/generate_multiscale_config.py \
        receptive_field_experiments/multiscale_rf.yaml \
        receptive_field_experiments/single_rf_127_baseline.yaml
git commit -m "feat(configs): add multi-scale and RF 127 baseline generator"
```

---

### Task 4: Create the Zeus SLURM script for the multi-scale experiment

**Files:**
- Create: `slurm/scripts/multiscale_rf_zeus.slurm`

- [ ] **Step 1: Write the SLURM script**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=multiscale_rf
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node030

set -euo pipefail

PROJECT_ROOT="/mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger"
CONTAINER="/mnt/beegfs/bioinf/wijesekara/jaeger/container/jaeger_dev.sif"
JAEGER_TRAIN_CONFIG="$PROJECT_ROOT/receptive_field_experiments/multiscale_rf.yaml"

JAEGER_FLAGS="-c $JAEGER_TRAIN_CONFIG --only_classification_head --force --xla"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

cd "$PROJECT_ROOT" || die "Could not cd to $PROJECT_ROOT"

[[ -f "$CONTAINER" ]] || die "Container not found: $CONTAINER"
[[ -f "$JAEGER_TRAIN_CONFIG" ]] || die "Config not found: $JAEGER_TRAIN_CONFIG"
command -v apptainer >/dev/null 2>&1 || die "apptainer not found"

mkdir -p /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs

LOG_FILE="/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/multiscale_rf_${SLURM_JOB_ID}.log"

srun bash -lc "
echo \"[\$(date)] Task \$SLURM_PROCID on node \$SLURMD_NODENAME: GPU \$CUDA_VISIBLE_DEVICES, job \$SLURM_JOB_ID\" >&2
apptainer run \
    --bind \"/mnt/beegfs\" \
    --bind \"$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger\" \
    --nv $CONTAINER jaeger train $JAEGER_FLAGS
" 2>&1 | tee -a "$LOG_FILE" || die "Training failed"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x slurm/scripts/multiscale_rf_zeus.slurm
```

- [ ] **Step 3: Commit**

```bash
git add slurm/scripts/multiscale_rf_zeus.slurm
git commit -m "feat(slurm): add Zeus job for multi-scale RF experiment"
```

---

### Task 5: Create the Zeus SLURM script for the RF 127 baseline

**Files:**
- Create: `slurm/scripts/multiscale_rf_baseline_zeus.slurm`

- [ ] **Step 1: Write the SLURM script**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=single_rf_127_baseline
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node030

set -euo pipefail

PROJECT_ROOT="/mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger"
CONTAINER="/mnt/beegfs/bioinf/wijesekara/jaeger/container/jaeger_dev.sif"
JAEGER_TRAIN_CONFIG="$PROJECT_ROOT/receptive_field_experiments/single_rf_127_baseline.yaml"

JAEGER_FLAGS="-c $JAEGER_TRAIN_CONFIG --only_classification_head --force --xla"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

cd "$PROJECT_ROOT" || die "Could not cd to $PROJECT_ROOT"

[[ -f "$CONTAINER" ]] || die "Container not found: $CONTAINER"
[[ -f "$JAEGER_TRAIN_CONFIG" ]] || die "Config not found: $JAEGER_TRAIN_CONFIG"
command -v apptainer >/dev/null 2>&1 || die "apptainer not found"

mkdir -p /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs

LOG_FILE="/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/single_rf_127_baseline_${SLURM_JOB_ID}.log"

srun bash -lc "
echo \"[\$(date)] Task \$SLURM_PROCID on node \$SLURMD_NODENAME: GPU \$CUDA_VISIBLE_DEVICES, job \$SLURM_JOB_ID\" >&2
apptainer run \
    --bind \"/mnt/beegfs\" \
    --bind \"$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger\" \
    --nv $CONTAINER jaeger train $JAEGER_FLAGS
" 2>&1 | tee -a "$LOG_FILE" || die "Training failed"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x slurm/scripts/multiscale_rf_baseline_zeus.slurm
```

- [ ] **Step 3: Commit**

```bash
git add slurm/scripts/multiscale_rf_baseline_zeus.slurm
git commit -m "feat(slurm): add Zeus job for RF 127 baseline"
```

---

### Task 6: Local smoke test before Zeus submission

**Files:**
- Use: `receptive_field_experiments/multiscale_rf.yaml`

- [ ] **Step 1: Create a tiny smoke config**

```bash
python - <<'PY'
import yaml
import tempfile
from pathlib import Path
src = Path('receptive_field_experiments/multiscale_rf.yaml')
cfg = yaml.safe_load(src.read_text())
cfg['training']['classifier_epochs'] = 1
cfg['training']['classifier_train_steps'] = 10
cfg['training']['classifier_validation_steps'] = 5
cfg['model']['base_dir'] = tempfile.mkdtemp(prefix='multiscale_smoke_')
Path('receptive_field_experiments/_multiscale_smoke.yaml').write_text(
    yaml.dump(cfg, sort_keys=False, default_flow_style=False)
)
PY
```

- [ ] **Step 2: Run the smoke test**

```bash
jaeger train -c receptive_field_experiments/_multiscale_smoke.yaml \
    --only_classification_head --force --xla 2>&1 | tail -n 20
```

Expected: model builds, trains 10 steps, validation runs, exits with `training completed!`.

- [ ] **Step 3: Clean up the smoke config**

```bash
rm -f receptive_field_experiments/_multiscale_smoke.yaml
```

---

### Task 7: Submit to Zeus

**Files:**
- Use: `slurm/scripts/multiscale_rf_zeus.slurm`, `slurm/scripts/multiscale_rf_baseline_zeus.slurm`

- [ ] **Step 1: Submit both jobs**

```bash
sbatch slurm/scripts/multiscale_rf_zeus.slurm
sbatch slurm/scripts/multiscale_rf_baseline_zeus.slurm
```

Expected: `Submitted batch job <jobid>` for each.

- [ ] **Step 2: Monitor**

```bash
squeue -u $USER
```

Expected: both jobs are queued/running.

- [ ] **Step 3: Inspect logs once running**

```bash
tail -f /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/multiscale_rf_<jobid>.out
tail -f /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/single_rf_127_baseline_<jobid>.out
```

---

## Spec coverage

| Spec section | Implementing task |
|--------------|-------------------|
| Add `parallel_branches` pseudo-layer | Task 1 |
| Config DSL with 4 branches + concat | Task 3 |
| Classifier input shape 512 | Task 3 |
| Zeus job, classifier-only, 10 epochs, 10k/1k steps, no mutation, XLA | Task 4 |
| Optional RF 127 baseline | Task 5 |
| Local smoke test | Task 6 |
| Submit and monitor | Task 7 |

## Runtime fixes applied during Task 7

- The generated configs use `training.data_dir: /mnt/beegfs/bioinf/wijesekara/jaeger/data/jaeger_train_data/numpy` so the NumPy training files are found on Zeus.
- The entire `src/jaeger` tree must be synced to Zeus before submission because the SLURM script bind-mounts it over the container's installed package. The initial submission failed with `ImportError: cannot import name 'MacroF1Score' from 'jaeger.nnlib.metrics'` because the remote copy was stale; a full `rsync --delete` resolved it.

## Placeholder scan

No `TBD`, `TODO`, or vague requirements remain. SLURM paths match the existing `full_jaeger_pipeline_zeus.slurm` template.
