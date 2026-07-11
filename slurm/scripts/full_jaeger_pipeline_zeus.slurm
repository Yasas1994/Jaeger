#!/usr/bin/env bash
#SBATCH --job-name=jaeger
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node030

# Full Jaeger training-to-evaluation pipeline.
#
# 1. Train fragment classifier (+ optional self-supervised projection pretraining)
# 2. Generate reliability (OOD) training data
# 3. Train reliability head
# 4. Tune the reliability threshold (inside 'jaeger train') and save the model
# 5. Predict on real-world metagenome assemblies
# 6. Calculate precision/recall/F1/confusion matrices against fraction labels
#
# Usage on Zeus:
#   sbatch slurm/scripts/full_jaeger_pipeline_zeus.slurm
#
# Local testing (no SLURM / apptainer): every configuration variable can be
# overridden from the environment, e.g.
#   MODE=local PROJECT_ROOT=$PWD JAEGER_BIN=/path/to/jaeger \
#     JAEGER_TRAIN_CONFIG=/path/to/config.yaml TEST_DATA="a.fasta b.fasta" \
#     TEST_LABELS_DIR=/path/to/labels bash slurm/scripts/full_jaeger_pipeline_zeus.slurm

set -euo pipefail

# ============================================================================
# ---- Configuration (all overridable via environment) ----
# ============================================================================

MODE="${MODE:-zeus}"  # zeus | local

EXPERIMENT_NUMBER="${EXPERIMENT_NUMBER:-90}"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger}"
CONTAINER="${CONTAINER:-/mnt/beegfs/bioinf/wijesekara/jaeger/container/jaeger_dev.sif}"
JAEGER_TRAIN_CONFIG="${JAEGER_TRAIN_CONFIG:-/mnt/beegfs/bioinf/wijesekara/jaeger/configs/jaeger_experiment_${EXPERIMENT_NUMBER}.yaml}"
JAEGER_MODEL_CONFIG="${JAEGER_MODEL_CONFIG:-/mnt/beegfs/bioinf/wijesekara/jaeger/configs/config.json}"
SINGULARITY_METADATA="${SINGULARITY_METADATA:-/mnt/beegfs/bioinf/wijesekara/jaeger/tmp/${EXPERIMENT_NUMBER}_train_meta.json}"
SLURM_DIR="${SLURM_DIR:-/mnt/beegfs/bioinf/wijesekara/jaeger/slurm}"

# Executables for local mode (ignored in zeus mode, where the container provides them)
JAEGER_BIN="${JAEGER_BIN:-jaeger}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

# Training flags
FROM_LAST_CHECKPOINT="${FROM_LAST_CHECKPOINT:-true}"
FORCE="${FORCE:-false}"
ONLY_HEADS="${ONLY_HEADS:-false}"
MIXED_PRECISION="${MIXED_PRECISION:-true}"
PRECISION="${PRECISION:-bf16}"
XLA="${XLA:-true}"
GENERATE_RELIABILITY_DATA="${GENERATE_RELIABILITY_DATA:-true}"
SELF_SUPERVISED_PRETRAINING="${SELF_SUPERVISED_PRETRAINING:-true}"

# Inference settings
STRIDE="${STRIDE:-1500}"
FSIZE="${FSIZE:-2000}"
MIN_LEN="${MIN_LEN:-1500}"
BATCH="${BATCH:-96}"
WRITE_WINDOW_SCORES="${WRITE_WINDOW_SCORES:-false}"
OVERWRITE="${OVERWRITE:-false}"

# Test data (space-separated FASTA paths; overridable via environment)
TEST_DATA="${TEST_DATA:-/mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data/gut_scaffolds_gt1500.fasta /mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data/seawater_scaffolds_gt1500.fasta /mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data/soil_scaffolds_gt1500.fasta}"
read -ra TEST_FASTAS <<<"$TEST_DATA"

# Real-world fraction labels: one TSV per sample, named <sample>_labels.tsv
TEST_LABELS_DIR="${TEST_LABELS_DIR:-/mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data}"

# Fallback reliability cutoff, used only when training did not write a tuned
# threshold (reliability_threshold.txt). The tuned threshold is on the
# probability scale and matches the metric configured under
# training.reliability_threshold.metric in the training YAML.
BEST_RC="${BEST_RC:-0}"

# Logs
PIPELINE_LOG="${PIPELINE_LOG:-${SLURM_DIR}/pipeline_logs/${EXPERIMENT_NUMBER}_pipeline.log}"

# ============================================================================
# ---- Helper functions ----
# ============================================================================

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >>"$PIPELINE_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

file_exists() {
  [[ -f "$1" ]] || die "Required file not found: $1"
}

# Mirror of the prefix matching done by calculate_metrics_realworld.py:
# predictions inherit FASTA names (e.g. gut_scaffolds_gt1500) while labels are
# stored as <sample>_labels.tsv (e.g. gut_labels). Match the longest
# underscore-separated prefix of the FASTA stem that has a label file.
find_label_file() {
  local candidate="$1"
  while true; do
    if [[ -f "$TEST_LABELS_DIR/${candidate}_labels.tsv" ]]; then
      printf '%s\n' "$TEST_LABELS_DIR/${candidate}_labels.tsv"
      return 0
    fi
    [[ "$candidate" == *_* ]] || break
    candidate="${candidate%_*}"
  done
  return 1
}

# Run a step directly on this node (containerized on zeus, plain locally).
run_step() {
  "$@" 2>&1 | tee -a "$PIPELINE_LOG"
}

# Run a heavy step on an allocated compute node via srun (zeus) or directly (local).
run_srun_step() {
  if [[ "$MODE" == "zeus" ]]; then
    srun bash -lc \
      "echo \"[\$(date)] Task \$SLURM_PROCID on node \$SLURMD_NODENAME: GPU \$CUDA_VISIBLE_DEVICES, job \$SLURM_JOB_ID\" >&2; $(printf '%q ' "$@")" \
      2>&1 | tee -a "$PIPELINE_LOG"
  else
    run_step "$@"
  fi
}

# ============================================================================
# ---- Pre-flight checks ----
# ============================================================================

mkdir -p "$(dirname "$SINGULARITY_METADATA")"
mkdir -p "$(dirname "$PIPELINE_LOG")"

cd "$PROJECT_ROOT" || die "Could not cd to $PROJECT_ROOT"

log "Starting Jaeger pipeline (mode=$MODE) for experiment: $EXPERIMENT_NUMBER"

if [[ "$MODE" == "zeus" ]]; then
  file_exists "$CONTAINER"
  command -v apptainer >/dev/null 2>&1 || die "apptainer is required but not installed"
  CONTAINER_BIND="$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger"
  JAEGER_RUN=(apptainer run --bind /mnt/beegfs --bind "$CONTAINER_BIND" --nv "$CONTAINER" jaeger)
else
  command -v "$JAEGER_BIN" >/dev/null 2>&1 || die "jaeger executable not found: $JAEGER_BIN"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python not found: $PYTHON_BIN"
  JAEGER_RUN=("$JAEGER_BIN")
fi

file_exists "$JAEGER_TRAIN_CONFIG"
file_exists "$JAEGER_MODEL_CONFIG"

for fasta in "${TEST_FASTAS[@]}"; do
  file_exists "$fasta"
  sample_name=$(basename "$fasta" .fasta)
  find_label_file "$sample_name" >/dev/null \
    || die "No label file matching sample '$sample_name' in $TEST_LABELS_DIR"
done

# ============================================================================
# ---- Training ----
# ============================================================================

TRAIN_FLAGS="-c $JAEGER_TRAIN_CONFIG --save_model --meta $SINGULARITY_METADATA"

if [[ "$MIXED_PRECISION" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --precision $PRECISION"
fi

if [[ "$FORCE" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --force"
fi

if [[ "$FROM_LAST_CHECKPOINT" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --from_last_checkpoint"
fi

if [[ "$ONLY_HEADS" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --only_heads"
fi

if [[ "$XLA" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --xla"
fi

if [[ "$GENERATE_RELIABILITY_DATA" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --generate_reliability_data"
fi

if [[ "$SELF_SUPERVISED_PRETRAINING" == true ]]; then
  TRAIN_FLAGS="$TRAIN_FLAGS --self_supervised_pretraining"
fi

log "Training Jaeger model with flags: $TRAIN_FLAGS"

# shellcheck disable=SC2086
run_srun_step "${JAEGER_RUN[@]}" train $TRAIN_FLAGS || die "Training failed"

# ============================================================================
# ---- Extract model and experiment paths from metadata ----
# ============================================================================

[[ -f "$SINGULARITY_METADATA" ]] || die "Training metadata file was not created: $SINGULARITY_METADATA"

MODEL_PATH=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['model_path'])" "$SINGULARITY_METADATA")
EXPERIMENT_PATH=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['experiment_path'])" "$SINGULARITY_METADATA")

[[ -n "$MODEL_PATH" && "$MODEL_PATH" != "null" ]] || die "MODEL_PATH is empty"
[[ -n "$EXPERIMENT_PATH" && "$EXPERIMENT_PATH" != "null" ]] || die "EXPERIMENT_PATH is empty"

log "MODEL_PATH=$MODEL_PATH"
log "EXPERIMENT_PATH=$EXPERIMENT_PATH"

# Prefer the reliability threshold tuned during training; fall back to BEST_RC.
THRESHOLD_FILE="$EXPERIMENT_PATH/checkpoints/reliability/reliability_threshold.txt"
if [[ -f "$THRESHOLD_FILE" ]]; then
  BEST_RC=$(tr -d '[:space:]' <"$THRESHOLD_FILE")
  log "Using tuned reliability threshold: $BEST_RC"
else
  log "No tuned threshold found at $THRESHOLD_FILE; using BEST_RC=$BEST_RC"
fi

PREDICTIONS_DIR="$EXPERIMENT_PATH/predictions"

mkdir -p "$PREDICTIONS_DIR"
mkdir -p "$EXPERIMENT_PATH/metrics"

for fasta in "${TEST_FASTAS[@]}"; do
  sample_name=$(basename "$fasta" .fasta)

  if [[ "$OVERWRITE" != true ]] && compgen -G "$PREDICTIONS_DIR/*/${sample_name}.tsv" >/dev/null; then
    log "Skipping prediction for $sample_name (output exists)"
    continue
  fi

  log "Running prediction on: $sample_name"

  WINDOW_FLAG=""
  if [[ "$WRITE_WINDOW_SCORES" == true ]]; then
    WINDOW_FLAG="--window-scores"
  fi

  OVERWRITE_FLAG=""
  if [[ "$OVERWRITE" == true ]]; then
    OVERWRITE_FLAG="--overwrite"
  fi

  # shellcheck disable=SC2086
  run_step "${JAEGER_RUN[@]}" predict \
    --model_path "$MODEL_PATH" \
    --config "$JAEGER_MODEL_CONFIG" \
    -i "$fasta" \
    -o "$PREDICTIONS_DIR" \
    --stride "$STRIDE" \
    --fsize "$FSIZE" \
    --min-len "$MIN_LEN" \
    --batch "$BATCH" \
    --rc "$BEST_RC" \
    $WINDOW_FLAG \
    $OVERWRITE_FLAG \
    || die "Prediction failed for $sample_name"
done

# ============================================================================
# ---- Calculate metrics ----
# ============================================================================

# 'jaeger predict' writes TSVs to <output>/<model_id>/<sample>.tsv; resolve the
# single model subdirectory that was just produced.
mapfile -t PRED_SUBDIRS < <(find "$PREDICTIONS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
[[ ${#PRED_SUBDIRS[@]} -ge 1 ]] || die "No prediction outputs found under $PREDICTIONS_DIR"
if [[ ${#PRED_SUBDIRS[@]} -gt 1 ]]; then
  log "WARNING: multiple prediction subdirs found; using ${PRED_SUBDIRS[0]}"
fi
PRED_OUTPUT_DIR="${PRED_SUBDIRS[0]}"

log "Calculating real-world metrics from $PRED_OUTPUT_DIR"

run_step "$PYTHON_BIN" "$PROJECT_ROOT/scripts/calculate_metrics_realworld.py" \
  --predictions-dir "$PRED_OUTPUT_DIR" \
  --labels-dir "$TEST_LABELS_DIR" \
  --output-dir "$EXPERIMENT_PATH/metrics" \
  --reliability-cutoff "$BEST_RC" \
  || die "Metric calculation failed"

log "Pipeline completed successfully"
log "Metrics written to: $EXPERIMENT_PATH/metrics"
