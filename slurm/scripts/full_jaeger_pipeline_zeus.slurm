#!/usr/bin/env bash
#SBATCH --job-name=jaeger_1500bp_nmd_merge_6_class
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/%x_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/%x_%j.err
#SBATCH -n 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=250G
#SBATCH --time=72:00:00
#SBATCH --partition=vision
#SBATCH --gres=gpu:1

# Full Jaeger training-to-evaluation pipeline for Zeus.
#
# 1. Train fragment classifier
# 2. Generate reliability (OOD) training data
# 3. Train reliability head
# 4. Save the final combined model
# 5. Tune reliability threshold on reliability_val.npz
# 6. Predict on real-world metagenome assemblies
# 7. Calculate precision/recall/F1/confusion matrices against fraction labels
#
# Usage:
#   sbatch slurm/scripts/full_jaeger_pipeline_zeus.slurm

set -euo pipefail

# ============================================================================
# ---- Configuration ----
# ============================================================================

TEST="1500bp_nmd_merge_6_class"

PROJECT_ROOT="/mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger"
CONTAINER="/mnt/beegfs/bioinf/wijesekara/jaeger/container/jaeger_dev.sif"
JAEGER_TRAIN_CONFIG="/mnt/beegfs/bioinf/wijesekara/jaeger/configs/nn_config_1500bp_nmd_merge_6_class_zeus.yaml"
JAEGER_MODEL_CONFIG="/mnt/beegfs/bioinf/wijesekara/jaeger/configs/config.json"
SINGULARITY_METADATA="/mnt/beegfs/bioinf/wijesekara/jaeger/tmp/${TEST}_train_meta.json"

# Training flags
FROM_LAST_CHECKPOINT=false
ONLY_HEADS=false
MIXED_PRECISION=true
XLA=false
WRITE_WINDOW_SCORES=false
OVERWRITE=false

# Inference settings
STRIDE=1500
FSIZE=1500
BATCH=96

# Test data (FASTA files)
TEST_DATA=(
  "/mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data/gut_scaffolds_gt1500.fasta"
  "/mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data/seawater_scaffolds_gt1500.fasta"
  "/mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data/soil_scaffolds_gt1500.fasta"
)

# Real-world fraction labels: one TSV per sample, named <sample>_labels.tsv
TEST_LABELS_DIR="/mnt/beegfs/bioinf/wijesekara/jaeger/benchmark_data"

# Threshold tuning metric: f1-id | f1-ood | youden | mcc
THRESHOLD_METRIC="f1-id"

# Logs
PIPELINE_LOG="$(dirname "$SINGULARITY_METADATA")/${TEST}_pipeline.log"

# ============================================================================
# ---- Helper functions ----
# ============================================================================

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$PIPELINE_LOG"
}

die() {
    log "ERROR: $*"
    exit 1
}

file_exists() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

# ============================================================================
# ---- Pre-flight checks ----
# ============================================================================

mkdir -p "$(dirname "$SINGULARITY_METADATA")"
mkdir -p "$(dirname "$PIPELINE_LOG")"

cd "$PROJECT_ROOT" || die "Could not cd to $PROJECT_ROOT"

log "Starting Jaeger pipeline for experiment: $TEST"

file_exists "$CONTAINER"
file_exists "$JAEGER_TRAIN_CONFIG"
file_exists "$JAEGER_MODEL_CONFIG"

command -v jq >/dev/null 2>&1 || die "jq is required but not installed"
command -v singularity >/dev/null 2>&1 || die "singularity is required but not installed"

for fasta in "${TEST_DATA[@]}"; do
    file_exists "$fasta"
    label_file="${TEST_LABELS_DIR}/$(basename "$fasta" .fasta)_labels.tsv"
    file_exists "$label_file"
done

# ============================================================================
# ---- Training ----
# ============================================================================

TRAIN_FLAGS="-c $JAEGER_TRAIN_CONFIG --generate_reliability_data --save_model --meta $SINGULARITY_METADATA"

if [[ "$MIXED_PRECISION" == true ]]; then
    TRAIN_FLAGS="$TRAIN_FLAGS --mixed_precision"
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

log "Training Jaeger model with flags: $TRAIN_FLAGS"

srun bash -lc "
echo \"[\$(date)] Task \$SLURM_PROCID on node \$SLURMD_NODENAME: GPU \$CUDA_VISIBLE_DEVICES, job \$SLURM_JOB_ID\" >&2
singularity run --nv $CONTAINER jaeger train $TRAIN_FLAGS
" 2>> "$PIPELINE_LOG" || die "Training failed"

# ============================================================================
# ---- Extract model and experiment paths from metadata ----
# ============================================================================

[[ -f "$SINGULARITY_METADATA" ]] || die "Training metadata file was not created: $SINGULARITY_METADATA"

MODEL_PATH=$(jq -r '.model_path' "$SINGULARITY_METADATA")
EXPERIMENT_PATH=$(jq -r '.experiment_path' "$SINGULARITY_METADATA")

[[ -n "$MODEL_PATH" && "$MODEL_PATH" != "null" ]] || die "MODEL_PATH is empty"
[[ -n "$EXPERIMENT_PATH" && "$EXPERIMENT_PATH" != "null" ]] || die "EXPERIMENT_PATH is empty"

log "MODEL_PATH=$MODEL_PATH"
log "EXPERIMENT_PATH=$EXPERIMENT_PATH"

mkdir -p "$EXPERIMENT_PATH/predictions"
mkdir -p "$EXPERIMENT_PATH/metrics"

# ============================================================================
# ---- Locate reliability validation NPZ ----
# ============================================================================

RELIABILITY_OUTPUT_DIR=$(python3 - <<'PY'
import yaml, sys
path = sys.argv[1]
with open(path) as f:
    cfg = yaml.safe_load(f)
out = cfg.get("training", {}).get("reliability_data_generation", {}).get("output_dir")
if out is None:
    base = cfg.get("model", {}).get("base_dir", "")
    data_dir = cfg.get("training", {}).get("data_dir", "")
    out = f"{base}/{data_dir}/reliability_data".replace("//", "/")
print(out)
PY
"$JAEGER_TRAIN_CONFIG")

RELIABILITY_VAL_NPZ="${RELIABILITY_OUTPUT_DIR}/reliability_val.npz"

file_exists "$RELIABILITY_VAL_NPZ"

log "Reliability validation NPZ: $RELIABILITY_VAL_NPZ"

# ============================================================================
# ---- Tune reliability threshold ----
# ============================================================================

BEST_RC_FILE="$EXPERIMENT_PATH/metrics/best_reliability_threshold.txt"

log "Tuning reliability threshold (metric=$THRESHOLD_METRIC)"

singularity run \
    --bind "$PROJECT_ROOT/scripts:/scripts" \
    --nv "$CONTAINER" \
    python /scripts/tune_reliability_threshold.py \
    --model-path "$MODEL_PATH" \
    --val-npz "$RELIABILITY_VAL_NPZ" \
    --output "$BEST_RC_FILE" \
    --metric "$THRESHOLD_METRIC" \
    --min-threshold 0.0 \
    --max-threshold 0.95 \
    --step 0.05 \
    --batch-size "$BATCH" \
    2>> "$PIPELINE_LOG" || die "Threshold tuning failed"

BEST_RC=$(cat "$BEST_RC_FILE" | tr -d '[:space:]')
[[ -n "$BEST_RC" ]] || die "Best reliability threshold is empty"
log "Selected reliability cutoff: $BEST_RC"

# ============================================================================
# ---- Inference on test metagenomes ----
# ============================================================================

PREDICTIONS_DIR="$EXPERIMENT_PATH/predictions"
mkdir -p "$PREDICTIONS_DIR"

for fasta in "${TEST_DATA[@]}"; do
    sample_name=$(basename "$fasta" .fasta)
    output_tsv="$PREDICTIONS_DIR/${sample_name}/${sample_name}.tsv"

    if [[ -f "$output_tsv" && "$OVERWRITE" != true ]]; then
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

    singularity run --nv "$CONTAINER" jaeger predict \
        --model_path "$MODEL_PATH" \
        --config "$JAEGER_MODEL_CONFIG" \
        -i "$fasta" \
        -o "$PREDICTIONS_DIR" \
        --stride "$STRIDE" \
        --fsize "$FSIZE" \
        --batch "$BATCH" \
        --rc "$BEST_RC" \
        --getalllabels \
        $WINDOW_FLAG \
        $OVERWRITE_FLAG \
        2>> "$PIPELINE_LOG" || die "Prediction failed for $sample_name"
done

# ============================================================================
# ---- Calculate metrics ----
# ============================================================================

log "Calculating real-world metrics"

singularity run \
    --bind "$PROJECT_ROOT/scripts:/scripts" \
    --nv "$CONTAINER" \
    python /scripts/calculate_metrics_realworld.py \
    --predictions-dir "$PREDICTIONS_DIR" \
    --labels-dir "$TEST_LABELS_DIR" \
    --output-dir "$EXPERIMENT_PATH/metrics" \
    --reliability-cutoff "$BEST_RC" \
    2>> "$PIPELINE_LOG" || die "Metric calculation failed"

log "Pipeline completed successfully"
log "Metrics written to: $EXPERIMENT_PATH/metrics"
