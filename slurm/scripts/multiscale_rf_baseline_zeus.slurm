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
