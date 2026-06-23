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

cd "$PROJECT_ROOT" || exit 1

mkdir -p /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs

srun bash -lc "
echo '[\$(date)] Task \$SLURM_PROCID on node \$SLURMD_NODENAME: GPU \$CUDA_VISIBLE_DEVICES, job \$SLURM_JOB_ID' >&2
apptainer run \
    --bind /mnt/beegfs \
    --bind '$PROJECT_ROOT/src/jaeger:/usr/local/lib/python3.12/site-packages/jaeger' \
    --nv $CONTAINER jaeger train $JAEGER_FLAGS
" 2>&1 | tee -a /mnt/beegfs/bioinf/wijesekara/jaeger/slurm/slurm_logs/multiscale_rf_\${SLURM_JOB_ID}.log
