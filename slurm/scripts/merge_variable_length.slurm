#!/bin/bash
#SBATCH --job-name=jaeger_merge_variable
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/merge_variable_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/merge_variable_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

TRAIN_FILES=""
VAL_FILES=""
for size in 300 600 900 1800 3600; do
    TRAIN_FILES="$TRAIN_FILES $PROJECT/data_variable/train_shuffled_translated_${size}.npz"
    VAL_FILES="$VAL_FILES $PROJECT/data_variable/val_shuffled_translated_${size}.npz"
done

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && \
        python scripts/merge_npz_for_variable_length.py \
            --inputs $TRAIN_FILES \
            --output $PROJECT/data_variable/train_mixed_translated.npz && \
        python scripts/merge_npz_for_variable_length.py \
            --inputs $VAL_FILES \
            --output $PROJECT/data_variable/val_mixed_translated.npz"
