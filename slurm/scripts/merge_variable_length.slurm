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

# Dependency note: this job must run after both optimize jobs finish.
# Capture the IDs from the two optimize submissions and submit merge with:
#   sbatch --dependency=afterok:<trans_jobid>:<nuc_jobid> slurm/scripts/merge_variable_length.slurm

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && \
        for split in train val; do \
            TRANS_FILES=; \
            NUC_FILES=; \
            for size in 300 600 900 1800 3600; do \
                TRANS_FILES=\"\$TRANS_FILES \$PROJECT/data_variable/\${split}_shuffled_translated_\${size}.npz\"; \
                NUC_FILES=\"\$NUC_FILES \$PROJECT/data_variable/\${split}_shuffled_nuc_\${size}.npz\"; \
            done; \
            python scripts/merge_npz_for_variable_length.py \
                --inputs \$TRANS_FILES \
                --output \$PROJECT/data_variable/\${split}_mixed_translated.npz; \
            python scripts/merge_npz_for_variable_length.py \
                --inputs \$NUC_FILES \
                --output \$PROJECT/data_variable/\${split}_mixed_nucleotide.npz; \
        done"
