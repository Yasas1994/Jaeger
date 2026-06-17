#!/bin/bash
#SBATCH --job-name=jaeger_opt_variable_trans
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/opt_variable_trans_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/opt_variable_trans_%j.err

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
        for size in 300 600 900 1800 3600; do \
            for split in train val; do \
                jaeger utils optimize-data \
                    --input \$PROJECT/data_variable/\${split}_\${size}.csv \
                    --output \$PROJECT/data_variable/\${split}_shuffled_translated_\${size}.npz \
                    --format translated \
                    --crop-size \$size \
                    --stride 0 \
                    --num-classes 3 \
                    --num-workers 4 \
                    --compress fast; \
            done; \
        done"
