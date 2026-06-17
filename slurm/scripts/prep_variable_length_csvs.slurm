#!/bin/bash
#SBATCH --job-name=jaeger_prep_variable_csvs
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/prep_variable_csvs_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/prep_variable_csvs_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && python scripts/prepare_length_csvs.py \
        --fasta $PROJECT/data/train_test_sequences.fasta \
        --tsv $PROJECT/data/sequence_weights.tsv \
        --out-dir $PROJECT/data_variable \
        --lengths 300 600 900 1800 3600"
