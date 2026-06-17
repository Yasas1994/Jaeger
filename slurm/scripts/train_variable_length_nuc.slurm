#!/bin/bash
#SBATCH --job-name=jaeger_train_variable_nuc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/train_variable_nuc_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/train_variable_nuc_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
SRC=$PROJECT/Jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif
CONFIG=$PROJECT/configs/nn_config_variable_length_nuc.yaml

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

apptainer exec --nv \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "cd /jaeger_src && jaeger train -c $CONFIG --force --xla --mixed_precision"
