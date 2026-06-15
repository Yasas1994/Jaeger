#!/bin/bash
#SBATCH --job-name=jaeger_500bp_crossframe
#SBATCH --partition=gpu
#SBATCH --nodelist=node030
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/crossframe_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/crossframe_%j.err

echo "=== Jaeger 500bp crossframe training (PyTorch) ==="
echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Job-specific install directory to avoid conflicts
INSTALL_DIR=/mnt/beegfs/bioinf/wijesekara/jaeger/.local_${SLURM_JOB_ID}_crossframe
rm -rf $INSTALL_DIR
mkdir -p $INSTALL_DIR/lib/python3.12/site-packages
mkdir -p $INSTALL_DIR/bin
export PYTHONPATH="$INSTALL_DIR/lib/python3.12/site-packages:$PYTHONPATH"
export PATH="$INSTALL_DIR/bin:$PATH"

# Ensure experiments directory exists
mkdir -p /mnt/beegfs/bioinf/wijesekara/jaeger/experiments

apptainer exec --nv \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/data:/data \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/experiments:/data/experiments \
  --bind $INSTALL_DIR:/home/wijesekara/.local \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  bash -c "cd /jaeger && pip install --target=/home/wijesekara/.local/lib/python3.12/site-packages --no-deps -e . && jaeger train -c /jaeger/train_config/nn_config_500bp_crossframe_zeus.yaml --force"

echo "Finished at: $(date)"
