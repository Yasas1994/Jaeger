#!/bin/bash
#SBATCH --job-name=jaeger_500bp_ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/ddp_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/ddp_%j.err

echo "=== Jaeger 500bp PyTorch DDP training ==="
echo "Running on: $(hostname)"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Started at: $(date)"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Job-specific install directory to avoid conflicts
INSTALL_DIR=/mnt/beegfs/bioinf/wijesekara/jaeger/.local_${SLURM_JOB_ID}_ddp
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
  bash -c "
    cd /jaeger
    pip install --target=/home/wijesekara/.local/lib/python3.12/site-packages --no-deps -e .

    srun torchrun \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$SLURM_GPUS_PER_NODE \
      --rdzv_id=jaeger_ddp_$SLURM_JOB_ID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1):29500 \
      jaeger train -c /jaeger/train_config/nn_config_500bp_baseline_zeus.yaml --force
  "

echo "Finished at: $(date)"
