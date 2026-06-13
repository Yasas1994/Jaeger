#!/bin/bash
#SBATCH --job-name=jaeger_container_test
#SBATCH --partition=gpu
#SBATCH --nodelist=node030
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/container_test_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/container_test_%j.err

echo "=== Running on $(hostname) ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

# Test 1: Container basic execution
echo ""
echo "=== Test 1: Jaeger version ==="
apptainer exec --nv \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  jaeger --version 2>&1 || echo "JAEGER VERSION FAILED"

# Test 2: TensorFlow GPU detection
echo ""
echo "=== Test 2: TensorFlow GPU ==="
apptainer exec --nv \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  python3 -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))" 2>&1 || echo "TF GPU TEST FAILED"

# Test 3: Install local Jaeger and run help
echo ""
echo "=== Test 3: Install local Jaeger ==="
apptainer exec --nv \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  bash -c "cd /jaeger && pip install -e . --quiet && jaeger train --help | head -5" 2>&1 || echo "INSTALL TEST FAILED"

echo ""
echo "=== Test complete at $(date) ==="
