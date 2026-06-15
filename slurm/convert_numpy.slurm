#!/bin/bash
#SBATCH --job-name=convert_numpy
#SBATCH --partition=gpu
#SBATCH --nodelist=node030
#SBATCH --gpus=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_numpy_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_numpy_%j.err

echo "=== NumPy conversion (PyTorch training data) ==="
echo "Started at: $(date)"
echo "CPUs: $(nproc)"

# Run conversion inside container where PyTorch/NumPy are installed
apptainer exec \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/data:/data \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  bash -c "
    cd /jaeger
    pip install -e . --quiet 2>/dev/null

    echo 'Converting training data...'
    jaeger utils optimize-data \
      -i /data/train_shuffled.csv \
      -o /data/train_shuffled_full.npz \
      --format numpy_full \
      --crop-size 500 \
      --num-classes 3

    echo 'Converting validation data...'
    jaeger utils optimize-data \
      -i /data/val_shuffled.csv \
      -o /data/val_shuffled_full.npz \
      --format numpy_full \
      --crop-size 500 \
      --num-classes 3
  "

echo "Finished at: $(date)"
ls -lh /mnt/beegfs/bioinf/wijesekara/jaeger/data/*.npz
