#!/bin/bash
#SBATCH --job-name=convert_tfrecords
#SBATCH --partition=gpu
#SBATCH --nodelist=node030
#SBATCH --gpus=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_tfrecords_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_tfrecords_%j.err

echo "=== TFRecord Conversion ==="
echo "Started at: $(date)"
echo "CPUs: $(nproc)"

# Run conversion inside container where TensorFlow is installed
apptainer exec \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/data:/data \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  bash -c "
    cd /jaeger
    pip install -e . --quiet 2>/dev/null
    
    echo 'Converting training data...'
    python3 /jaeger/convert_to_tfrecord.py \
      --csv /data/train_shuffled.csv \
      --output /data/train_shuffled.tfrecord \
      --crop-size 500
    
    echo 'Converting validation data...'
    python3 /jaeger/convert_to_tfrecord.py \
      --csv /data/val_shuffled.csv \
      --output /data/val_shuffled.tfrecord \
      --crop-size 500
  "

echo "Finished at: $(date)"
ls -lh /mnt/beegfs/bioinf/wijesekara/jaeger/data/*.tfrecord
