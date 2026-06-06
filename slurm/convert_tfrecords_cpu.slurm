#!/bin/bash
#SBATCH --job-name=convert_tfrecords_cpu
#SBATCH --partition=gpu
#SBATCH --nodelist=node030
#SBATCH --gpus=0
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_cpu_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_cpu_%j.err

echo "=== CPU TFRecord Conversion ==="
echo "Started at: $(date)"
echo "CPUs: $(nproc)"

# Run without --nv flag (CPU only)
apptainer exec \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/Jaeger_current:/jaeger \
  --bind /mnt/beegfs/bioinf/wijesekara/jaeger/data:/data \
  /mnt/beegfs/bioinf/wijesekara/jaeger/jaeger_dev.sif \
  bash -c "
    cd /jaeger
    pip install -e . --quiet 2>/dev/null
    
    echo 'Converting training data...'
    python3 /jaeger/convert_to_tfrecord_fast.py \
      --csv /data/train_shuffled.csv \
      --output /data/train_shuffled.tfrecord \
      --crop-size 500
    
    echo 'Converting validation data...'
    python3 /jaeger/convert_to_tfrecord_fast.py \
      --csv /data/val_shuffled.csv \
      --output /data/val_shuffled.tfrecord \
      --crop-size 500
  "

echo "Finished at: $(date)"
ls -lh /data/*.tfrecord
