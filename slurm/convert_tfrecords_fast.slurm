#!/bin/bash
#SBATCH --job-name=convert_tfrecords_fast
#SBATCH --partition=gpu
#SBATCH --nodelist=node030
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_fast_%j.log
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/convert_fast_%j.err

echo "=== Fast TFRecord Conversion ==="
echo "Started at: $(date)"

apptainer exec --nv \
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
