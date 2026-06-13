#!/bin/bash
#SBATCH --job-name=submit_axial
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/logs/submit_axial_%j.log

# Wait until at most 1 jaeger job is running
while [ $(squeue -u wijesekara -n jaeger_500bp_baseline,jaeger_500bp_crossframe -h -t R | wc -l) -ge 2 ]; do
    echo "Waiting for a GPU to free up..."
    sleep 60
done

echo "Submitting axial job..."
cd /mnt/beegfs/bioinf/wijesekara/jaeger/slurm
sbatch axial_500bp.slurm
