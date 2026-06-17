#!/bin/bash
#SBATCH --job-name=jaeger_eval_benchmark
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/eval_benchmark_%j.out
#SBATCH --error=/mnt/beegfs/bioinf/wijesekara/jaeger/slurm/logs/eval_benchmark_%j.err

set -euo pipefail

PROJECT=/mnt/beegfs/bioinf/wijesekara/jaeger
CONTAINER=$PROJECT/container/jaeger_dev.sif
SRC=$PROJECT/Jaeger

export PYTHONPATH="$SRC/src:${PYTHONPATH:-}"

apptainer exec \
    -B "$PROJECT":"$PROJECT" \
    -B "$SRC":"/jaeger_src" \
    "$CONTAINER" \
    bash -c "
        cd /jaeger_src
        python scripts/benchmark_report.py \
            --experiments-root $PROJECT/experiments \
            --output-md $PROJECT/benchmark_report.md \
            --output-csv $PROJECT/benchmark_report.csv
        python scripts/run_benchmark_evaluation.py \
            --experiments-root $PROJECT/experiments \
            --data-root $PROJECT/data \
            --output-csv $PROJECT/evaluation_metrics.csv \
            --batch-size 128
        python scripts/plot_benchmark_results.py \
            --report-csv $PROJECT/benchmark_report.csv \
            --metrics-csv $PROJECT/evaluation_metrics.csv \
            --output-dir $PROJECT/benchmark_report/figures
    "

echo "Benchmark reports written to $PROJECT/benchmark_report.md and $PROJECT/evaluation_metrics.csv"
