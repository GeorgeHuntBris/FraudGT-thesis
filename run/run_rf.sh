#!/bin/bash
#SBATCH --job-name=RF-Baseline
#SBATCH --partition=cpu
#SBATCH --account=COMS037985
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"

source ~/fraudgt-env/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

python run_rf.py

echo "Job finished: $(date)"
