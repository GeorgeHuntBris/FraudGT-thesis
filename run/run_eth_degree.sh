#!/bin/bash
#SBATCH --job-name=ETH-Degree
#SBATCH --partition=compute
#SBATCH --account=COMS037985
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started: $(date)"

source ~/fraudgt-env/bin/activate

cd ~/FraudGT-thesis
mkdir -p logs

python3 compute_eth_degree_dist.py

echo "Job finished: $(date)"
