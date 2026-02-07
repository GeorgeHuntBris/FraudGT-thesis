#!/bin/bash

# ============================================================
# SLURM RESOURCE REQUESTS
# These tell the scheduler what your job needs.
# ============================================================

# Job name — shows up in squeue so you can tell jobs apart
#SBATCH --job-name=ETH-MLP

# Partition — the pool of nodes to use. Change this to match
# BluePebble's GPU partition name (check with: sinfo)
#SBATCH --partition=gpu

# Request 1 GPU
#SBATCH --gres=gpu:1

# Request 8 CPU cores. This should be >= num_workers + 2
# (4 workers + main process + PyTorch threads)
#SBATCH --cpus-per-task=8

# Memory — 24GB is plenty for ETH dataset
#SBATCH --mem=24G

# Time limit — 12 hours. If the job exceeds this, Slurm kills it.
# 500 epochs on ETH should finish well within this.
#SBATCH --time=12:00:00

# Where stdout and stderr go. %x = job name, %j = job ID
# Creates e.g. logs/ETH-MLP_12345.out
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# ENVIRONMENT SETUP
# Runs on the compute node before your training starts.
# ============================================================

# Print job info for debugging
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Load any required modules. Uncomment and adjust as needed:
# module load CUDA/11.8
# module load Python/3.9

# Activate your conda/virtualenv environment.
# CHANGE THIS to match your environment name on BluePebble:
source activate fraudgt
# Or if using virtualenv:
# source ~/envs/fraudgt/bin/activate

# Move to the repo root (where you submitted from)
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================
# RUN TRAINING
# ============================================================

python -m fraudGT.main \
    --cfg configs/ETH/ETH-MLP.yaml \
    --gpu 0 \
    --repeat 3 \
    num_threads 6 num_workers 4

echo "Job finished: $(date)"
