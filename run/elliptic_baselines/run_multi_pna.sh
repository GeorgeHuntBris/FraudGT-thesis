#!/bin/bash
#SBATCH --job-name=Ell-MultiPNA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# module load CUDA/11.8
source activate fraudgt

cd $SLURM_SUBMIT_DIR
mkdir -p logs

python -m fraudGT.main \
    --cfg configs/Elliptic/Elliptic-Multi-PNA.yaml \
    --gpu 0 \
    --repeat 5 \
    num_threads 6 num_workers 4

echo "Job finished: $(date)"
