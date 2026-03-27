#!/bin/bash
#SBATCH --job-name=ETH-Aux-Multi
#SBATCH --partition=gpu
#SBATCH --account=COMS037985
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

source ~/fraudgt-env/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

python -m fraudGT.main \
    --cfg configs/ETH-Aux/ETH-Aux-Multi-SparseNodeGT.yaml \
    --gpu 0 \
    --repeat 5 \
    num_threads 6 num_workers 4

echo "Job finished: $(date)"
