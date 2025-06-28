#!/bin/bash
#SBATCH --cpus-per-task=18
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -t 24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

# load modules
module load 2024
module load Python/3.12.3-GCCcore-13.3.0


python main.py --k_folds 5 --model dhooge --dataset dadagp_full_aug --history_len 1 --run_num 46
