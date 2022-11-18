#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p short
#SBATCH -t 20:00:00
#SBATCH --output=slurm_outputs/rljob_%j.out
#SBATCH --error=slurm_outputs/rljob_error_%j.out
#SBATCH -J rl_project

echo "RL Job running on $(hostname)"

python3 src/main.py --yaml_path=configs/DQN.yaml