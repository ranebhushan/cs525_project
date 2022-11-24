#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p long
#SBATCH -t 167:59:59
#SBATCH -o rljob_%j.out
#SBATCH --error rljob_%j.err
#SBATCH -J rl_project
#SBATCH --mail-user=yrpatil@wpi.edu
#SBATCH --mail-type=ALL

echo "RL Job running on $(hostname)"

echo "Loading Python Virtual Environment"

source ~/RL_F22/cs525_project/rl_project/bin/activate

module load python/3.9.12/uabo2y2
module load cuda11.7/toolkit/11.7.1
module load cudnn8.5-cuda11.7/8.5.0.96

echo "Running Python Code"

python3 src/main.py --train_config_path=configs/DQN.yaml --env_config_path=configs/highway-env_config.json
