#!/bin/bash

#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -p long
#SBATCH -t 167:59:59
#SBATCH -o slurm_outputs/rljob_%j.out
#SBATCH --error slurm_outputs/rljob_%j.err
#SBATCH -J DDDQN

echo "Running python code on $(hostname) with algorithm $1"

python3 src/main.py --agent_config_path=configs/$1.yaml --env_config_path=configs/highway-env_config.json