#!/bin/bash
#SBATCH --job-name=baselines_sim
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=baselines_%j.out
#SBATCH --error=baselines_%j.err

cd "$(dirname "$0")"
python -u train_simple_baselines.py
