#!/bin/bash
#SBATCH --job-name=dcgf_train
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=dcgf_train_%j.out
#SBATCH --error=dcgf_train_%j.err

cd "$(dirname "$0")"
python -u train_dcgf.py
