#!/bin/bash
#SBATCH --job-name=dcgf_ctrl
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=dcgf_controls_%j.out
#SBATCH --error=dcgf_controls_%j.err

cd "$(dirname "$0")"
python -u train_dcgf_controls.py
