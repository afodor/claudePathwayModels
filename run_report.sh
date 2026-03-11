#!/bin/bash
#SBATCH --job-name=dcgf_report
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=report_%j.out
#SBATCH --error=report_%j.err

cd "$(dirname "$0")"
python -u generate_report.py
