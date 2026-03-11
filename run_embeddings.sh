#!/bin/bash
#SBATCH --job-name=dcgf_embeddings
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=embeddings_%j.out
#SBATCH --error=embeddings_%j.err

cd "$(dirname "$0")"
python -u generate_embeddings.py
