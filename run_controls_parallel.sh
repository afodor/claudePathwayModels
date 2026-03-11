#!/bin/bash
# Submit parallel SLURM jobs:
#   4 metabolites x 1 actual = 4 jobs
#   4 metabolites x 1 random = 4 jobs
#   4 metabolites x 10 permutations = 40 jobs
#   Total: 48 jobs

DIR="$(cd "$(dirname "$0")" && pwd)"

for METAB in Acetate Butyrate Lactate Succinate; do
    # Actual
    sbatch --job-name="${METAB}_actual" \
           --partition=Orion \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
           --time=24:00:00 \
           --output="${DIR}/ctrl_${METAB}_actual_%j.out" \
           --error="${DIR}/ctrl_${METAB}_actual_%j.err" \
           --wrap="cd ${DIR} && python -u train_dcgf_controls.py --metabolite ${METAB} --condition actual"

    # Random abundances
    sbatch --job-name="${METAB}_random" \
           --partition=Orion \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
           --time=24:00:00 \
           --output="${DIR}/ctrl_${METAB}_random_%j.out" \
           --error="${DIR}/ctrl_${METAB}_random_%j.err" \
           --wrap="cd ${DIR} && python -u train_dcgf_controls.py --metabolite ${METAB} --condition random"

    # 10 individual shuffled permutations
    for PERM in $(seq 0 9); do
        sbatch --job-name="${METAB}_shuf${PERM}" \
               --partition=Orion \
               --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
               --time=24:00:00 \
               --output="${DIR}/ctrl_${METAB}_shuf${PERM}_%j.out" \
               --error="${DIR}/ctrl_${METAB}_shuf${PERM}_%j.err" \
               --wrap="cd ${DIR} && python -u train_dcgf_controls.py --metabolite ${METAB} --condition shuffled --perm ${PERM}"
    done
done
