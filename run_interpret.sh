#!/bin/bash
# Submit 4 parallel interpretation jobs, one per metabolite

DIR="$(cd "$(dirname "$0")" && pwd)"

for METAB in Acetate Butyrate Lactate Succinate; do
    sbatch --job-name="interp_${METAB}" \
           --partition=Orion \
           --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G \
           --time=24:00:00 \
           --output="${DIR}/interp_${METAB}_%j.out" \
           --error="${DIR}/interp_${METAB}_%j.err" \
           --wrap="cd ${DIR} && python -u interpret_dcgf.py --metabolite ${METAB}"
done
