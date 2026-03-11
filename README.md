# Evaluating the dCGF Metabolomic Model: Does It Learn Pathway Biology or Just Species Identity?

This repository contains an independent evaluation of the **dCGF (data-driven Community Genotype-Function)** model from [Qian et al. (2025)](https://doi.org/10.1101/2025.01.04.631316), which predicts metabolite production by synthetic microbial communities from their genomic pathway composition.

The dCGF model takes as input a matrix of species abundances and their metabolic pathway presence/absence vectors, then predicts community-level metabolite concentrations (acetate, butyrate, lactate, succinate). The original paper frames this as a **genotype-to-function mapping** -- learning how metabolic pathways drive community metabolic output.

## Key Question

**Does the model actually learn pathway biology, or does it primarily learn which species are present?**

Because each species has a nearly unique combination of metabolic pathways, the pathway vectors effectively act as species fingerprints. A model could achieve high accuracy simply by learning "species X produces Y amount of butyrate" without understanding anything about metabolic pathways.

## What We Found

We ran a series of controls to disentangle species identity from genuine pathway biology:

| Model | Acetate | Butyrate | Lactate | Succinate |
|---|---|---|---|---|
| **dCGF-IS (real pathways)** | 0.785 | 0.919 | 0.866 | 0.922 |
| RF (species + pathways) | 0.677 | 0.905 | 0.840 | 0.877 |
| dCGF-IS (random pathways) | 0.702 | 0.892 | 0.854 | 0.874 |
| RF (species only) | 0.537 | 0.893 | 0.824 | 0.809 |
| dCGF-IS (species only) | 0.559 | 0.882 | 0.821 | 0.802 |
| Ridge (species only) | 0.591 | 0.867 | 0.772 | 0.692 |
| Shuffled targets (null) | ~0.00 | ~-0.03 | ~-0.03 | ~-0.04 |

*All values are Pearson r from 8-fold cross-validation averaged over 5 random seeds.*

**Key findings:**

1. **Species identity captures most of the signal.** For butyrate and lactate, a one-hot species model achieves >94% of the full model's performance. The deep learning architecture with pathway information barely improves over knowing who is present.

2. **Random pathways work almost as well as real ones.** Replacing KEGG pathways with random binary vectors (80% shared, 20% random per species) gives performance between species-only and real pathways. This suggests the extra dimensions help the neural network, but real biology adds only modestly.

3. **Pathway similarity does not explain performance.** Adding pairwise phylogenetic/pathway similarity features to Random Forest does not improve over species-only RF, ruling out the hypothesis that pathways encode useful genomic similarity.

4. **The sensitivity analysis (paper's Figure 4) is confounded.** Pathways unique to a single species dominate the sensitivity rankings because silencing a rare pathway is equivalent to removing a species. Biologically relevant pathways (glycolysis, TCA cycle, butanoate metabolism) are shared by nearly all species and therefore show negligible sensitivity.

5. **Real pathways do add something for acetate and succinate.** The gap between real pathways (0.785/0.922) and species-only (0.559/0.802) is substantial for these metabolites, and real pathways beat random pathways, indicating some genuine pathway biology beyond species identity.

The full analysis is in [`dcgf_results_report_v1.1.pdf`](dcgf_results_report_v1.1.pdf).

## Repository Structure

### Data
- `MasterDF.csv` -- Clark et al. (2021) community composition and metabolite measurements (1850 samples, 25 species)
- `genetic_features.csv` -- Binary KEGG pathway matrix (25 species x 144 pathways)
- `kegg_pathways_cache.json` -- Cached KEGG API responses
- `paper_fig6_coordinates.json` -- Manually scraped PCA coordinates from paper's Figure 6

### Core Model Code
- `dcgf_model.py` -- PyTorch implementations of dCGF-ES and dCGF-IS (Deep Set architecture)
- `data_processing.py` -- Data loading, community filtering, replicate averaging, GF matrix construction
- `metadata.py` -- Species metadata, abbreviations, community definitions from Clark et al.

### Training Scripts
- `train_dcgf.py` -- Baseline dCGF-ES and dCGF-IS training with k-fold CV
- `train_dcgf_controls.py` -- Control experiments: actual data, random abundances, shuffled targets (permutation null)
- `train_species_only.py` -- One-hot species identity model (no pathway information)
- `train_random_pathways.py` -- Random binary pathway control
- `train_simple_baselines.py` -- Ridge regression and Random Forest baselines

### Analysis Scripts
- `interpret_dcgf.py` -- Pathway importance via permutation importance and sensitivity analysis
- `generate_predictions.py` -- Save CV predictions for report figures
- `generate_embeddings.py` -- Extract species functional embeddings for PCA analysis
- `generate_report.py` -- Generate PDF report with all figures and tables
- `fetch_kegg_pathways.py` -- Fetch KEGG pathway data and build genetic_features.csv

### SLURM Job Scripts
- `run_dcgf.sh` -- Submit baseline training
- `run_controls_parallel.sh` -- Submit 48 parallel control jobs (4 metabolites x {actual, random, 10 shuffled})
- `run_interpret.sh` -- Submit interpretation jobs (4 metabolites)
- `run_baselines.sh` -- Submit Ridge + Random Forest baselines
- `run_embeddings.sh` -- Submit embedding generation
- `run_report.sh` -- Submit PDF report generation

### Output
- `dcgf_results_report_v1.1.pdf` -- Full PDF report with figures, tables, and discussion

## Dependencies

```
numpy
torch
scipy
scikit-learn
pandas
matplotlib
```

## Citation

This evaluation analyzes the model from:

> Qian Y, et al. (2025). A data-driven modeling framework for mapping genotypes to synthetic microbial community functions. *bioRxiv* 2025.01.04.631316.

The experimental data comes from:

> Clark RL, et al. (2021). Design of synthetic human gut microbiome assembly and butyrate production. *Nature Communications* 12, 3254.

---

# Instructions for Claude Code Agent to Reproduce All Results

The following instructions allow a Claude Code agent to reproduce the complete analysis from scratch, starting from the data files in this repository. All compute-intensive jobs must be submitted via SLURM on the Orion partition.

## Prerequisites

- Access to a SLURM cluster with an `Orion` partition
- Python with numpy, torch, scipy, scikit-learn, pandas, matplotlib
- All scripts assume they run from the repository root directory
- Always use `python -u` and `flush=True` for real-time output on the cluster

## Step-by-Step Reproduction

### Step 1: Fetch KEGG Pathway Data (optional -- already cached)

The file `kegg_pathways_cache.json` and `genetic_features.csv` are already included. To regenerate from the KEGG API:

```bash
python -u fetch_kegg_pathways.py
```

This queries the KEGG REST API for all 25 species and produces `genetic_features.csv` (25 species x 144 KEGG pathways, binary presence/absence).

**Note:** PC (Prevotella copri) and HB (Holdemanella biformis) are not in KEGG and will have zero pathway vectors. Communities containing HB are excluded from all analyses.

### Step 2: Train Baseline dCGF Models

Submit via SLURM:

```bash
sbatch run_dcgf.sh
```

Or equivalently:
```bash
sbatch --job-name=dcgf_train --partition=Orion --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --time=24:00:00 \
  --wrap="cd $(pwd) && python -u train_dcgf.py"
```

This trains both dCGF-ES and dCGF-IS for all 4 metabolites using 8-fold CV with 5 random seeds and 500 epochs. Expected results: dCGF-IS Pearson r of ~0.785 (Acetate), ~0.919 (Butyrate), ~0.866 (Lactate), ~0.922 (Succinate).

### Step 3: Run Control Experiments

Submit 48 parallel jobs (4 metabolites x {actual, random abundance, 10 shuffled permutations}):

```bash
# Edit run_controls_parallel.sh to set DIR to your working directory, then:
bash run_controls_parallel.sh
```

This runs `train_dcgf_controls.py` with `--condition actual|random|shuffled` and evaluates at 500, 1000, and 2000 epochs. The shuffled-target permutation test establishes the null distribution (~0.00 correlation).

### Step 4: Train Species-Only Model

```bash
sbatch --job-name=sp_only --partition=Orion --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --time=24:00:00 \
  --wrap="cd $(pwd) && python -u train_species_only.py"
```

Replaces pathway vectors with one-hot species identity (24-dimensional). Tests whether pathway information adds predictive value beyond species identity.

### Step 5: Train Random Pathway Control

```bash
sbatch --job-name=rand_pw --partition=Orion --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --time=24:00:00 \
  --wrap="cd $(pwd) && python -u train_random_pathways.py"
```

Replaces real KEGG pathways with 144-dimensional random binary vectors (80% shared across all species, 20% random per species). Saves results to `random_pathway_results.json`.

### Step 6: Train Simple Baselines (Ridge + Random Forest)

```bash
sbatch run_baselines.sh
```

Runs Ridge regression and Random Forest on species presence/absence, species+pathways, and species+pairwise pathway similarity. Saves results to `simple_baseline_results.json`. This is fast (~minutes).

### Step 7: Run Pathway Importance Analysis

```bash
bash run_interpret.sh
```

Submits 4 parallel jobs (one per metabolite). Each trains an ensemble of 5 models on all data, then runs:
- **Permutation importance**: shuffle each pathway, measure drop in Pearson r
- **Sensitivity analysis**: silence/activate each pathway in each species, measure prediction change

Outputs: `pathway_importance_{Acetate,Butyrate,Lactate,Succinate}.csv`

### Step 8: Generate Cross-Validation Predictions

```bash
sbatch --job-name=gen_preds --partition=Orion --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=8G --time=24:00:00 \
  --wrap="cd $(pwd) && python -u generate_predictions.py"
```

Saves measured vs predicted values to `cv_predictions.npz` for use in the report figures.

### Step 9: Generate Species Embeddings for PCA

```bash
sbatch run_embeddings.sh
```

Trains dCGF-IS on all data for each metabolite, then extracts 30-dimensional per-species embeddings. Saves to `species_embeddings.npz`. This is slow (~2 hours).

### Step 10: Generate PDF Report

After all previous steps are complete:

```bash
sbatch run_report.sh
```

Reads `cv_predictions.npz`, `species_embeddings.npz`, `simple_baseline_results.json`, `random_pathway_results.json`, `pathway_importance_*.csv`, and `paper_fig6_coordinates.json` to generate `dcgf_results_report_v1.1.pdf`.

### Expected Run Order and Dependencies

```
Step 1 (optional) ──> genetic_features.csv
Step 2 ──────────────> baseline r values (stdout)
Step 3 ──────────────> control r values (stdout)
Step 4 ──────────────> species-only r values (stdout)
Step 5 ──────────────> random_pathway_results.json
Step 6 ──────────────> simple_baseline_results.json
Step 7 ──────────────> pathway_importance_*.csv
Step 8 ──────────────> cv_predictions.npz
Step 9 ──────────────> species_embeddings.npz
Step 10 ─────────────> dcgf_results_report_v1.1.pdf (requires outputs from steps 5-9)
```

Steps 2-9 can run in parallel. Step 10 depends on the outputs of steps 5-9.

### SLURM Notes

- All jobs use partition `Orion`
- Set walltime to at least 24 hours (no cost to over-requesting)
- Training jobs (steps 2-5, 7-9) typically take 1-4 hours each
- Baselines (step 6) and report generation (step 10) take <30 minutes
- The `run_controls_parallel.sh` script submits 48 independent jobs; adjust the `DIR` variable to match your working directory
