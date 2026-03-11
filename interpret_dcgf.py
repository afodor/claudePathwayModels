"""
Pathway importance analysis for dCGF-IS using two methods:

Method 1: Permutation importance (model-agnostic)
  - Train ensemble of models on all data
  - For each pathway, shuffle that pathway across all species in all communities
  - Measure drop in prediction Pearson r
  - Bigger drop = more important pathway

Method 2: Sensitivity analysis (paper's Figure 4 method)
  - Train ensemble of models on all data
  - For each pathway in each species, silence it (if present) or activate it (if absent)
  - Measure change in predicted output
  - Average across species and models

Both methods train an ensemble of models (different random seeds) and average results.

Usage:
  python -u interpret_dcgf.py --metabolite Butyrate
  python -u interpret_dcgf.py  # runs all metabolites
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
import os
import argparse

from dcgf_model import dCGF_IS_Batched
from data_processing import prepare_dataset, METABOLITES


def train_model(model, train_matrices, train_targets, n_epochs=500, lr=1e-3,
                weight_decay=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(train_matrices)
        loss = loss_fn(predictions, train_targets)
        loss.backward()
        optimizer.step()


def predict(model, matrices):
    model.eval()
    with torch.no_grad():
        return model(matrices).numpy()


def train_ensemble(gf_matrices, targets, n_gf, n_models=5, n_epochs=500,
                   **model_kwargs):
    """Train an ensemble of dCGF-IS models on all data."""
    torch_mats = [torch.tensor(m) for m in gf_matrices]
    torch_tgt = torch.tensor(targets)
    models = []
    for i in range(n_models):
        seed = i + 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = dCGF_IS_Batched(n_gf, **model_kwargs)
        train_model(model, torch_mats, torch_tgt, n_epochs=n_epochs)
        models.append(model)
        preds = predict(model, torch_mats)
        r, _ = pearsonr(preds, targets)
        print(f"  Ensemble model {i+1}/{n_models}: train r = {r:.4f}", flush=True)
    return models


def ensemble_predict(models, matrices):
    """Average predictions across ensemble."""
    all_preds = np.array([predict(m, matrices) for m in models])
    return all_preds.mean(axis=0)


# ============================================================
# Method 1: Permutation importance
# ============================================================
def permutation_importance(models, gf_matrices, targets, pathway_names,
                           n_repeats=10):
    """
    For each pathway, shuffle it across communities and measure drop in r.
    """
    torch_mats = [torch.tensor(m) for m in gf_matrices]
    base_preds = ensemble_predict(models, torch_mats)
    base_r, _ = pearsonr(base_preds, targets)
    print(f"\n  Baseline ensemble r = {base_r:.4f}", flush=True)

    n_pathways = len(pathway_names)
    importance = np.zeros(n_pathways)

    for p_idx in range(n_pathways):
        drops = []
        for rep in range(n_repeats):
            rng = np.random.RandomState(rep * 1000 + p_idx)
            # For each community, randomly reassign this pathway's values
            # across its species (breaks the pathway-species association)
            perturbed = []
            for mat in gf_matrices:
                new_mat = mat.copy()
                rng.shuffle(new_mat[p_idx, :])
                perturbed.append(new_mat)

            torch_perturbed = [torch.tensor(m) for m in perturbed]
            perm_preds = ensemble_predict(models, torch_perturbed)
            perm_r, _ = pearsonr(perm_preds, targets)
            drops.append(base_r - perm_r)

        importance[p_idx] = np.mean(drops)
        if (p_idx + 1) % 20 == 0:
            print(f"    Permutation importance: {p_idx+1}/{n_pathways} pathways done",
                  flush=True)

    return importance, base_r


# ============================================================
# Method 2: Sensitivity analysis (paper's method)
# ============================================================
def sensitivity_analysis(models, gf_matrices, targets, pathway_names,
                         gf_vectors, species_list):
    """
    Paper's method: for each pathway in each species, silence/activate it
    and measure prediction change.

    sensitivity[pathway] = average |F(u + delta) - F(u)| across species, models, communities
    """
    n_pathways = len(pathway_names)
    n_communities = len(gf_matrices)

    torch_mats = [torch.tensor(m) for m in gf_matrices]
    base_preds = ensemble_predict(models, torch_mats)

    # Sensitivity per pathway (averaged across species)
    sensitivity = np.zeros(n_pathways)
    # Also track per-pathway direction (positive = enhances, negative = inhibits)
    sensitivity_signed = np.zeros(n_pathways)

    for p_idx in range(n_pathways):
        pathway_sens = []

        for comm_idx, mat in enumerate(gf_matrices):
            n_gf, n_sp = mat.shape

            for sp_idx in range(n_sp):
                # Get the abundance for this species in this community
                # (recover from the GF matrix: abundance = mat[any_present_feature, sp] / 1.0)
                abundance = 1.0 / n_sp  # equal abundance assumption

                # Check if pathway is present in this species
                original_val = mat[p_idx, sp_idx]
                is_present = original_val > 0

                # Create perturbation
                perturbed_mat = mat.copy()
                if is_present:
                    # Silence: set to 0
                    perturbed_mat[p_idx, sp_idx] = 0.0
                    delta = -abundance
                else:
                    # Activate: set to abundance
                    perturbed_mat[p_idx, sp_idx] = abundance
                    delta = abundance

                # Predict with perturbation (just this one community)
                torch_perturbed = [torch.tensor(perturbed_mat)]
                perturbed_pred = ensemble_predict(models, torch_perturbed)[0]
                original_pred = base_preds[comm_idx]

                # Sensitivity = (F(u+delta) - F(u)) / delta
                if abs(delta) > 0:
                    sens = (perturbed_pred - original_pred) / delta
                    pathway_sens.append(sens)

        if pathway_sens:
            sensitivity[p_idx] = np.mean(np.abs(pathway_sens))
            sensitivity_signed[p_idx] = np.mean(pathway_sens)

        if (p_idx + 1) % 20 == 0:
            print(f"    Sensitivity analysis: {p_idx+1}/{n_pathways} pathways done",
                  flush=True)

    return sensitivity, sensitivity_signed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metabolite', type=str, default=None)
    args = parser.parse_args()

    metabolites = [args.metabolite] if args.metabolite else METABOLITES
    data_dir = os.path.dirname(os.path.abspath(__file__))

    model_kwargs = {
        'z_dim': 30,
        'encoder_hidden': 200,
        'decoder_hidden': 100,
    }
    n_models = 5
    n_epochs = 500
    top_n = 15  # show top N pathways

    for metabolite in metabolites:
        print(f"\n{'='*70}", flush=True)
        print(f"Pathway Importance: {metabolite}", flush=True)
        print(f"{'='*70}", flush=True)

        gf_matrices, targets, tgt_mean, tgt_std, gf_vecs, pathway_names = \
            prepare_dataset(data_dir, metabolite)
        n_gf = len(pathway_names)
        species_list = list(gf_vecs.keys())

        # Train ensemble on all data
        print(f"\nTraining ensemble of {n_models} models...", flush=True)
        models = train_ensemble(gf_matrices, targets, n_gf, n_models=n_models,
                                n_epochs=n_epochs, **model_kwargs)

        # Method 1: Permutation importance
        print(f"\n--- Method 1: Permutation Importance ---", flush=True)
        perm_imp, base_r = permutation_importance(
            models, gf_matrices, targets, pathway_names, n_repeats=10
        )

        # Sort by importance
        perm_order = np.argsort(perm_imp)[::-1]
        print(f"\n  Top {top_n} pathways by permutation importance (drop in r):", flush=True)
        print(f"  {'Rank':<6} {'Pathway':<50} {'Importance':>12}", flush=True)
        print(f"  {'-'*6} {'-'*50} {'-'*12}", flush=True)
        for rank, idx in enumerate(perm_order[:top_n]):
            print(f"  {rank+1:<6} {pathway_names[idx]:<50} {perm_imp[idx]:>12.6f}", flush=True)

        # Method 2: Sensitivity analysis
        print(f"\n--- Method 2: Sensitivity Analysis (Paper Method) ---", flush=True)
        sens_abs, sens_signed = sensitivity_analysis(
            models, gf_matrices, targets, pathway_names, gf_vecs, species_list
        )

        # Sort by absolute sensitivity
        sens_order = np.argsort(sens_abs)[::-1]
        print(f"\n  Top {top_n} pathways by sensitivity (|dF/d_pathway|):", flush=True)
        print(f"  {'Rank':<6} {'Pathway':<50} {'|Sensitivity|':>14} {'Direction':>10}", flush=True)
        print(f"  {'-'*6} {'-'*50} {'-'*14} {'-'*10}", flush=True)
        for rank, idx in enumerate(sens_order[:top_n]):
            direction = "+" if sens_signed[idx] > 0 else "-"
            print(f"  {rank+1:<6} {pathway_names[idx]:<50} {sens_abs[idx]:>14.6f} "
                  f"{direction:>10}", flush=True)

        # Compare rankings
        print(f"\n--- Comparison: Method 1 vs Method 2 ---", flush=True)
        print(f"  {'Perm Rank':<12} {'Sens Rank':<12} {'Pathway':<50}", flush=True)
        print(f"  {'-'*12} {'-'*12} {'-'*50}", flush=True)

        # Build rank lookup for sensitivity
        sens_rank_lookup = {idx: rank for rank, idx in enumerate(sens_order)}
        for perm_rank, p_idx in enumerate(perm_order[:top_n]):
            s_rank = sens_rank_lookup[p_idx]
            marker = " <-- AGREE" if s_rank < top_n else ""
            print(f"  {perm_rank+1:<12} {s_rank+1:<12} {pathway_names[p_idx]:<50}{marker}",
                  flush=True)

        # Save results to CSV
        out_file = os.path.join(data_dir, f"pathway_importance_{metabolite}.csv")
        with open(out_file, 'w') as f:
            f.write("pathway,perm_importance,sensitivity_abs,sensitivity_signed\n")
            for idx in range(n_gf):
                f.write(f"{pathway_names[idx]},{perm_imp[idx]:.8f},"
                        f"{sens_abs[idx]:.8f},{sens_signed[idx]:.8f}\n")
        print(f"\n  Results saved to {out_file}", flush=True)


if __name__ == '__main__':
    main()
