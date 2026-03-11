"""
Training and evaluation of dCGF models with k-fold cross-validation.

Reproduces the evaluation procedure from Qian et al. (2025):
  - k-fold cross-validation with 5 random seeds
  - Reports Pearson correlation between predicted and measured functions
  - Compares dCGF-ES and dCGF-IS performance
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import sys
import os

from dcgf_model import dCGF_ES, dCGF_IS_Batched
from data_processing import prepare_dataset, METABOLITES


def train_model(model, train_matrices, train_targets, n_epochs=500, lr=1e-3,
                weight_decay=1e-4, verbose=False):
    """
    Train a dCGF model.

    Args:
        model: dCGF_ES or dCGF_IS_Batched instance
        train_matrices: list of torch tensors (GF matrices)
        train_targets: torch tensor of target values
        n_epochs: number of training epochs
        lr: learning rate
        weight_decay: L2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(train_matrices)
        loss = loss_fn(predictions, train_targets)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")


def evaluate_model(model, test_matrices, test_targets):
    """
    Evaluate model predictions using Pearson correlation.

    Returns:
        pearson_r: Pearson correlation coefficient
        predictions: numpy array of predictions
    """
    model.eval()
    with torch.no_grad():
        predictions = model(test_matrices).numpy()
    targets = test_targets.numpy()

    if len(predictions) < 3:
        return 0.0, predictions

    r, p = pearsonr(predictions, targets)
    return r, predictions


def kfold_cv(model_class, gf_matrices, targets, n_gf, k=8, n_seeds=5,
             n_epochs=500, lr=1e-3, weight_decay=1e-4, verbose=True,
             **model_kwargs):
    """
    Perform k-fold cross-validation with multiple random seeds.

    Following the paper's procedure:
      - Split data into k folds
      - Train on (k-1) folds, test on remaining fold
      - Repeat with different random seeds
      - Report averaged predictions and correlations

    Args:
        model_class: dCGF_ES or dCGF_IS_Batched
        gf_matrices: list of numpy arrays
        targets: numpy array (standardized)
        n_gf: number of genetic features
        k: number of folds
        n_seeds: number of random seeds
    """
    n_samples = len(gf_matrices)
    all_predictions = np.zeros((n_seeds, n_samples))
    seed_correlations = []

    for seed_idx in range(n_seeds):
        seed = seed_idx + 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        predictions_this_seed = np.full(n_samples, np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gf_matrices)):
            # Prepare fold data
            train_mats = [torch.tensor(gf_matrices[i]) for i in train_idx]
            test_mats = [torch.tensor(gf_matrices[i]) for i in test_idx]
            train_tgt = torch.tensor(targets[train_idx])
            test_tgt = torch.tensor(targets[test_idx])

            # Create and train model
            model = model_class(n_gf, **model_kwargs)
            train_model(model, train_mats, train_tgt, n_epochs=n_epochs,
                       lr=lr, weight_decay=weight_decay)

            # Evaluate
            r, preds = evaluate_model(model, test_mats, test_tgt)
            predictions_this_seed[test_idx] = preds

            if verbose:
                print(f"    Fold {fold_idx+1}/{k}: r = {r:.4f}", flush=True)

        all_predictions[seed_idx] = predictions_this_seed

        # Correlation for this seed
        r_seed, _ = pearsonr(predictions_this_seed, targets)
        seed_correlations.append(r_seed)

        if verbose:
            print(f"  Seed {seed_idx+1}/{n_seeds}: Pearson r = {r_seed:.4f}", flush=True)

    # Average predictions across seeds
    avg_predictions = all_predictions.mean(axis=0)
    avg_r, _ = pearsonr(avg_predictions, targets)

    return {
        'avg_pearson_r': avg_r,
        'seed_correlations': seed_correlations,
        'mean_correlation': np.mean(seed_correlations),
        'std_correlation': np.std(seed_correlations),
        'avg_predictions': avg_predictions,
        'all_predictions': all_predictions,
    }


def main():
    print("=" * 70, flush=True)
    print("dCGF Model Training and Evaluation", flush=True)
    print("Reproducing k-fold cross-validation from Qian et al. (2025)", flush=True)
    print("=" * 70, flush=True)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Hyperparameters from the paper
    model_kwargs = {
        'z_dim': 30,
        'encoder_hidden': 200,
        'decoder_hidden': 100,
    }
    train_kwargs = {
        'n_epochs': 500,
        'lr': 1e-3,
        'weight_decay': 1e-4,
    }
    k_folds = 8
    n_seeds = 5

    results = {}

    for metabolite in METABOLITES:
        print(f"\n{'='*70}", flush=True)
        print(f"Metabolite: {metabolite}", flush=True)
        print(f"{'='*70}", flush=True)

        gf_matrices, targets, tgt_mean, tgt_std, gf_vecs, pathway_names = \
            prepare_dataset(data_dir, metabolite)

        n_gf = len(pathway_names)

        # --- dCGF-ES ---
        print(f"\n--- dCGF-ES (Enzyme Soup) ---", flush=True)
        es_results = kfold_cv(
            dCGF_ES, gf_matrices, targets, n_gf,
            k=k_folds, n_seeds=n_seeds, **train_kwargs, **model_kwargs
        )
        print(f"  Average Pearson r: {es_results['avg_pearson_r']:.4f}", flush=True)
        print(f"  Mean +/- Std across seeds: "
              f"{es_results['mean_correlation']:.4f} +/- "
              f"{es_results['std_correlation']:.4f}", flush=True)

        # --- dCGF-IS ---
        print(f"\n--- dCGF-IS (Individual Species) ---", flush=True)
        is_results = kfold_cv(
            dCGF_IS_Batched, gf_matrices, targets, n_gf,
            k=k_folds, n_seeds=n_seeds, **train_kwargs, **model_kwargs
        )
        print(f"  Average Pearson r: {is_results['avg_pearson_r']:.4f}", flush=True)
        print(f"  Mean +/- Std across seeds: "
              f"{is_results['mean_correlation']:.4f} +/- "
              f"{is_results['std_correlation']:.4f}", flush=True)

        results[metabolite] = {
            'dCGF-ES': es_results,
            'dCGF-IS': is_results,
        }

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Pearson Correlation (averaged predictions across 5 seeds)")
    print(f"{'='*70}")
    print(f"{'Metabolite':<15} {'dCGF-ES':>12} {'dCGF-IS':>12}")
    print(f"{'-'*15} {'-'*12} {'-'*12}")
    for metabolite in METABOLITES:
        es_r = results[metabolite]['dCGF-ES']['avg_pearson_r']
        is_r = results[metabolite]['dCGF-IS']['avg_pearson_r']
        print(f"{metabolite:<15} {es_r:>12.4f} {is_r:>12.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY: Mean +/- Std Pearson r across seeds")
    print(f"{'='*70}")
    print(f"{'Metabolite':<15} {'dCGF-ES':>20} {'dCGF-IS':>20}")
    print(f"{'-'*15} {'-'*20} {'-'*20}")
    for metabolite in METABOLITES:
        es = results[metabolite]['dCGF-ES']
        is_ = results[metabolite]['dCGF-IS']
        print(f"{metabolite:<15} "
              f"{es['mean_correlation']:>7.4f} +/- {es['std_correlation']:.4f} "
              f"{is_['mean_correlation']:>7.4f} +/- {is_['std_correlation']:.4f}")

    return results


if __name__ == '__main__':
    results = main()
