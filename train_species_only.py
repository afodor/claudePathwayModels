"""
Test: run dCGF-IS using one-hot species identity vectors instead of pathway vectors.
If performance matches the pathway-based model, pathways add nothing beyond species ID.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import os

from dcgf_model import dCGF_IS_Batched
from data_processing import (load_community_data, average_replicates,
                             load_genetic_features, ALL_SPECIES, METABOLITES)


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


def prepare_species_only(data_dir, metabolite):
    """Build GF matrices using one-hot species identity instead of pathways."""
    communities = load_community_data(data_dir)
    gf_vectors, _ = load_genetic_features(data_dir)

    # Determine which species to exclude (same logic as prepare_dataset)
    exclude_species = set()
    for sp in ALL_SPECIES:
        if sp not in gf_vectors or np.sum(gf_vectors.get(sp, np.zeros(1))) == 0:
            exclude_species.add(sp)

    # Get the list of valid species (sorted for consistent ordering)
    valid_species = sorted([sp for sp in ALL_SPECIES if sp not in exclude_species])
    n_species_total = len(valid_species)
    sp_to_idx = {sp: i for i, sp in enumerate(valid_species)}

    # Filter communities (same logic as prepare_dataset)
    filtered = []
    for comm in communities:
        if any(s in exclude_species for s in comm['species']):
            continue
        species_in_comm = [s for s in comm['species'] if s not in exclude_species]
        if len(species_in_comm) == 0:
            continue
        filtered.append(comm)

    filtered = average_replicates(filtered)

    # Build one-hot GF matrices: (n_species_total, n_species_in_community)
    gf_matrices = []
    targets = []
    for comm in filtered:
        species = sorted(comm['species'])
        n_sp = len(species)
        u = np.zeros((n_species_total, n_sp), dtype=np.float32)
        for j, sp in enumerate(species):
            u[sp_to_idx[sp], j] = 1.0 / n_sp  # one-hot scaled by abundance
        gf_matrices.append(u)
        targets.append(comm['metabolites'][metabolite])

    targets = np.array(targets, dtype=np.float32)
    tgt_mean = targets.mean()
    tgt_std = targets.std()
    targets_std = (targets - tgt_mean) / tgt_std if tgt_std > 0 else targets - tgt_mean

    print(f"Dataset prepared for {metabolite} (SPECIES-ONLY):", flush=True)
    print(f"  Communities: {len(gf_matrices)}", flush=True)
    print(f"  Feature dim: {n_species_total} (one-hot species identity)", flush=True)
    print(f"  Valid species: {n_species_total}", flush=True)

    return gf_matrices, targets_std, tgt_mean, tgt_std, n_species_total, valid_species


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    model_kwargs = {'z_dim': 30, 'encoder_hidden': 200, 'decoder_hidden': 100}
    n_epochs = 500
    k = 8
    n_seeds = 5

    print("=" * 70, flush=True)
    print("SPECIES-ONLY MODEL (no pathway information)", flush=True)
    print("One-hot species identity replaces pathway vectors", flush=True)
    print("=" * 70, flush=True)

    for metabolite in METABOLITES:
        print(f"\n{metabolite}:", flush=True)
        gf_matrices, targets_std, tgt_mean, tgt_std, n_features, species_list = \
            prepare_species_only(data_dir, metabolite)
        n_samples = len(gf_matrices)
        all_predictions = np.zeros((n_seeds, n_samples))

        for seed_idx in range(n_seeds):
            seed = seed_idx + 42
            np.random.seed(seed)
            torch.manual_seed(seed)
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gf_matrices)):
                train_mats = [torch.tensor(gf_matrices[i]) for i in train_idx]
                test_mats = [torch.tensor(gf_matrices[i]) for i in test_idx]
                train_tgt = torch.tensor(targets_std[train_idx])

                model = dCGF_IS_Batched(n_features, **model_kwargs)
                train_model(model, train_mats, train_tgt, n_epochs=n_epochs)

                model.eval()
                with torch.no_grad():
                    preds = model(test_mats).numpy()
                all_predictions[seed_idx, test_idx] = preds

            r_seed, _ = pearsonr(all_predictions[seed_idx], targets_std)
            print(f"  Seed {seed_idx+1}/{n_seeds}: r = {r_seed:.4f}", flush=True)

        avg_preds = all_predictions.mean(axis=0)
        avg_r, _ = pearsonr(avg_preds, targets_std)
        print(f"  avg r = {avg_r:.4f}", flush=True)


if __name__ == '__main__':
    main()
