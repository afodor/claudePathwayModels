"""
Control: replace real KEGG pathways with random binary pathways.
80% of pathways shared across all species, 20% randomly assigned per species.
Tests whether pathway STRUCTURE matters or just having distinguishing features.
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


def make_random_pathways(n_pathways, species_list, shared_frac=0.80, rng=None):
    """
    Create random binary pathway matrix.
    - shared_frac of pathways are present in ALL species (1 for all)
    - remaining pathways are randomly assigned per species (50% chance each)
    """
    if rng is None:
        rng = np.random.RandomState(0)

    n_species = len(species_list)
    n_shared = int(n_pathways * shared_frac)
    n_variable = n_pathways - n_shared

    gf_vectors = {}
    for sp in species_list:
        vec = np.zeros(n_pathways, dtype=np.float32)
        # Shared pathways: all 1
        vec[:n_shared] = 1.0
        # Variable pathways: random 0/1 with 50% probability
        vec[n_shared:] = rng.binomial(1, 0.5, n_variable).astype(np.float32)
        gf_vectors[sp] = vec

    return gf_vectors


def prepare_random_pathway_dataset(data_dir, metabolite, gf_vectors, n_pathways,
                                    exclude_species):
    """Build GF matrices using random pathway vectors."""
    communities = load_community_data(data_dir)

    filtered = []
    for comm in communities:
        if any(s in exclude_species for s in comm['species']):
            continue
        species_in_comm = [s for s in comm['species'] if s not in exclude_species]
        if len(species_in_comm) == 0:
            continue
        filtered.append(comm)

    filtered = average_replicates(filtered)

    gf_matrices = []
    targets = []
    for comm in filtered:
        species = sorted(comm['species'])
        n_sp = len(species)
        u = np.zeros((n_pathways, n_sp), dtype=np.float32)
        for j, sp in enumerate(species):
            u[:, j] = gf_vectors[sp] * (1.0 / n_sp)
        gf_matrices.append(u)
        targets.append(comm['metabolites'][metabolite])

    targets = np.array(targets, dtype=np.float32)
    tgt_mean = targets.mean()
    tgt_std = targets.std()
    targets_std = (targets - tgt_mean) / tgt_std if tgt_std > 0 else targets - tgt_mean

    return gf_matrices, targets_std, tgt_mean, tgt_std


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    model_kwargs = {'z_dim': 30, 'encoder_hidden': 200, 'decoder_hidden': 100}
    n_epochs = 500
    k = 8
    n_seeds = 5
    n_pathways = 144  # same dimensionality as real KEGG

    # Determine excluded species (same as real model)
    real_gf, _ = load_genetic_features(data_dir)
    exclude_species = set()
    for sp in ALL_SPECIES:
        if sp not in real_gf or np.sum(real_gf.get(sp, np.zeros(1))) == 0:
            exclude_species.add(sp)

    valid_species = sorted([sp for sp in ALL_SPECIES if sp not in exclude_species])

    print("=" * 70, flush=True)
    print("RANDOM PATHWAY CONTROL", flush=True)
    print(f"  {n_pathways} random binary pathways", flush=True)
    print(f"  80% shared across all species, 20% random per species", flush=True)
    print(f"  Valid species: {len(valid_species)}", flush=True)
    print(f"  Excluded: {exclude_species}", flush=True)
    print("=" * 70, flush=True)

    # Generate random pathways (fixed seed for reproducibility)
    rng = np.random.RandomState(12345)
    random_gf = make_random_pathways(n_pathways, valid_species, shared_frac=0.80, rng=rng)

    # Show how many variable pathways differ per species pair
    vecs = np.array([random_gf[sp] for sp in valid_species])
    pairwise_diff = []
    for i in range(len(valid_species)):
        for j in range(i+1, len(valid_species)):
            pairwise_diff.append(np.sum(vecs[i] != vecs[j]))
    print(f"  Pairwise pathway differences: mean={np.mean(pairwise_diff):.1f}, "
          f"min={np.min(pairwise_diff)}, max={np.max(pairwise_diff)}", flush=True)
    print(f"  (Real KEGG pairwise diffs for comparison would be similar range)", flush=True)

    all_results = {}
    for metabolite in METABOLITES:
        print(f"\n{metabolite}:", flush=True)
        gf_matrices, targets_std, tgt_mean, tgt_std = \
            prepare_random_pathway_dataset(data_dir, metabolite, random_gf,
                                           n_pathways, exclude_species)
        n_samples = len(gf_matrices)
        print(f"  Communities: {n_samples}", flush=True)

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

                model = dCGF_IS_Batched(n_pathways, **model_kwargs)
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
        all_results[metabolite] = avg_r

    # Save results
    import json
    out_path = os.path.join(data_dir, 'random_pathway_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
