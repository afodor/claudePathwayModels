"""
Simple baseline models using only species presence/absence.
1. Random Forest on species presence/absence vector
2. Linear regression on species presence/absence (no interaction terms)

If these perform comparably to dCGF-IS, the deep learning architecture adds nothing.
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import os
import json

from data_processing import (load_community_data, average_replicates,
                             load_genetic_features, ALL_SPECIES, METABOLITES)


def prepare_species_matrix(data_dir):
    """Build species presence/absence matrix for all communities."""
    communities = load_community_data(data_dir)
    gf_vectors, _ = load_genetic_features(data_dir)

    exclude_species = set()
    for sp in ALL_SPECIES:
        if sp not in gf_vectors or np.sum(gf_vectors.get(sp, np.zeros(1))) == 0:
            exclude_species.add(sp)

    valid_species = sorted([sp for sp in ALL_SPECIES if sp not in exclude_species])
    sp_to_idx = {sp: i for i, sp in enumerate(valid_species)}

    filtered = []
    for comm in communities:
        if any(s in exclude_species for s in comm['species']):
            continue
        species_in_comm = [s for s in comm['species'] if s not in exclude_species]
        if len(species_in_comm) == 0:
            continue
        filtered.append(comm)

    filtered = average_replicates(filtered)

    # Build binary species presence matrix (n_communities x n_species)
    X = np.zeros((len(filtered), len(valid_species)), dtype=np.float32)
    all_targets = {}
    for metabolite in METABOLITES:
        all_targets[metabolite] = np.array([c['metabolites'][metabolite] for c in filtered],
                                           dtype=np.float32)

    for i, comm in enumerate(filtered):
        for sp in comm['species']:
            if sp in sp_to_idx:
                X[i, sp_to_idx[sp]] = 1.0

    print(f"Dataset: {X.shape[0]} communities, {X.shape[1]} species features", flush=True)

    # Also build species + pathway matrix (species presence + summed pathway vectors)
    n_pathways = len(list(gf_vectors.values())[0])
    X_sp_pw = np.zeros((len(filtered), len(valid_species) + n_pathways), dtype=np.float32)
    for i, comm in enumerate(filtered):
        # First 24 dims: species presence/absence
        for sp in comm['species']:
            if sp in sp_to_idx:
                X_sp_pw[i, sp_to_idx[sp]] = 1.0
        # Next 144 dims: sum of pathway vectors (weighted by 1/n like dCGF)
        n_sp = len(comm['species'])
        for sp in comm['species']:
            if sp in gf_vectors:
                X_sp_pw[i, len(valid_species):] += gf_vectors[sp] / n_sp

    print(f"Species+Pathways: {X_sp_pw.shape[1]} features "
          f"({len(valid_species)} species + {n_pathways} pathways)", flush=True)

    # Build species + pairwise pathway similarity matrix
    # For each pair (i,j), compute correlation of their pathway vectors (fixed).
    # For each community, feature = similarity if both species present, else 0.
    n_sp_total = len(valid_species)
    n_pairs = n_sp_total * (n_sp_total - 1) // 2
    # Precompute pairwise correlations
    pw_vecs = np.array([gf_vectors[sp] for sp in valid_species])
    pair_corrs = np.zeros(n_pairs, dtype=np.float32)
    pair_indices = []
    idx = 0
    for a in range(n_sp_total):
        for b in range(a + 1, n_sp_total):
            r_val, _ = pearsonr(pw_vecs[a], pw_vecs[b])
            pair_corrs[idx] = r_val
            pair_indices.append((a, b))
            idx += 1

    X_sp_sim = np.zeros((len(filtered), n_sp_total + n_pairs), dtype=np.float32)
    for i, comm in enumerate(filtered):
        present = set()
        for sp in comm['species']:
            if sp in sp_to_idx:
                j = sp_to_idx[sp]
                X_sp_sim[i, j] = 1.0
                present.add(j)
        # Fill pairwise similarity features
        for p, (a, b) in enumerate(pair_indices):
            if a in present and b in present:
                X_sp_sim[i, n_sp_total + p] = pair_corrs[p]

    print(f"Species+PairwiseSim: {X_sp_sim.shape[1]} features "
          f"({n_sp_total} species + {n_pairs} pairwise similarities)", flush=True)

    return X, X_sp_pw, X_sp_sim, all_targets, valid_species


def kfold_cv(model_class, model_kwargs, X, y, k=8, n_seeds=5):
    """Run k-fold CV and return average r."""
    n_samples = len(y)
    all_predictions = np.zeros((n_seeds, n_samples))

    for seed_idx in range(n_seeds):
        seed = seed_idx + 42
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)

        for train_idx, test_idx in kf.split(X):
            model = model_class(**model_kwargs)
            model.fit(X[train_idx], y[train_idx])
            all_predictions[seed_idx, test_idx] = model.predict(X[test_idx])

        r_seed, _ = pearsonr(all_predictions[seed_idx], y)
        print(f"    Seed {seed_idx+1}/{n_seeds}: r = {r_seed:.4f}", flush=True)

    avg_preds = all_predictions.mean(axis=0)
    avg_r, _ = pearsonr(avg_preds, y)
    return avg_r


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70, flush=True)
    print("SIMPLE BASELINES: Species presence/absence only", flush=True)
    print("=" * 70, flush=True)

    X, X_sp_pw, X_sp_sim, all_targets, species_list = prepare_species_matrix(data_dir)

    results = {}

    for metabolite in METABOLITES:
        y = all_targets[metabolite]
        print(f"\n{metabolite}:", flush=True)

        # Ridge regression (linear, no interactions)
        print(f"  Ridge Regression (linear, alpha=1.0):", flush=True)
        r_ridge = kfold_cv(Ridge, {'alpha': 1.0}, X, y)
        print(f"  avg r = {r_ridge:.4f}", flush=True)

        # Random Forest on species only
        print(f"  Random Forest - species only (100 trees):", flush=True)
        r_rf = kfold_cv(RandomForestRegressor,
                        {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
                        X, y)
        print(f"  avg r = {r_rf:.4f}", flush=True)

        # Random Forest on species + pathways
        print(f"  Random Forest - species + pathways (100 trees):", flush=True)
        r_rf_pw = kfold_cv(RandomForestRegressor,
                           {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
                           X_sp_pw, y)
        print(f"  avg r = {r_rf_pw:.4f}", flush=True)

        # Random Forest on species + pairwise pathway similarity
        print(f"  Random Forest - species + pairwise similarity (100 trees):", flush=True)
        r_rf_sim = kfold_cv(RandomForestRegressor,
                            {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
                            X_sp_sim, y)
        print(f"  avg r = {r_rf_sim:.4f}", flush=True)

        results[metabolite] = {'ridge': r_ridge, 'random_forest': r_rf,
                               'rf_species_pathways': r_rf_pw,
                               'rf_species_pairwise_sim': r_rf_sim}

    # Save results
    # Convert numpy floats to Python floats for JSON serialization
    json_results = {}
    for m, d in results.items():
        json_results[m] = {k: float(v) for k, v in d.items()}

    out_path = os.path.join(data_dir, 'simple_baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Metabolite':<12} {'Ridge(sp)':>10} {'RF(sp)':>8} {'RF(sp+pw)':>10} {'RF(sp+sim)':>11}", flush=True)
    print("-" * 56, flush=True)
    for m in METABOLITES:
        print(f"{m:<12} {results[m]['ridge']:>10.4f} {results[m]['random_forest']:>8.4f} "
              f"{results[m]['rf_species_pathways']:>10.4f} "
              f"{results[m]['rf_species_pairwise_sim']:>11.4f}", flush=True)


if __name__ == '__main__':
    main()
