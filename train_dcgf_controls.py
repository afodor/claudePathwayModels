"""
dCGF control experiments:
  1. Actual data (baseline)
  2. Random abundances (paper's Figure 3C control)
  3. Shuffled targets (permutation null, 10 permutations)

All conditions run to 2000 epochs. We evaluate at 500, 1000, and 2000 epochs
to see how performance changes with more training.

dCGF-IS only (the paper's main model).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import os
import argparse

from dcgf_model import dCGF_IS_Batched
from data_processing import prepare_dataset, METABOLITES


def train_and_eval_checkpoints(model, train_matrices, train_targets,
                                test_matrices, test_targets,
                                checkpoints=(500, 1000, 2000),
                                lr=1e-3, weight_decay=1e-4):
    """
    Train model up to max(checkpoints) epochs, evaluating at each checkpoint.
    Returns dict of {epoch: (r, predictions)}.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    max_epochs = max(checkpoints)
    checkpoint_set = set(checkpoints)
    results = {}

    model.train()
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        predictions = model(train_matrices)
        loss = loss_fn(predictions, train_targets)
        loss.backward()
        optimizer.step()

        if epoch in checkpoint_set:
            model.eval()
            with torch.no_grad():
                preds = model(test_matrices).numpy()
            targets_np = test_targets.numpy()
            if len(preds) < 3:
                r = 0.0
            else:
                r, _ = pearsonr(preds, targets_np)
            results[epoch] = (r, preds.copy())
            model.train()

    return results


def kfold_cv_checkpoints(gf_matrices, targets, n_gf, k=8, n_seeds=5,
                          checkpoints=(500, 1000, 2000),
                          lr=1e-3, weight_decay=1e-4, label="",
                          **model_kwargs):
    """
    k-fold CV evaluating at multiple epoch checkpoints.
    Returns dict of {epoch: {'avg_pearson_r', 'mean_correlation', 'std_correlation', ...}}
    """
    n_samples = len(gf_matrices)
    # predictions[epoch][seed] = array of per-sample predictions
    predictions = {ep: np.zeros((n_seeds, n_samples)) for ep in checkpoints}
    seed_corrs = {ep: [] for ep in checkpoints}

    for seed_idx in range(n_seeds):
        seed = seed_idx + 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        preds_this_seed = {ep: np.full(n_samples, np.nan) for ep in checkpoints}

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gf_matrices)):
            train_mats = [torch.tensor(gf_matrices[i]) for i in train_idx]
            test_mats = [torch.tensor(gf_matrices[i]) for i in test_idx]
            train_tgt = torch.tensor(targets[train_idx])
            test_tgt = torch.tensor(targets[test_idx])

            model = dCGF_IS_Batched(n_gf, **model_kwargs)
            fold_results = train_and_eval_checkpoints(
                model, train_mats, train_tgt, test_mats, test_tgt,
                checkpoints=checkpoints, lr=lr, weight_decay=weight_decay
            )
            for ep in checkpoints:
                r, preds = fold_results[ep]
                preds_this_seed[ep][test_idx] = preds

            # Print progress for last checkpoint only
            r_last = fold_results[max(checkpoints)][0]
            print(f"    {label} Fold {fold_idx+1}/{k}: r@{max(checkpoints)} = {r_last:.4f}", flush=True)

        for ep in checkpoints:
            predictions[ep][seed_idx] = preds_this_seed[ep]
            r_seed, _ = pearsonr(preds_this_seed[ep], targets)
            seed_corrs[ep].append(r_seed)

        # Print seed summary for each checkpoint
        for ep in sorted(checkpoints):
            print(f"  {label} Seed {seed_idx+1}/{n_seeds} @{ep}ep: "
                  f"r = {seed_corrs[ep][-1]:.4f}", flush=True)

    # Aggregate
    out = {}
    for ep in checkpoints:
        avg_preds = predictions[ep].mean(axis=0)
        avg_r, _ = pearsonr(avg_preds, targets)
        out[ep] = {
            'avg_pearson_r': avg_r,
            'seed_correlations': seed_corrs[ep],
            'mean_correlation': np.mean(seed_corrs[ep]),
            'std_correlation': np.std(seed_corrs[ep]),
        }
    return out


def randomize_abundances(gf_matrices, rng):
    """Replace equal abundances (1/n_sp) with random weights."""
    randomized = []
    for mat in gf_matrices:
        n_gf, n_sp = mat.shape
        binary = mat * n_sp  # recover binary vectors
        rand_abd = rng.random(n_sp).astype(np.float32)
        rand_abd = rand_abd / rand_abd.sum()
        randomized.append(binary * rand_abd[np.newaxis, :])
    return randomized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metabolite', type=str, default=None,
                        help='Single metabolite to run (e.g. Acetate). Runs all if omitted.')
    parser.add_argument('--condition', type=str, default='all',
                        choices=['all', 'actual', 'random', 'shuffled'],
                        help='Which condition to run.')
    parser.add_argument('--perm', type=int, default=None,
                        help='Single permutation index (0-9) for shuffled condition.')
    args = parser.parse_args()

    metabolites = [args.metabolite] if args.metabolite else METABOLITES
    condition = args.condition

    print("=" * 70, flush=True)
    print(f"dCGF-IS Controls: {', '.join(metabolites)}", flush=True)
    print("Evaluating at 500, 1000, 2000 epochs", flush=True)
    print("10 shuffled-target permutations for null distribution", flush=True)
    print("=" * 70, flush=True)

    data_dir = os.path.dirname(os.path.abspath(__file__))

    model_kwargs = {
        'z_dim': 30,
        'encoder_hidden': 200,
        'decoder_hidden': 100,
    }
    train_kwargs = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
    }
    checkpoints = (500, 1000, 2000)
    k_folds = 8
    n_seeds = 5
    n_permutations = 10

    results = {}

    for metabolite in metabolites:
        print(f"\n{'='*70}", flush=True)
        print(f"Metabolite: {metabolite}", flush=True)
        print(f"{'='*70}", flush=True)

        gf_matrices, targets, tgt_mean, tgt_std, gf_vecs, pathway_names = \
            prepare_dataset(data_dir, metabolite)
        n_gf = len(pathway_names)

        actual = None
        rand_abd = None
        shuf_all = None

        # --- 1. Actual data ---
        if condition in ('all', 'actual'):
            print(f"\n--- Actual Data ---", flush=True)
            actual = kfold_cv_checkpoints(
                gf_matrices, targets, n_gf, k=k_folds, n_seeds=n_seeds,
                checkpoints=checkpoints, label="Actual", **train_kwargs, **model_kwargs
            )
            for ep in checkpoints:
                print(f"  Actual @{ep}ep: avg_r={actual[ep]['avg_pearson_r']:.4f}  "
                      f"mean={actual[ep]['mean_correlation']:.4f} +/- "
                      f"{actual[ep]['std_correlation']:.4f}", flush=True)

        # --- 2. Random abundances ---
        if condition in ('all', 'random'):
            print(f"\n--- Random Abundances ---", flush=True)
            rng = np.random.RandomState(99)
            rand_gf = randomize_abundances(gf_matrices, rng)
            rand_abd = kfold_cv_checkpoints(
                rand_gf, targets, n_gf, k=k_folds, n_seeds=n_seeds,
                checkpoints=checkpoints, label="RandAbd", **train_kwargs, **model_kwargs
            )
            for ep in checkpoints:
                print(f"  RandAbd @{ep}ep: avg_r={rand_abd[ep]['avg_pearson_r']:.4f}  "
                      f"mean={rand_abd[ep]['mean_correlation']:.4f} +/- "
                      f"{rand_abd[ep]['std_correlation']:.4f}", flush=True)

        # --- 3. Shuffled targets ---
        if condition in ('all', 'shuffled'):
            if args.perm is not None:
                perm_indices = [args.perm]
                print(f"\n--- Shuffled Targets (permutation {args.perm}) ---", flush=True)
            else:
                perm_indices = list(range(n_permutations))
                print(f"\n--- Shuffled Targets ({n_permutations} permutations) ---", flush=True)

            shuf_all = {ep: [] for ep in checkpoints}
            for perm_idx in perm_indices:
                rng_shuf = np.random.RandomState(200 + perm_idx)
                shuffled_targets = targets.copy()
                rng_shuf.shuffle(shuffled_targets)

                print(f"\n  Permutation {perm_idx+1}/{n_permutations}", flush=True)
                shuf = kfold_cv_checkpoints(
                    gf_matrices, shuffled_targets, n_gf, k=k_folds, n_seeds=n_seeds,
                    checkpoints=checkpoints, label=f"Shuf{perm_idx+1}",
                    **train_kwargs, **model_kwargs
                )
                for ep in checkpoints:
                    shuf_all[ep].append(shuf[ep]['avg_pearson_r'])
                    print(f"    Perm {perm_idx+1} @{ep}ep: "
                          f"avg_r={shuf[ep]['avg_pearson_r']:.4f}", flush=True)

            # Shuffled summary
            if len(perm_indices) > 1:
                print(f"\n  Shuffled null distribution:", flush=True)
                for ep in checkpoints:
                    vals = shuf_all[ep]
                    print(f"    @{ep}ep: mean={np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                          f"range=[{np.min(vals):.4f}, {np.max(vals):.4f}]", flush=True)

        results[metabolite] = {
            'actual': actual,
            'random_abundance': rand_abd,
            'shuffled_null': shuf_all,
        }

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"FINAL SUMMARY (condition={condition})", flush=True)
    print(f"{'='*70}", flush=True)

    for ep in checkpoints:
        print(f"\n--- At {ep} epochs ---", flush=True)
        for metabolite in metabolites:
            r = results[metabolite]
            parts = [f"{metabolite:<12}"]
            if r['actual'] is not None:
                parts.append(f"Actual={r['actual'][ep]['avg_pearson_r']:.4f}")
            if r['random_abundance'] is not None:
                parts.append(f"RandAbd={r['random_abundance'][ep]['avg_pearson_r']:.4f}")
            if r['shuffled_null'] is not None:
                vals = r['shuffled_null'][ep]
                parts.append(f"ShufMean={np.mean(vals):.4f} ShufMax={np.max(vals):.4f}")
            print("  ".join(parts), flush=True)

    return results


if __name__ == '__main__':
    results = main()
