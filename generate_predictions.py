"""
Run k-fold CV and save predictions to a .npz file for report generation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import os

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


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    model_kwargs = {'z_dim': 30, 'encoder_hidden': 200, 'decoder_hidden': 100}
    n_epochs = 500
    k = 8
    n_seeds = 5
    # Use fewer seeds for Succinate to save time (we know it's ~0.92)
    seeds_override = {'Succinate': 2}

    results = {}

    for metabolite in METABOLITES:
        print(f"\n{metabolite}:", flush=True)
        gf_matrices, targets_std, tgt_mean, tgt_std, gf_vecs, pathway_names = \
            prepare_dataset(data_dir, metabolite)
        n_gf = len(pathway_names)
        n_samples = len(gf_matrices)
        m_seeds = seeds_override.get(metabolite, n_seeds)
        all_predictions = np.zeros((m_seeds, n_samples))

        for seed_idx in range(m_seeds):
            seed = seed_idx + 42
            np.random.seed(seed)
            torch.manual_seed(seed)
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(gf_matrices)):
                train_mats = [torch.tensor(gf_matrices[i]) for i in train_idx]
                test_mats = [torch.tensor(gf_matrices[i]) for i in test_idx]
                train_tgt = torch.tensor(targets_std[train_idx])

                model = dCGF_IS_Batched(n_gf, **model_kwargs)
                train_model(model, train_mats, train_tgt, n_epochs=n_epochs)

                model.eval()
                with torch.no_grad():
                    preds = model(test_mats).numpy()
                all_predictions[seed_idx, test_idx] = preds

            r_seed, _ = pearsonr(all_predictions[seed_idx], targets_std)
            print(f"  Seed {seed_idx+1}/{m_seeds}: r = {r_seed:.4f}", flush=True)

        avg_preds = all_predictions.mean(axis=0)
        avg_r, _ = pearsonr(avg_preds, targets_std)
        print(f"  avg r = {avg_r:.4f}", flush=True)

        # Store in original units
        measured = targets_std * tgt_std + tgt_mean
        predicted = avg_preds * tgt_std + tgt_mean

        results[f'{metabolite}_measured'] = measured
        results[f'{metabolite}_predicted'] = predicted
        results[f'{metabolite}_r'] = np.array([avg_r])

    out_path = os.path.join(data_dir, 'cv_predictions.npz')
    np.savez(out_path, **results)
    print(f"\nSaved predictions to {out_path}", flush=True)


if __name__ == '__main__':
    main()
