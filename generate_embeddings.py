"""
Pre-compute species functional embeddings for PCA analysis (Paper Figure 6).

Trains dCGF-IS on all data for each metabolite, then extracts per-species
30-dim embeddings by feeding each species (with unity abundance) through
the trained encoder. Saves results to species_embeddings.npz.

This is the slow step (~15 min). generate_report.py reads the output.
"""
import numpy as np
import torch
import os
import sys
import time
import logging

from data_processing import METABOLITES, prepare_dataset, load_genetic_features
from dcgf_model import dCGF_IS_Batched

# Set up logging to both console and file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generate_embeddings.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def extract_species_embeddings(data_dir='.', metabolite='Butyrate', n_seeds=5, n_epochs=500):
    """Train dCGF-IS on all data and extract per-species 30-dim embeddings.

    Following the paper's Figure 6 method: feed each species with unity abundance
    (a_i = 1) through the trained encoder, using the all-species community as context.
    """
    gf_matrices, targets, tgt_mean, tgt_std, gf_vectors, pathway_names = \
        prepare_dataset(data_dir, metabolite)

    n_gf = len(pathway_names)

    # Get list of species with GF data (exclude PC, HB)
    species_list = sorted([sp for sp in gf_vectors if np.sum(gf_vectors[sp]) > 0])

    # Build the "all species" community context with unity abundances (a_i = 1)
    all_species_gf = np.stack([gf_vectors[sp] for sp in species_list], axis=1)
    w_context = torch.tensor(all_species_gf.sum(axis=1), dtype=torch.float32)

    # Train multiple models and average embeddings
    all_embeddings = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = dCGF_IS_Batched(n_gf, z_dim=30, encoder_hidden=200, decoder_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()

        gf_tensors = [torch.tensor(m, dtype=torch.float32) for m in gf_matrices]
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            preds = model(gf_tensors)
            loss = loss_fn(preds, target_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = {}
            for sp in species_list:
                u_i = torch.tensor(gf_vectors[sp], dtype=torch.float32)
                encoder_input = torch.cat([u_i, w_context], dim=0)
                z_i = model.encoder(encoder_input)
                embeddings[sp] = z_i.numpy()
            all_embeddings.append(embeddings)

        log.info(f"  {metabolite} seed {seed}: train loss = {loss.item():.4f}")

    # Average across seeds
    avg_embeddings = {}
    for sp in species_list:
        avg_embeddings[sp] = np.mean([emb[sp] for emb in all_embeddings], axis=0)

    return avg_embeddings, species_list, gf_vectors, pathway_names


def main():
    data_dir = '.'
    out_path = os.path.join(data_dir, 'species_embeddings.npz')

    all_data = {}
    species_list = None
    gf_vecs = None
    pw_names = None

    t_start = time.time()
    for metabolite in METABOLITES:
        t_metab = time.time()
        log.info(f"Training model for {metabolite} embeddings...")
        emb, sp_list, gf_v, pw_n = extract_species_embeddings(data_dir, metabolite)

        # Save embeddings as array (species x 30)
        emb_matrix = np.array([emb[sp] for sp in sp_list])
        all_data[f'{metabolite}_embeddings'] = emb_matrix
        log.info(f"  {metabolite} done in {time.time() - t_metab:.1f}s")

        if species_list is None:
            species_list = sp_list
            gf_vecs = gf_v
            pw_names = pw_n

    # Save raw pathway matrix too
    raw_matrix = np.array([gf_vecs[sp] for sp in species_list])

    np.savez(out_path,
             species_list=np.array(species_list),
             pathway_names=np.array(pw_names),
             raw_pathways=raw_matrix,
             **all_data)

    log.info(f"Saved to {out_path}")
    log.info(f"  Species: {len(species_list)}")
    log.info(f"  Pathways: {len(pw_names)}")
    log.info(f"  Embedding dim: 30")
    for m in METABOLITES:
        log.info(f"  {m}: {all_data[f'{m}_embeddings'].shape}")
    log.info(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
