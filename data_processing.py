"""
Data processing for dCGF model training.

Reads the Clark et al. 2021 dataset and KEGG pathway genetic features,
and prepares community GF matrices and metabolite targets.
"""
import pandas as pd
import numpy as np
import os
import json


# 25 species abbreviations used in the Clark et al. dataset
ALL_SPECIES = ['ER','FP','AC','HB','CC','RI','EL','CH','DP','BH','CA',
               'PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC',
               'BY','DF','BL','BP','BA']

# The 25 species in the dCGF paper (excludes HB which is in Clark data
# but listed differently - HB = Holdemanella biformis)
DCGF_SPECIES = ['ER','FP','AC','CC','RI','EL','CH','DP','BH','CA',
                'PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC',
                'BY','DF','BL','BP','BA']

METABOLITES = ['Acetate', 'Butyrate', 'Lactate', 'Succinate']


def load_community_data(data_dir='.'):
    """
    Load Clark et al. 2021 community data.

    Returns:
        communities: list of dicts, each with:
            - 'species': list of species codes present
            - 'n_species': number of species
            - 'metabolites': dict of metabolite -> concentration
            - 'treatment': treatment string
    """
    df = pd.read_csv(os.path.join(data_dir, 'MasterDF.csv'))

    # Species columns contain 1/0 for presence/absence
    # Use the species columns that overlap with our 25 species
    species_cols = [s for s in ALL_SPECIES if s in df.columns]

    communities = []
    for _, row in df.iterrows():
        present_species = [s for s in species_cols if row[s] == 1]
        if len(present_species) == 0:
            continue

        metab_vals = {}
        valid = True
        for m in METABOLITES:
            val = row[m]
            if pd.isna(val):
                valid = False
                break
            metab_vals[m] = val
        if not valid:
            continue

        communities.append({
            'species': present_species,
            'n_species': len(present_species),
            'metabolites': metab_vals,
            'treatment': row.get('Treatment', ''),
        })

    return communities


def load_genetic_features(data_dir='.'):
    """
    Load genetic feature matrix from KEGG pathway data.

    Returns:
        gf_vectors: dict of species_code -> numpy array (binary GF vector)
        pathway_names: list of pathway names (column labels)
    """
    gf_df = pd.read_csv(os.path.join(data_dir, 'genetic_features.csv'))
    pathway_names = [c for c in gf_df.columns if c != 'species']

    gf_vectors = {}
    for _, row in gf_df.iterrows():
        sp = row['species']
        gf_vectors[sp] = np.array([row[p] for p in pathway_names], dtype=np.float32)

    return gf_vectors, pathway_names


def average_replicates(communities):
    """
    Average metabolite values across replicates for the same treatment.
    The paper says: "When multiple replicates were available, we took the
    average of the replicates."
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for comm in communities:
        # Key by the sorted set of species present
        key = tuple(sorted(comm['species']))
        grouped[key].append(comm)

    averaged = []
    for key, comms in grouped.items():
        avg_metab = {}
        for m in METABOLITES:
            vals = [c['metabolites'][m] for c in comms]
            avg_metab[m] = np.mean(vals)

        averaged.append({
            'species': list(key),
            'n_species': len(key),
            'metabolites': avg_metab,
            'treatment': comms[0]['treatment'],
        })

    return averaged


def prepare_dataset(data_dir='.', metabolite='Butyrate', exclude_species=None):
    """
    Prepare dataset for dCGF training.

    Args:
        data_dir: directory containing MasterDF.csv and genetic_features.csv
        metabolite: which metabolite to predict
        exclude_species: list of species to exclude (for communities containing
                        species without KEGG data)

    Returns:
        gf_matrices: list of numpy arrays, each (n_gf, n_species_in_community)
        targets: numpy array of metabolite values (standardized)
        target_mean: mean before standardization
        target_std: std before standardization
        gf_vectors: dict of species -> GF vector
        pathway_names: list of pathway names
    """
    communities = load_community_data(data_dir)
    gf_vectors, pathway_names = load_genetic_features(data_dir)

    if exclude_species is None:
        exclude_species = []
    # Always exclude species without GF data
    exclude_species = set(exclude_species)
    for sp in ALL_SPECIES:
        if sp not in gf_vectors or np.sum(gf_vectors.get(sp, np.zeros(1))) == 0:
            exclude_species.add(sp)

    # Filter communities: remove those with excluded species
    filtered = []
    for comm in communities:
        species_in_comm = [s for s in comm['species'] if s not in exclude_species]
        if len(species_in_comm) == 0:
            continue
        # Only keep community if ALL original species have GF data
        if any(s in exclude_species for s in comm['species']):
            continue
        filtered.append(comm)

    # Average replicates
    filtered = average_replicates(filtered)

    n_gf = len(pathway_names)

    # Build GF matrices
    gf_matrices = []
    targets = []
    for comm in filtered:
        species = sorted(comm['species'])
        n_sp = len(species)
        # Each column = species GF vector scaled by initial abundance
        # Clark et al: equal relative abundances, so a_i = 1/n_species
        u = np.zeros((n_gf, n_sp), dtype=np.float32)
        for j, sp in enumerate(species):
            u[:, j] = gf_vectors[sp] * (1.0 / n_sp)
        gf_matrices.append(u)
        targets.append(comm['metabolites'][metabolite])

    targets = np.array(targets, dtype=np.float32)

    # Standardize targets (subtract mean, divide by std)
    target_mean = targets.mean()
    target_std = targets.std()
    if target_std > 0:
        targets_standardized = (targets - target_mean) / target_std
    else:
        targets_standardized = targets - target_mean

    print(f"Dataset prepared for {metabolite}:")
    print(f"  Communities: {len(gf_matrices)}")
    print(f"  Genetic features (pathways): {n_gf}")
    print(f"  Species with GF data: {len(gf_vectors) - len(exclude_species)}")
    print(f"  Excluded species: {exclude_species}")
    print(f"  Target mean: {target_mean:.4f}, std: {target_std:.4f}")
    print(f"  Community sizes: min={min(c['n_species'] for c in filtered)}, "
          f"max={max(c['n_species'] for c in filtered)}")

    return (gf_matrices, targets_standardized, target_mean, target_std,
            gf_vectors, pathway_names)


if __name__ == '__main__':
    # Test data loading
    for metab in METABOLITES:
        print(f"\n{'='*60}")
        prepare_dataset('.', metab)
