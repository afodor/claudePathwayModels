"""
Fetch KEGG metabolic pathway data for all 25 species in the Clark et al. 2021 dataset.
Builds a binary genetic feature (GF) matrix: species x pathways.
"""
import urllib.request
import time
import json
import os
import sys

# Map species abbreviations to KEGG organism codes
KEGG_ORG_CODES = {
    'ER': 'ere',   # Eubacterium rectale (Agathobacter rectalis)
    'FP': 'fpr',   # Faecalibacterium prausnitzii
    'AC': 'acac',  # Anaerostipes caccae
    'CC': 'ccom',  # Coprococcus comes (Allocoprococcus comes)
    'RI': 'rix',   # Roseburia intestinalis
    'EL': 'ele',   # Eggerthella lenta
    'CH': 'phx',   # Clostridium hiranonis (Peptacetobacter hiranonis)
    'DP': 'dpg',   # Desulfovibrio piger
    'BH': 'bhv',   # Blautia hydrogenotrophica
    'CA': 'caer',  # Collinsella aerofaciens
    'PJ': 'pjo',   # Parabacteroides johnsonii
    'DL': 'dlo',   # Dorea longicatena
    'CG': 'casp',  # Clostridium asparagiforme
    'BF': 'bfr',   # Bacteroides fragilis
    'BO': 'boa',   # Bacteroides ovatus
    'BT': 'bth',   # Bacteroides thetaiotaomicron
    'BU': 'bun',   # Bacteroides uniformis
    'BV': 'bvu',   # Bacteroides vulgatus (Phocaeicola vulgatus)
    'BC': 'bcac',  # Bacteroides caccae
    'BY': 'bcel',  # Bacteroides cellulosilyticus
    'DF': 'dfm',   # Dorea formicigenerans
    'BL': 'blo',   # Bifidobacterium longum
    'BP': 'bpsc',  # Bifidobacterium pseudocatenulatum
    'BA': 'bad',   # Bifidobacterium adolescentis
    # PC (Prevotella copri) is not in KEGG - will be handled separately
}

SPECIES_ORDER = ['ER','FP','AC','CC','RI','EL','CH','DP','BH','CA',
                 'PC','PJ','DL','CG','BF','BO','BT','BU','BV','BC',
                 'BY','DF','BL','BP','BA']

def fetch_kegg(url, max_retries=3):
    """Fetch data from KEGG REST API with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = urllib.request.urlopen(url, timeout=30)
            return resp.read().decode('utf-8').strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise e
    return ""

def get_pathways_for_organism(org_code):
    """Get list of KEGG pathway IDs for an organism."""
    url = f'https://rest.kegg.jp/list/pathway/{org_code}'
    data = fetch_kegg(url)
    pathways = {}
    if data:
        for line in data.split('\n'):
            if line.strip():
                parts = line.split('\t')
                # pathway ID like "path:ere00010"
                path_id = parts[0].split(':')[1] if ':' in parts[0] else parts[0]
                # Remove organism prefix to get generic pathway ID (e.g., "00010")
                generic_id = path_id[-5:]  # last 5 chars are the pathway number
                path_name = parts[1] if len(parts) > 1 else ""
                # Remove organism-specific suffix like " - Agathobacter rectalis"
                if ' - ' in path_name:
                    path_name = path_name.split(' - ')[0].strip()
                pathways[generic_id] = path_name
    return pathways

def main():
    cache_file = os.path.join(os.path.dirname(__file__), 'kegg_pathways_cache.json')

    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached pathway data from {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
        species_pathways = data['species_pathways']
        all_pathway_names = data['all_pathway_names']
    else:
        print("Fetching KEGG pathway data for all species...")
        species_pathways = {}  # species_code -> {pathway_id: name}

        for sp_code in SPECIES_ORDER:
            if sp_code == 'PC':
                print(f"  {sp_code}: Prevotella copri not in KEGG, skipping")
                species_pathways[sp_code] = {}
                continue

            org_code = KEGG_ORG_CODES[sp_code]
            print(f"  {sp_code} ({org_code}): ", end="", flush=True)
            pathways = get_pathways_for_organism(org_code)
            species_pathways[sp_code] = pathways
            print(f"{len(pathways)} pathways")
            time.sleep(0.5)  # rate limit

        # Collect all unique pathway IDs and names
        all_pathway_names = {}
        for sp_code, pathways in species_pathways.items():
            for pid, pname in pathways.items():
                if pid not in all_pathway_names:
                    all_pathway_names[pid] = pname

        # Save cache
        with open(cache_file, 'w') as f:
            json.dump({
                'species_pathways': species_pathways,
                'all_pathway_names': all_pathway_names
            }, f, indent=2)
        print(f"\nSaved cache to {cache_file}")

    # Build binary GF matrix
    pathway_ids = sorted(all_pathway_names.keys())
    print(f"\nTotal unique pathways across all species: {len(pathway_ids)}")
    print(f"Species with data: {sum(1 for s in SPECIES_ORDER if len(species_pathways.get(s, {})) > 0)}/25")

    # Create and save the GF matrix as CSV
    import csv
    gf_file = os.path.join(os.path.dirname(__file__), 'genetic_features.csv')
    with open(gf_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: species, pathway1, pathway2, ...
        header = ['species'] + [f"{pid}_{all_pathway_names[pid]}" for pid in pathway_ids]
        writer.writerow(header)

        for sp_code in SPECIES_ORDER:
            sp_pathways = species_pathways.get(sp_code, {})
            row = [sp_code] + [1 if pid in sp_pathways else 0 for pid in pathway_ids]
            writer.writerow(row)

    print(f"Saved genetic feature matrix to {gf_file}")

    # Print summary statistics
    for sp_code in SPECIES_ORDER:
        n = len(species_pathways.get(sp_code, {}))
        print(f"  {sp_code}: {n} pathways")

if __name__ == '__main__':
    main()
