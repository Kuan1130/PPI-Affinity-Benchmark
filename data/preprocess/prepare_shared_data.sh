#!/usr/bin/env bash
set -euo pipefail

# Prepare canonical labels, normalized PDBs, PDBQT files, and partner FASTAs.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PPI_REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
data_scripts="$repo_root/data/preprocess"

echo "Repository: $repo_root"
python3 "$data_scripts/01_prepare_labels.py" --repo-root "$repo_root"
python3 "$data_scripts/02_collect_pdbs.py" --repo-root "$repo_root"
python3 "$data_scripts/03_pdb_to_pdbqt.py" \
    --repo-root "$repo_root" \
    --workers "${PDBQT_WORKERS:-8}"
python3 "$data_scripts/04_pdbqt_to_partner_fasta.py" --repo-root "$repo_root"

echo "SHARED DATA PREPARATION PASS"
