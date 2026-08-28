#!/usr/bin/env bash
set -euo pipefail

# Build the locked MMseqs graph and canonical five-seed splits.
# Optional arguments select individual seeds, for example: 0 42

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PPI_REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
mmseqs_scripts="$repo_root/data/mmseqs"
export PPI_REPO_ROOT="$repo_root"

echo "Repository: $repo_root"
python3 "$mmseqs_scripts/01_check_fasta.py" --repo-root "$repo_root"
python3 "$mmseqs_scripts/02_merge_fasta.py" --repo-root "$repo_root"
python3 "$mmseqs_scripts/03_filter_unclusterable.py" --repo-root "$repo_root"
bash "$mmseqs_scripts/04_run_mmseqs_all_vs_all.sh"
python3 "$mmseqs_scripts/05_build_ppi_groups.py" --repo-root "$repo_root"
bash "$mmseqs_scripts/09_generate_multiple_splits.sh" "$@"

if (( $# == 0 )); then
    python3 "$mmseqs_scripts/10_validate_final_splits.py" --repo-root "$repo_root"
else
    python3 "$mmseqs_scripts/10_validate_final_splits.py" \
        --repo-root "$repo_root" \
        --seeds "$@"
fi

echo "MMSEQS SPLIT PIPELINE PASS"
