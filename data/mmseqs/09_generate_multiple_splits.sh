#!/usr/bin/env bash
set -euo pipefail

# Generate all canonical seeds by default, or only seeds supplied as arguments.
# Completed files are validated and reused; missing files are filled safely.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PPI_REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
split_root="$repo_root/data/mmseqs_seeds_splits"

if (( $# == 0 )); then
    seeds=(0 1 42 142 4242)
else
    seeds=("$@")
fi
for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: seed must be a non-negative integer: $seed" >&2
        exit 1
    fi
done

for required in \
    "$repo_root/data/mmseqs/usable_index.csv" \
    "$repo_root/data/mmseqs/all_proteins.fasta" \
    "$repo_root/data/mmseqs/mmseqs_out/ppi_groups.tsv" \
    "$repo_root/data/mmseqs/mmseqs_out/all_vs_all.tsv"
do
    if [[ ! -s "$required" ]]; then
        echo "ERROR: missing or empty prerequisite: $required" >&2
        exit 1
    fi
done
mkdir -p "$split_root"

for seed in "${seeds[@]}"; do
    seed_name="seed_$seed"
    seed_dir="$split_root/$seed_name"
    echo
    echo "===== $seed_name ====="

    python3 "$script_dir/06_make_group_split.py" \
        --repo-root "$repo_root" \
        --seed "$seed" \
        --output-dir "$seed_dir"

    python3 "$script_dir/07_make_split_fastas.py" \
        --repo-root "$repo_root" \
        --split-dir "$seed_dir"

    PPI_REPO_ROOT="$repo_root" \
        bash "$script_dir/08_crosscheck_splits.sh" "$seed_dir"

    python3 "$script_dir/10_validate_final_splits.py" \
        --repo-root "$repo_root" \
        --seeds "$seed"

    assignment_hash=$(sha256sum "$seed_dir/split_assignments.csv" | awk '{print $1}')
    echo "Assignment SHA-256: $assignment_hash"
    echo "$seed_name PASS"
done

echo
echo "All requested seeds passed generation and validation: ${seeds[*]}"
