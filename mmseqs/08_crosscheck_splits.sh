#!/usr/bin/env bash
set -euo pipefail

# Search all six directions between train, validation, and test FASTA files.
# The default audit is seed_0; 09_generate_multiple_splits.sh passes explicit
# directories for every seed.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
data_dir="$(cd "$script_dir/.." && pwd -P)"
split_root="$data_dir/mmseqs_seeds_splits"
work_dir="$script_dir/work"

split_dir="${1:-$split_root/seed_0}"
output_dir="${2:-$work_dir/crosschecks/seed_0}"
tmp_dir="${output_dir}_tmp"
merged_fasta="$script_dir/all_proteins.fasta"

if ! command -v mmseqs >/dev/null 2>&1; then
    echo "ERROR: mmseqs was not found in PATH." >&2
    exit 1
fi

if [[ ! -f "$merged_fasta" ]]; then
    echo "ERROR: missing merged FASTA: $merged_fasta" >&2
    exit 1
fi

max_sequences=$(grep -c '^>' "$merged_fasta")
if [[ "$max_sequences" -ne 2486 ]]; then
    echo "ERROR: expected 2486 merged sequences, found $max_sequences" >&2
    exit 1
fi

echo "Split directory: $split_dir"
echo "Crosscheck output: $output_dir"
echo "Total merged sequences: $max_sequences"

for split in train validation test; do
    fasta_path="$split_dir/${split}.fasta"
    if [[ ! -f "$fasta_path" ]]; then
        echo "ERROR: missing split FASTA: $fasta_path" >&2
        exit 1
    fi
done

# Refuse to mix new searches with an existing audit.
if [[ -d "$output_dir" ]] && compgen -G "$output_dir/*.tsv" >/dev/null; then
    echo "ERROR: TSV results already exist in $output_dir" >&2
    echo "Use a new output directory or remove the old audit explicitly." >&2
    exit 1
fi

mkdir -p "$output_dir" "$tmp_dir"

for query_split in train validation test; do
    for target_split in train validation test; do
        if [[ "$query_split" == "$target_split" ]]; then
            continue
        fi

        echo "Running ${query_split} vs ${target_split}"
        mmseqs easy-search \
            "$split_dir/${query_split}.fasta" \
            "$split_dir/${target_split}.fasta" \
            "$output_dir/${query_split}_vs_${target_split}.tsv" \
            "$tmp_dir/${query_split}_vs_${target_split}" \
            --min-seq-id 0.3 \
            -c 0.8 \
            --cov-mode 0 \
            -e 1000000 \
            --prefilter-mode 1 \
            --min-ungapped-score 0 \
            --mask 0 \
            --max-seqs "$max_sequences" \
            --format-output \
            "query,target,fident,alnlen,qcov,tcov,evalue,bits"
    done
done

shopt -s nullglob
result_files=("$output_dir"/*.tsv)
shopt -u nullglob

if (( ${#result_files[@]} != 6 )); then
    echo "ERROR: expected six directional TSV files, found ${#result_files[@]}" >&2
    exit 1
fi

total_hits=0
for result_file in "${result_files[@]}"; do
    hit_count=$(wc -l < "$result_file")
    total_hits=$((total_hits + hit_count))
    printf '%s\t%s\n' "$(basename "$result_file")" "$hit_count"
done

summary_path="$output_dir/crosscheck_summary.txt"
{
    echo "split_dir=$split_dir"
    echo "directions=6"
    echo "total_cross_split_hits=$total_hits"
    echo "min_seq_id=0.30"
    echo "minimum_query_coverage=0.80"
    echo "minimum_target_coverage=0.80"
} > "$summary_path"

if (( total_hits == 0 )); then
    echo "PASS: all six search directions contain zero cross-split hits."
    echo "Summary: $summary_path"
else
    echo "FAIL: detected $total_hits cross-split hits." >&2
    exit 2
fi
