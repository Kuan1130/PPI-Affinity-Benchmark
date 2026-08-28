#!/usr/bin/env bash
set -euo pipefail

# Run the strict all-vs-all MMseqs2 search from any working directory.
# Execute this script in Linux or WSL with `mmseqs` available in PATH.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
input_fasta="$script_dir/all_proteins.fasta"
output_dir="$script_dir/mmseqs_out"
work_dir="$script_dir/work"
tmp_dir="$work_dir/mmseqs_tmp_all_vs_all"
output_tsv="$output_dir/all_vs_all.tsv"

if ! command -v mmseqs >/dev/null 2>&1; then
    echo "ERROR: mmseqs was not found in PATH." >&2
    exit 1
fi

if [[ ! -f "$input_fasta" ]]; then
    echo "ERROR: missing input FASTA: $input_fasta" >&2
    echo "Run 02_merge_fasta.py and 03_filter_unclusterable.py first." >&2
    exit 1
fi

sequence_count=$(grep -c '^>' "$input_fasta")
if [[ "$sequence_count" -ne 2486 ]]; then
    echo "ERROR: expected 2486 sequences, found $sequence_count in $input_fasta" >&2
    exit 1
fi

mkdir -p "$output_dir" "$work_dir"

if [[ -s "$output_tsv" ]]; then
    echo "ERROR: output already exists: $output_tsv" >&2
    echo "Move or remove it explicitly before starting a new search." >&2
    exit 1
fi

echo "Input FASTA: $input_fasta"
echo "Sequences: $sequence_count"
echo "Output TSV: $output_tsv"
echo "Temporary directory: $tmp_dir"

mmseqs easy-search \
    "$input_fasta" \
    "$input_fasta" \
    "$output_tsv" \
    "$tmp_dir" \
    --min-seq-id 0.3 \
    -c 0.8 \
    --cov-mode 0 \
    -e 1000000 \
    --prefilter-mode 1 \
    --min-ungapped-score 0 \
    --mask 0 \
    --max-seqs "$sequence_count" \
    --format-output \
    "query,target,fident,alnlen,qcov,tcov,evalue,bits"

echo "MMseqs2 all-vs-all search completed: $output_tsv"
