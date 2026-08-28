#!/usr/bin/env bash
set -euo pipefail

# Run the locked all-vs-all MMseqs2 search. A validated existing result is reused.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PPI_REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
work_dir="$repo_root/data/mmseqs"
input_fasta="$work_dir/all_proteins.fasta"
output_dir="$work_dir/mmseqs_out"
output_file="$output_dir/all_vs_all.tsv"
manifest="$output_dir/all_vs_all.settings.txt"
expected_sequences=${PPI_EXPECTED_SEQUENCES:-2486}
expected_hits=${PPI_EXPECTED_DIRECTED_HITS:-10890}
threads=${MMSEQS_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}

if ! command -v mmseqs >/dev/null 2>&1; then
    echo "ERROR: mmseqs was not found in PATH" >&2
    exit 1
fi
if [[ ! -s "$input_fasta" ]]; then
    echo "ERROR: missing or empty FASTA: $input_fasta" >&2
    exit 1
fi

sequence_count=$(grep -c '^>' "$input_fasta")
if [[ "$sequence_count" -ne "$expected_sequences" ]]; then
    echo "ERROR: FASTA sequences=$sequence_count; expected $expected_sequences" >&2
    exit 1
fi

mkdir -p "$output_dir" "$work_dir/mmseqs_work"

settings=$(cat <<EOF
min_seq_id=0.3
coverage=0.8
cov_mode=0
evalue=1000000
prefilter_mode=1
min_ungapped_score=0
mask=0
max_seqs=$expected_sequences
format=query,target,fident,alnlen,qcov,tcov,evalue,bits
EOF
)

if [[ -s "$output_file" ]]; then
    observed_hits=$(wc -l < "$output_file")
    if [[ "$observed_hits" -ne "$expected_hits" ]]; then
        echo "ERROR: existing $output_file has $observed_hits rows; expected $expected_hits" >&2
        exit 1
    fi
    if awk 'NF != 8 { exit 1 }' "$output_file"; then :; else
        echo "ERROR: existing MMseqs output has a row that is not 8 columns" >&2
        exit 1
    fi
    if [[ -f "$manifest" ]] && [[ "$(<"$manifest")" != "$settings" ]]; then
        echo "ERROR: existing search settings differ from the locked settings" >&2
        exit 1
    fi
    printf '%s\n' "$settings" > "$manifest"
    echo "Reusing validated MMseqs result: $output_file"
    echo "Directed hits: $observed_hits"
    echo "MMSEQS ALL-VS-ALL PASS"
    exit 0
fi

temporary_output="$output_dir/.all_vs_all.tsv.$$.tmp"
temporary_root=$(mktemp -d "$work_dir/mmseqs_work/all_vs_all.XXXXXX")
cleanup() {
    rm -f -- "$temporary_output"
    rm -rf -- "$temporary_root"
}
trap cleanup EXIT

mmseqs easy-search \
    "$input_fasta" \
    "$input_fasta" \
    "$temporary_output" \
    "$temporary_root/search_tmp" \
    --min-seq-id 0.3 \
    -c 0.8 \
    --cov-mode 0 \
    -e 1000000 \
    --prefilter-mode 1 \
    --min-ungapped-score 0 \
    --mask 0 \
    --max-seqs "$expected_sequences" \
    --threads "$threads" \
    --format-output "query,target,fident,alnlen,qcov,tcov,evalue,bits"

if [[ ! -s "$temporary_output" ]]; then
    echo "ERROR: MMseqs produced no output" >&2
    exit 1
fi
observed_hits=$(wc -l < "$temporary_output")
if [[ "$observed_hits" -ne "$expected_hits" ]]; then
    diagnostic="$output_dir/all_vs_all.unexpected_${observed_hits}_rows.tsv"
    mv -- "$temporary_output" "$diagnostic"
    echo "ERROR: directed hits=$observed_hits; expected $expected_hits" >&2
    echo "Candidate result retained for diagnosis: $diagnostic" >&2
    exit 1
fi
if awk 'NF != 8 { exit 1 }' "$temporary_output"; then :; else
    echo "ERROR: MMseqs output contains a row that is not 8 columns" >&2
    exit 1
fi

mv -- "$temporary_output" "$output_file"
printf '%s\n' "$settings" > "$manifest"
echo "Directed hits: $observed_hits"
echo "Output: $output_file"
echo "MMSEQS ALL-VS-ALL PASS"
