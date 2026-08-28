#!/usr/bin/env bash
set -euo pipefail

# Independently search all six cross-split directions. Existing zero-hit files
# are reused; missing directions are rebuilt independently.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PPI_REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
split_arg=${1:?"Usage: 08_crosscheck_splits.sh <seed directory or seed name>"}
if [[ "$split_arg" = /* ]]; then
    split_dir="$split_arg"
else
    split_dir="$repo_root/data/mmseqs_seeds_splits/$split_arg"
fi
crosscheck_dir="$split_dir/crosscheck"
work_root="$repo_root/data/mmseqs/mmseqs_work/crosscheck_$(basename -- "$split_dir")"
max_sequences=${PPI_EXPECTED_SEQUENCES:-2486}
threads=${MMSEQS_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}

if ! command -v mmseqs >/dev/null 2>&1; then
    echo "ERROR: mmseqs was not found in PATH" >&2
    exit 1
fi
for split in train validation test; do
    if [[ ! -s "$split_dir/$split.fasta" ]]; then
        echo "ERROR: missing split FASTA: $split_dir/$split.fasta" >&2
        exit 1
    fi
done
mkdir -p "$crosscheck_dir" "$work_root"

for query in train validation test; do
    for target in train validation test; do
        if [[ "$query" == "$target" ]]; then
            continue
        fi
        output="$crosscheck_dir/${query}_vs_${target}.tsv"
        if [[ -f "$output" ]]; then
            rows=$(wc -l < "$output")
            if [[ "$rows" -ne 0 ]]; then
                echo "ERROR: existing crosscheck contains $rows hit(s): $output" >&2
                exit 2
            fi
            echo "UNCHANGED zero-hit audit: $output"
            continue
        fi

        temporary="$crosscheck_dir/.${query}_vs_${target}.$$.tmp.tsv"
        temporary_work=$(mktemp -d "$work_root/${query}_vs_${target}.XXXXXX")
        mmseqs easy-search \
            "$split_dir/$query.fasta" \
            "$split_dir/$target.fasta" \
            "$temporary" \
            "$temporary_work/search_tmp" \
            --min-seq-id 0.3 \
            -c 0.8 \
            --cov-mode 0 \
            -e 1000000 \
            --prefilter-mode 1 \
            --min-ungapped-score 0 \
            --mask 0 \
            --max-seqs "$max_sequences" \
            --threads "$threads" \
            --format-output "query,target,fident,alnlen,qcov,tcov,evalue,bits"
        rm -rf -- "$temporary_work"

        rows=$(wc -l < "$temporary")
        if [[ "$rows" -ne 0 ]]; then
            diagnostic="$crosscheck_dir/${query}_vs_${target}.LEAKAGE.tsv"
            mv -- "$temporary" "$diagnostic"
            echo "ERROR: detected $rows cross-split hit(s): $diagnostic" >&2
            exit 2
        fi
        mv -- "$temporary" "$output"
        echo "WROTE zero-hit audit: $output"
    done
done

file_count=$(find "$crosscheck_dir" -maxdepth 1 -type f -name '*_vs_*.tsv' ! -name '*.LEAKAGE.tsv' | wc -l)
total_hits=$(find "$crosscheck_dir" -maxdepth 1 -type f -name '*_vs_*.tsv' ! -name '*.LEAKAGE.tsv' -exec cat {} + | wc -l)
if [[ "$file_count" -ne 6 || "$total_hits" -ne 0 ]]; then
    echo "ERROR: crosscheck files=$file_count, hits=$total_hits; expected 6 and 0" >&2
    exit 2
fi
echo "Crosscheck files / total hits: 6 / 0"
echo "CROSS-SPLIT AUDIT PASS"
