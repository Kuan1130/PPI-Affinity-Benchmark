#!/usr/bin/env bash
set -euo pipefail

# Generate and audit the five canonical MMseqs2 group-disjoint splits.
#
# Default:
#   bash 09_generate_multiple_splits.sh
#
# Selected seeds only:
#   bash 09_generate_multiple_splits.sh 0 42
#
# Final directories:
#   data/mmseqs_seeds_splits/seed_0
#   data/mmseqs_seeds_splits/seed_1
#   data/mmseqs_seeds_splits/seed_42
#   data/mmseqs_seeds_splits/seed_142
#   data/mmseqs_seeds_splits/seed_4242

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
data_dir="$(cd "$script_dir/.." && pwd -P)"
split_root="$data_dir/mmseqs_seeds_splits"
work_dir="$script_dir/work"
crosscheck_root="$work_dir/crosschecks"

split_script="$script_dir/06_make_group_split.py"
fasta_script="$script_dir/07_make_split_fastas.py"
crosscheck_script="$script_dir/08_crosscheck_splits.sh"
source_index="$script_dir/usable_index.csv"
merged_fasta="$script_dir/all_proteins.fasta"
ppi_groups="$script_dir/mmseqs_out/ppi_groups.tsv"
all_hits="$script_dir/mmseqs_out/all_vs_all.tsv"

if (( $# == 0 )); then
    requested_seeds=(0 1 42 142 4242)
else
    requested_seeds=("$@")
fi

for required_path in \
    "$split_script" \
    "$fasta_script" \
    "$crosscheck_script" \
    "$source_index" \
    "$merged_fasta" \
    "$ppi_groups" \
    "$all_hits"
do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: missing required path: $required_path" >&2
        exit 1
    fi
done

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found in PATH." >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum was not found in PATH." >&2
    exit 1
fi

mkdir -p "$split_root" "$crosscheck_root"

echo "Script directory: $script_dir"
echo "Split root:       $split_root"
echo "Crosscheck root:  $crosscheck_root"
echo "Seeds:            ${requested_seeds[*]}"

declare -A assignment_hashes
declare -A hash_to_seed

for seed in "${requested_seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: seed must be a non-negative integer: $seed" >&2
        exit 1
    fi

    split_dir="$split_root/seed_${seed}"
    crosscheck_dir="$crosscheck_root/seed_${seed}"

    echo
    echo "============================================================"
    echo "Processing split seed $seed"
    echo "Output directory: $split_dir"
    echo "============================================================"

    # Reuse only a complete assignment. Refuse ambiguous partial directories.
    if [[ -f "$split_dir/split_assignments.csv" ]]; then
        echo "Reusing existing split assignment."
    else
        if [[ -d "$split_dir" ]] && [[ -n "$(ls -A "$split_dir")" ]]; then
            echo "ERROR: $split_dir exists but has no split_assignments.csv." >&2
            echo "Inspect the partial directory manually before rerunning." >&2
            exit 1
        fi

        python3 "$split_script" --seed "$seed" --output-dir "$split_dir"
    fi

    # Step 06 writes train.csv, validation.csv, and test.csv. If a completed
    # repository retains only the labeled files from step 10, accept those as
    # idempotent validation inputs instead.
    python3 - "$split_dir" "$source_index" <<'PY'
from pathlib import Path
import csv
import sys

split_dir = Path(sys.argv[1])
source_path = Path(sys.argv[2])

specs = [
    ("train.csv", "train_split.csv", 994),
    ("validation.csv", "val_split.csv", 124),
    ("test.csv", "test_split.csv", 125),
]

with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    source_fields = reader.fieldnames
    source_rows = list(reader)

if source_fields is None or len(source_rows) != 1243:
    raise SystemExit(
        f"ERROR: {source_path} must contain a header and 1243 rows"
    )

for raw_name, labeled_name, expected_rows in specs:
    raw_path = split_dir / raw_name
    labeled_path = split_dir / labeled_name
    path = raw_path if raw_path.exists() else labeled_path

    if not path.exists():
        raise SystemExit(
            f"ERROR: missing both {raw_path} and {labeled_path}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames

    if len(rows) != expected_rows:
        raise SystemExit(
            f"ERROR: {path} has {len(rows)} rows; expected {expected_rows}"
        )
    # After step 10, usable_index.csv contains proaffinity_label while the raw
    # step-06 CSVs intentionally remain unchanged. Both schemas are valid.
    source_fields_without_label = [
        field for field in source_fields if field != "proaffinity_label"
    ]
    if fields not in (source_fields, source_fields_without_label):
        raise SystemExit(
            f"ERROR: {path} columns are inconsistent with {source_path.name}"
        )

print("CSV check: PASS (994 / 124 / 125; columns unchanged)")
PY

    if [[ ! -f "$split_dir/train.fasta" ]] || \
       [[ ! -f "$split_dir/validation.fasta" ]] || \
       [[ ! -f "$split_dir/test.fasta" ]]; then
        python3 "$fasta_script" --split-dir "$split_dir"
    else
        echo "Reusing existing split FASTA files."
    fi

    train_sequence_count=$(grep -c '^>' "$split_dir/train.fasta")
    validation_sequence_count=$(grep -c '^>' "$split_dir/validation.fasta")
    test_sequence_count=$(grep -c '^>' "$split_dir/test.fasta")

    if [[ "$train_sequence_count" -ne 1988 ]] || \
       [[ "$validation_sequence_count" -ne 248 ]] || \
       [[ "$test_sequence_count" -ne 250 ]]; then
        echo "ERROR: incorrect split FASTA sequence counts:" >&2
        echo "train=$train_sequence_count validation=$validation_sequence_count test=$test_sequence_count" >&2
        exit 1
    fi
    echo "FASTA check: PASS (1988 / 248 / 250 sequences)"

    # A reusable audit consists of exactly six empty directional result TSVs.
    if [[ -d "$crosscheck_dir" ]]; then
        shopt -s nullglob
        crosscheck_files=("$crosscheck_dir"/*.tsv)
        shopt -u nullglob

        if (( ${#crosscheck_files[@]} != 6 )); then
            echo "ERROR: $crosscheck_dir contains ${#crosscheck_files[@]} TSV files; expected 6." >&2
            exit 1
        fi

        total_hits=0
        for hit_file in "${crosscheck_files[@]}"; do
            hit_count=$(wc -l < "$hit_file")
            total_hits=$((total_hits + hit_count))
        done
        if (( total_hits != 0 )); then
            echo "ERROR: existing crosscheck contains $total_hits hits." >&2
            exit 2
        fi
        echo "Reusing existing six-direction zero-hit audit."
    else
        bash "$crosscheck_script" "$split_dir" "$crosscheck_dir"
    fi

    assignment_path="$split_dir/split_assignments.csv"
    assignment_hash=$(sha256sum "$assignment_path" | awk '{print $1}')
    assignment_hashes["$seed"]="$assignment_hash"

    if [[ -n "${hash_to_seed[$assignment_hash]:-}" ]]; then
        previous_seed="${hash_to_seed[$assignment_hash]}"
        echo "WARNING: seeds $seed and $previous_seed produced identical assignments."
    else
        hash_to_seed["$assignment_hash"]="$seed"
    fi

    echo "Assignment SHA-256: $assignment_hash"
    echo "Seed $seed: PASS"
done

echo
echo "===================== FINAL SUMMARY ====================="
printf '%-10s  %-60s  %s\n' "Seed" "Split directory" "Assignment SHA-256"
for seed in "${requested_seeds[@]}"; do
    split_dir="$split_root/seed_${seed}"
    printf '%-10s  %-60s  %s\n' \
        "$seed" "$split_dir" "${assignment_hashes[$seed]}"
done

echo
echo "All requested splits passed CSV, FASTA, and leakage checks."
echo "Next: python3 $script_dir/10_add_proaffinity_labels.py"
echo "Then: python3 $script_dir/10_add_proaffinity_labels.py --apply"
