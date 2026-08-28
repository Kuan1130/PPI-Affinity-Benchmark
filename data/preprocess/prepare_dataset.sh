#!/usr/bin/env bash
set -euo pipefail

# One-command entry point for shared preprocessing and MMseqs split generation.
# Optional arguments select individual seeds, for example: 0 42

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${PPI_REPO_ROOT:-$(cd -- "$script_dir/../.." && pwd)}
export PPI_REPO_ROOT="$repo_root"

bash "$script_dir/prepare_shared_data.sh"
bash "$script_dir/prepare_mmseqs_splits.sh" "$@"

echo "FULL DATASET PIPELINE PASS"
