# Dataset and MMseqs scripts

These scripts must be placed at `PPI-Affinity-Benchmark/scripts/`. They infer
the repository root from their own location, so execution does not depend on a
Windows drive letter, user name, or current working directory.

## Required source data

```text
data/MCGLPPI_RawData/
├── PDBBINDdimer_strict_index.csv
└── pdbs/m2_pdbbind_dimer_strict/<pdb_code>/
```

Required commands: Python 3, Open Babel (`obabel`), MMseqs2 (`mmseqs`), and
standard GNU tools available inside WSL/Linux.

## Recommended commands

From the repository root:

```bash
bash data/preprocess/prepare_dataset.sh
```

Generate only selected split seeds:

```bash
bash data/preprocess/prepare_mmseqs_splits.sh 0 42
```

If shared PDBQT/FASTA data already passed, run only MMseqs:

```bash
bash data/preprocess/prepare_mmseqs_splits.sh
```

## Fixed real-dataset checkpoints

```text
Raw PPI rows:                       1,270
PPI pairs with both partner FASTA:  1,245
Final PPI rows:                     1,243
Final partner sequences:            2,486
MMseqs directed hits:               10,890
Qualifying non-self hits:            8,404
Protein clusters:                    1,651
Indivisible PPI groups:                582
Per seed PPI counts:               994 / 124 / 125
Per seed sequence counts:         1988 / 248 / 250
Cross-split hits:                         0
```

Canonical seeds are `0`, `1`, `42`, `142`, and `4242`. Each final directory is
named `data/mmseqs_seeds_splits/seed_<seed>/` and contains:

```text
split_assignments.csv
train_split.csv
val_split.csv
test_split.csv
cross_split_leakage.tsv
train.fasta
validation.fasta
test.fasta
crosscheck/                         # six zero-hit TSV files
```

`proaffinity_label` is computed once in shared preprocessing and propagated
unchanged through every split. `10_validate_final_splits.py` is read-only; it
checks labels, membership, counts, group integrity, FASTAs, and both leakage
audits.

## Failure and rerun behavior

- Generated files are written through temporary files and atomically renamed.
- Completed PDB/PDBQT files are reused. A rerun only retries missing failures.
- Partner FASTA preparation never deletes the FASTA directory.
- Existing deterministic split files are reused when their content agrees.
- A conflicting existing file stops before related outputs are changed.
- Successful earlier seeds remain valid if a later seed fails.
- Detailed failure/audit tables are kept under `data/metadata/`.

Do not use the obsolete 1,245-PPI split directory or the old random-state
splits as MMseqs benchmark inputs.
