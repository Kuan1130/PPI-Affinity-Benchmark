# MMseqs2 PPI Split Pipeline

## 1. Aim
This pipeline constructs five homology-aware train/validation/test splits
for the MCGLPPI dataset using MMseqs2 sequence similarity clustering.

Final split location:
data/mmseqs_seeds_splits/seed_\<seed\>/

## 2. Data
1,270 raw PPIs
  - 25 PPIs excluded because of missing FASTA files
= 1,245 PPIs
  - 2 PPIs excluded because their sequences could not be clustered(3KV4, 4FT4)
= 1,243 final PPIs

Final sequence count: 2,486 partner sequences

## 3. Parameters
```
MMseqs2 homology definition:
- Sequence identity (fident) >= 0.30
- Query coverage (qcov) >= 0.80
- Target coverage (tcov) >= 0.80

Split sizes:
- Train:      994 PPIs
- Validation: 124 PPIs
- Test:       125 PPIs

Seeds:
- 0
- 1
- 42
- 142
- 4242
```

## 4. Run from WSL

MMseqs2 and the Bash scripts should be run in WSL or Linux. For the repository
location supplied with this package:

```bash
cd 'PPI-Affinity-Benchmark\data\mmseqs'

python3 data/mmseqs/01_check_fasta.py
python3 data/mmseqs/02_merge_fasta.py
python3 data/mmseqs/03_filter_unclusterable.py
bash data/mmseqs/04_run_mmseqs_all_vs_all.sh
python3 data/mmseqs/05_build_ppi_groups.py
bash data/mmseqs/09_generate_multiple_splits.sh

# Validate label conversion without changing any file.
python3 data/mmseqs/10_add_proaffinity_labels.py

# Write the final labeled model-input CSVs after the dry run passes.
python3 data/mmseqs/10_add_proaffinity_labels.py --apply
```

Scripts 06–08 are normally called automatically by script 09.
They only need to be executed manually when debugging an individual seed.

## 5. Expected results

|Script|Function|Expected Results|Main Outputs|
|---|---|---|---|
|`01_check_fasta.py`|Check if there are actually 2 fastas for each PPIs|25 PPIs are irreparable FASTA|FASTA report|
|`02_merge_fasta.py`|Merge available FASTA|1,270 → 1,245 PPI; 2,490 sequences|`all_proteins_before_unknown_filter.fasta`, `excluded_missing_fasta.csv`|
|`03_filter_unclusterable.py`|Remove unclusterable data|Remove 2 PPIs; Valid 1,243 PPI, 2,486 sequences|`all_proteins.fasta`, `usable_index.csv`, `excluded_unclusterable_sequence.csv`|
|`04_run_mmseqs_all_vs_all.sh`|MMseqs2 all-vs-all search|Successfully non-empty results generated; 10,890 directed hits|`mmseqs_out/all_vs_all.tsv`|
|`05_build_ppi_groups.py`|Construct protein clusters and indivisible PPI groups|1,651 protein clusters; 582 PPI groups|`protein_clusters.tsv`, `ppi_groups.tsv`|
|`06_make_group_split.py`|Distribution of PPI groups to 3 splits|994/124/125; broken groups = 0|`split_assignments.csv`, `train.csv`, `validation.csv`, `test.csv`|
|`07_make_split_fastas.py`|Generate splits' FASTAs|1,988/248/250 sequences|`train.fasta`, `validation.fasta`, `test.fasta`|
|`08_crosscheck_splits.sh`|Detecting cross-split contamination|0 hits in 6 directions; PASS|`cross_split_leakage.tsv` and TSV of 6 directions|
|`09_generate_multiple_splits.sh`|Execute 06–08 for the 5 seeds|5 seeds PASS|`data/mmseqs_seeds_splits/seed_<seed>/`|
|`10_add_proaffinity_labels.py`|Verify membership and add lables|Each seeds: 994/124/125; NO overlap and NO contamination|`train_split.csv`, `val_split.csv`, `test_split.csv`|

## 5. Further successful norm

### `04_run_mmseqs_all_vs_all.sh`

```text
Sequences: 2,486
Directed MMseqs2 hits: 10,890
Qualifying non-self directed hits: 8,404
Sequences without output: 0
Maximum hits for one query: 45
```

### `05_build_ppi_groups.py`

```text
Protein clusters: 1,651
Singleton protein clusters: 1,368
Largest protein cluster: 55 sequences

PPI groups: 582
Singleton PPI groups: 459
Largest PPI group: 284 PPIs
Second-largest PPI group: 108 PPIs
```

### `06_make_group_split.py`

```text
Train:      994 PPIs
Validation: 124 PPIs
Test:       125 PPIs
Total:    1,243 PPIs

Exact target reached: True
Broken PPI groups: 0
Cross-split MMseqs hits: 0
```

### `07_make_split_fastas.py`

```text
Train:      1,988 sequences
Validation:   248 sequences
Test:         250 sequences
Total:      2,486 sequences
```

### `08_crosscheck_splits.sh`

```text
train -> validation: 0
validation -> train: 0
train -> test: 0
test -> train: 0
validation -> test: 0
test -> validation: 0

Total cross-split hits: 0
PASS
```

Note: If header is kept in `cross_split_leakage.tsv`, `wc -l` might display `1`, which still represents 0 leakage. This still notifies success. 

### `10_add_proaffinity_labels.py`

Dry run：

```text
Source rows: 1,243

seed_0:    994 / 124 / 125
seed_1:    994 / 124 / 125
seed_42:   994 / 124 / 125
seed_142:  994 / 124 / 125
seed_4242: 994 / 124 / 125

No overlaps
No missing PDB codes
Membership validation PASS
No files modified
```

After using `--apply`, all seeds should appear:

```text
train_split.csv
val_split.csv
test_split.csv
```

and include `proaffinity_label`

## 6. Final folder structure

```text
data/
├── MCGLPPI_RawData/
│   └── PDBBINDdimer_strict_index.csv
│
├── mmseqs/
│   ├── 01_check_fasta.py
│   ├── 02_merge_fasta.py
│   ├── 03_filter_unclusterable.py
│   ├── 04_run_mmseqs_all_vs_all.sh
│   ├── 05_build_ppi_groups.py
│   ├── 06_make_group_split.py
│   ├── 07_make_split_fastas.py
│   ├── 08_crosscheck_splits.sh
│   ├── 09_generate_multiple_splits.sh
│   ├── 10_add_proaffinity_labels.py
│   ├── README.md
│   ├── fasta/
│   ├── all_proteins.fasta
│   ├── usable_index.csv
│   └── mmseqs_out/
│
└── mmseqs_seeds_splits/
    ├── seed_0/
    ├── seed_1/
    ├── seed_42/
    ├── seed_142/
    └── seed_4242/
```

每個 seed 最少應包含：

```text
seed_<seed>/
├── split_assignments.csv
├── train.csv
├── validation.csv
├── test.csv
├── train.fasta
├── validation.fasta
├── test.fasta
├── cross_split_leakage.tsv
├── train_split.csv
├── val_split.csv
└── test_split.csv
```


## Canonical split seeds

The five split seeds are:

```text
0, 1, 42, 142, 4242
```

Each split contains:

```text
train:       994 PPI samples / 1,988 partner sequences
validation:  124 PPI samples /   248 partner sequences
test:        125 PPI samples /   250 partner sequences
total:     1,243 PPI samples / 2,486 partner sequences
```

## Final outputs

The final model-input files are written to:

```text
data/splits/mmseqs/
├── seed_0/
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── seed_1/
├── seed_42/
├── seed_142/
└── seed_4242/
```

Intermediate assignments, split FASTAs, and leakage reports are stored in the
same seed directory while the pipeline runs. Crosscheck search results and
MMseqs2 temporary data are stored under `data/mmseqs/work/`.

## Locked homology rule

All searches use:

```text
minimum sequence identity: 0.30
minimum query coverage:    0.80
minimum target coverage:   0.80
```

The split unit is an indivisible PPI group: homologous sequences and both
partners of the same PPI cannot cross train, validation, and test boundaries.

## Safe reruns

- Script 03 is idempotent after its exclusion audit exists.
- Script 04 refuses to overwrite an existing non-empty `all_vs_all.tsv`.
- Script 09 reuses complete assignments and zero-hit crosschecks, but refuses
  ambiguous partial directories.
- Script 10 performs a dry run by default and writes only with `--apply`.
