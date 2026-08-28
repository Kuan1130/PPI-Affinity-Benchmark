# MMseqs2-based PPI affinity split

## 1. Purpose

This workflow replaces the previous sample-level random train/validation/test split with a sequence-disjoint split for evaluating unseen-target generalization.

The final rule is:

- two partner sequences are considered related when MMseqs2 reports sequence identity >= 0.30 and bidirectional alignment coverage >= 0.80;
- related sequences are placed in the same connected component;
- the two partners of every PPI sample (`pdb_1` and `pdb_2`) are also joined because one PPI sample cannot be split;
- each resulting PPI group is assigned wholly to train, validation, or test;
- the final split is independently searched in all six cross-split directions, with zero qualifying MMseqs2 hits required.

This is stricter than simply keeping the representative clusters from a greedy clustering run intact. It uses connected components of MMseqs2 homology hits and then verifies the completed split directly.

## 2. Software and environment

- Host system: Windows
- Linux environment: Ubuntu under WSL2
- MMseqs2 version: `18-8cc5c+ds-1`
- Project directory in Windows:
  `E:\使用者\kuok\HKU\0_0\DrLi\5\mmseqs`
- The same directory in WSL:
  `/mnt/e/使用者/kuok/HKU/0_0/DrLi/5/mmseqs`

Enter the environment with:

```cmd
wsl -d Ubuntu
```

Then enter the project directory:

```bash
cd "/mnt/e/使用者/kuok/HKU/0_0/DrLi/5/mmseqs"
```

Exit the WSL shell with `exit`. Use `python3`, not `python`, in Ubuntu.

## 3. Input data and exclusions

### Original dataset

- Original PPI samples in `PDBBINDdimer_strict_index.csv`: 1,270
- Expected FASTA files: 2,540 (two partner sequences per PPI)
- FASTA naming convention: `pdb_1.fasta` and `pdb_2.fasta`
- The existing structure-processing pipeline may concatenate multiple chains belonging to the same interaction partner. Therefore the split is defined over the resulting partner-level sequences, not over every PDB chain separately.

### Missing FASTA exclusions

Twenty-five PPI samples had neither usable partner FASTA and were excluded:

```text
1e6e, 1lzw, 1m5n, 1pjn, 1zv5,
2m0j, 2n01, 2ru4, 3lb8, 3u82,
4bru, 4c99, 4dm6, 4etw, 4mp0,
4mqv, 4rws, 4zkc, 5dmj, 5lxq,
5nqf, 5nqg, 5tl6, 6h71, 6ire
```

After these exclusions:

- PPI samples: 1,245
- partner sequences: 2,490

The exclusions are recorded in `excluded_missing_fasta.csv`.

### Unclusterable sequence exclusions

Two additional PPIs were excluded because one partner did not contain a meaningful clusterable amino-acid sequence:

| PDB | Sequence | Reason |
| --- | --- | --- |
| `3kv4_2` | `X` | length 1; unknown residue only |
| `4ft4_2` | `ATRX` | length 4; contains unknown residue |

The whole PPI sample was excluded in each case so that every retained sample has two assessable partners. These exclusions are recorded in `excluded_unclusterable_sequence.csv`.

### Final usable dataset

- PPI samples: **1,243**
- partner sequences: **2,486**
- usable metadata: `usable_index.csv`
- merged FASTA: `all_proteins.fasta`

The merged FASTA headers were normalized to unique identifiers such as:

```text
>1a22_1
SEQUENCE...
>1a22_2
SEQUENCE...
```

## 4. Why the MMseqs2 search was made stricter

The first search used the conventional settings:

```text
--min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7.5
```

It returned 10,647 directed hits. However, an independent train-to-validation search found two short-peptide relationships that had not passed the initial full-database search:

| Query | Target | Identity | Query coverage | Target coverage |
| --- | --- | ---: | ---: | ---: |
| `3kj0_2` | `5vmo_2` | 0.590 | 0.957 | 0.833 |
| `3kj1_2` | `5vmo_2` | 0.619 | 0.955 | 0.833 |

The short alignments had E-values of approximately `4.1e-4` and `7.2e-4` against the smaller validation database, but larger E-values against the complete database. This exposed an unwanted dependence on the default `E-value <= 0.001` filter even though the intended split criterion was identity plus coverage.

The final all-vs-all search therefore used relaxed significance/prefilter settings so that short sequences were not silently discarded:

```bash
mmseqs easy-search \
  all_proteins.fasta \
  all_proteins.fasta \
  mmseqs_out/all_vs_all.tsv \
  mmseqs_tmp_strict_1243 \
  --min-seq-id 0.3 \
  -c 0.8 \
  --cov-mode 0 \
  -e 1000000 \
  --prefilter-mode 1 \
  --min-ungapped-score 0 \
  --mask 0 \
  --max-seqs 2486 \
  --format-output "query,target,fident,alnlen,qcov,tcov,evalue,bits"
```

Parameter interpretation:

- `--min-seq-id 0.3`: retain alignments with identity >= 30%;
- `-c 0.8 --cov-mode 0`: require at least 80% coverage of both query and target;
- `-e 1000000`: prevent the default E-value threshold from removing short alignments before applying the intended identity/coverage rule;
- `--prefilter-mode 1`: use the ungapped prefilter mode, which is closer to an exhaustive candidate search than the default k-mer prefilter;
- `--min-ungapped-score 0`: allow short candidates into the alignment stage;
- `--mask 0`: do not hide low-complexity residues when applying the literal identity criterion;
- `--max-seqs 2486`: do not cap candidates below the size of the sequence database.

The final search produced:

- all directed hits: **10,890**
- qualifying non-self directed hits used as homology edges: **8,404**
- maximum hits for one query: **45**
- sequences with no MMseqs2 output: **0**

The claim supported by this workflow is: **no cross-split MMseqs2 hit satisfies the stated identity and coverage thresholds under the recorded search configuration.** It should not be restated as a mathematical guarantee under every possible alignment algorithm or alternative coverage definition.

## 5. Building protein clusters and PPI groups

The `grouping.py` script performs two union-find/connected-component steps.

### Protein connected components

Every non-self MMseqs2 hit satisfying all of the following becomes an undirected homology edge:

```text
fident >= 0.30
qcov   >= 0.80
tcov   >= 0.80
```

Connected components of these edges form the protein clusters.

Final protein-cluster statistics:

- sequences: 2,486
- protein clusters: **1,651**
- singleton protein clusters: **1,368**
- largest 20 protein-cluster sizes:
  `[55, 51, 37, 32, 23, 20, 18, 17, 17, 15, 13, 13, 12, 11, 11, 10, 10, 9, 8, 8]`

The mapping is stored in `mmseqs_out/protein_clusters.tsv`.

### PPI connected components

The two partners of every retained PPI (`pdb_1` and `pdb_2`) are then joined. This propagates the homology constraints through complete PPI samples. The resulting component is the smallest unit that may be assigned to a split.

Final PPI-group statistics:

- PPI samples: 1,243
- indivisible PPI groups: **582**
- singleton PPI groups: **459**
- largest 20 PPI-group sizes:
  `[284, 108, 12, 12, 10, 9, 9, 9, 7, 7, 7, 7, 7, 7, 6, 5, 5, 5, 5, 5]`

The increase from protein-cluster size to PPI-group size is expected: a PPI connects its two partner components, and those components can connect additional PPIs transitively.

The mapping is stored in `mmseqs_out/ppi_groups.tsv`.

## 6. Final 80/10/10 split

The `split.py` script uses the PPI group as the indivisible assignment unit. It does not use affinity labels to choose the split. A deterministic dynamic-programming assignment with `SEED = 42` was used to meet the requested proportions exactly.

Because 1,243 is not divisible into exact decimal percentages, the target counts were:

| Split | PPI samples | Fraction | PPI groups | Largest group sizes |
| --- | ---: | ---: | ---: | --- |
| Train | 994 | 79.97% | 408 | 284, 108, 12, 12, 9, ... |
| Validation | 124 | 9.98% | 100 | 7, 5, 4, 2, 2, ... |
| Test | 125 | 10.06% | 74 | 10, 7, 7, 6, 5, ... |

Checks performed during assignment:

- exact target counts reached: `True`;
- PPI groups split across subsets: `0`;
- qualifying cross-split hits in the final all-vs-all table: `0`.

Important outputs in `mmseqs_split/`:

- `split_assignments.csv`: complete metadata plus group and split assignment;
- `train.csv`, `validation.csv`, `test.csv`: final dataset tables;
- `train.fasta`, `validation.fasta`, `test.fasta`: final partner sequences;
- `cross_split_leakage.tsv`: audit output from the assignment script.

Final FASTA counts:

- train: 1,988 sequences;
- validation: 248 sequences;
- test: 250 sequences.

## 7. Independent cross-split verification

The completed FASTA subsets were searched in all six directions using the same strict parameters:

```text
train -> validation
validation -> train
train -> test
test -> train
validation -> test
test -> validation
```

All six files in `mmseqs_crosscheck_strict/` contained **0 hits**. This confirms that the final fixed split contains no cross-split MMseqs2 match satisfying identity >= 0.30 and bidirectional coverage >= 0.80 under the recorded strict configuration.

## 8. Main reproducibility files

| File or directory | Purpose |
| --- | --- |
| `PDBBINDdimer_strict_index.csv` | original 1,270-sample metadata |
| `fasta/` | original per-partner FASTA files |
| `check.py` | validates FASTA availability and sequence characters |
| `merge.py` | creates unique merged FASTA and usable metadata |
| `filter_unclusterable.py` | removes `3kv4` and `4ft4` and records the reasons |
| `grouping.py` | builds protein and PPI connected components |
| `split.py` | makes the deterministic group-level 80/10/10 split |
| `make_split_fastas.py` | writes split-specific FASTA files |
| `excluded_missing_fasta.csv` | records the 25 missing-FASTA exclusions |
| `excluded_unclusterable_sequence.csv` | records the 2 unclusterable-sequence exclusions |
| `usable_index.csv` | final 1,243-sample metadata |
| `all_proteins.fasta` | final 2,486 partner sequences |
| `mmseqs_out/all_vs_all.tsv` | final strict MMseqs2 all-vs-all result |
| `mmseqs_out/protein_clusters.tsv` | protein connected-component mapping |
| `mmseqs_out/ppi_groups.tsv` | indivisible PPI-group mapping |
| `mmseqs_split/` | final seed-42 split files |
| `mmseqs_crosscheck_strict/` | six independent zero-hit audits |

The older `mmseqs_split_1245_obsolete/` split is retained only as an audit trail and must not be used for model training or reporting.

## 9. Is one split enough?

### Current development benchmark

One fixed, published split is sufficient for initial model development and direct baseline comparison, provided that every model uses exactly the same assignment. The seed-42 split should be treated as the primary benchmark and should not be regenerated after looking at model test performance.

### Stronger final-paper evaluation

A single split can have variance caused by which unseen groups happen to enter validation and test. If compute permits, the recommended robustness analysis is **3 to 5 repeated group-disjoint splits**:

1. keep the final 1,243 samples, strict MMseqs2 hit table, protein clusters, and 582 PPI groups fixed;
2. vary only the assignment seed in `split.py`, for example `0, 1, 2, 3, 4`;
3. generate a separate directory for each seed rather than overwriting seed 42;
4. run the same group-overlap audit and six-direction MMseqs2 cross-check for every seed;
5. train every model and baseline on the same set of seeds;
6. report the mean and standard deviation of Pearson, Spearman, RMSE/MAE, or the metrics used by the project.

Do **not** return to ordinary sample-level random splitting for this robustness test. That would mix related targets across splits and answer a different, easier question.

### Alternative: group K-fold cross-validation

Group K-fold evaluation is possible, but the largest PPI group contains 284 samples. A five-fold target is only about 249 samples, so balanced 5-fold cross-validation is impossible without accepting a substantially oversized fold. Four-fold group CV has a target of about 311 samples and is more feasible, but it requires four training runs per model and a separate validation strategy.

For this project, the practical recommendation is:

- use seed 42 as the primary fixed benchmark now;
- for the final paper, add at least three and preferably five repeated PPI-group-disjoint seeds and report mean +/- standard deviation;
- keep hyperparameter tuning and test evaluation separated, and publish every assignment CSV and seed.

## 10. Suggested Methods wording

> We removed complexes lacking valid partner sequences and retained 1,243 PPI complexes (2,486 partner sequences). MMseqs2 v18 was used to identify sequence relationships at a minimum sequence identity of 30% and a bidirectional alignment coverage of 80%. All qualifying relationships were converted into connected components, after which the two partners of each PPI were joined to form indivisible PPI groups. These groups were assigned to training, validation, and test sets at an approximately 80:10:10 ratio, yielding 994, 124, and 125 complexes, respectively. No PPI group was shared across subsets. Independent MMseqs2 searches in all six cross-subset directions identified no hit satisfying the identity and coverage thresholds.

The exact MMseqs2 command and all assignment files should accompany the released benchmark.
