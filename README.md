# PPI Affinity Benchmark

This repository benchmarks seven protein–protein interaction (PPI) affinity predictors on a processed MCGLPPI dataset under one shared evaluation protocol.

> **This is an MMseqs2 benchmark, not a random-split benchmark.** The five seed names identify five independently generated, sequence-aware data partitions. They are not five random model initializations on one split and are not k-fold cross-validation.

## Benchmark protocol

- Final dataset: **1,243 PPI complexes** and **2,486 partner FASTA files**.
- Split seeds: `0`, `1`, `42`, `142`, and `4242`.
- Every split contains **994 train**, **124 validation**, and **125 test** complexes.
- MMseqs2 rule: minimum sequence identity `0.30`, query coverage `0.80`, and target coverage `0.80`.
- Homologous proteins and both partners of one PPI are assigned as indivisible groups.
- Six-direction cross-checks must find zero train/validation/test MMseqs2 hits.
- Target: `proaffinity_label = -log10(Kd in mol/L)`.
- `3KV4` and `4FT4` are excluded because one partner sequence in each complex cannot be clustered under the locked pipeline.

## Results

Values are mean ± population standard deviation across the five MMseqs2 partitions.

| Method | Pearson (Rp) ↑ | Spearman (Rs) ↑ | RMSE ↓ | MAE ↓ |
|---|---:|---:|---:|---:|
| ProAffinity | 0.2462 ± 0.1207 | 0.2189 ± 0.1235 | 1.9585 ± 0.4425 | 1.5025 ± 0.3453 |
| **Graphomer** | **0.3874 ± 0.0686** | **0.3638 ± 0.0695** | **1.5649 ± 0.2396** | **1.2005 ± 0.1510** |
| GearNet-Res | 0.2173 ± 0.0627 | 0.2111 ± 0.0460 | 1.8592 ± 0.2378 | 1.4120 ± 0.1273 |
| GearNet-Atom | 0.1387 ± 0.1138 | 0.1431 ± 0.0837 | 1.7122 ± 0.2303 | 1.3255 ± 0.1469 |
| Frozen ESM2-3B + MLP | 0.3391 ± 0.1184 | 0.3271 ± 0.1290 | 1.6495 ± 0.1786 | 1.2721 ± 0.1393 |
| FoldX | 0.2218 ± 0.1250 | 0.2468 ± 0.1116 | 1.6562 ± 0.2570 | 1.2840 ± 0.1553 |
| Rosetta InterfaceAnalyzer | 0.1683 ± 0.0600 | 0.2099 ± 0.1382 | 1.6631 ± 0.2454 | 1.2944 ± 0.1503 |

For neural models, metrics are calculated directly from test predictions. For FoldX and Rosetta, correlations use the raw affinity scores (`-Interaction Energy` and `-dG_separated`). Their RMSE and MAE use an affine calibration fitted **only on the corresponding training split**. Rosetta's primary result uses `-dG_separated`; interface density is diagnostic only.

## Repository layout

```text
PPI-Affinity-Benchmark/
├── baselines/                  # Seven baseline implementations
├── data/
│   ├── MCGLPPI_RawData/        # Original source data
│   ├── preprocess/             # Shared PDB/PDBQT/FASTA preparation
│   ├── mmseqs/                 # MMseqs2 pipeline and partner FASTAs
│   ├── mmseqs_seeds_splits/    # Five final split directories
│   ├── metadata/               # Canonical labels and audit tables
│   └── local/                  # Generated PDB/PDBQT files; normally untracked
├── environments/               # Conda base environments for incompatible stacks
├── requirements/               # Curated direct pip dependencies per model
├── results/
│   ├── runtime/                # Checkpoints, caches, logs, and job files
│   └── summaries/              # Final metrics, predictions, and reports
└── software/                   # External binaries and model caches; untracked
```

## Requirements

- Use Linux or WSL and create one environment per neural baseline. The PyTorch/CUDA stacks are intentionally not combined.
- Install PyTorch before each ordinary requirements file so pip cannot silently replace the required GPU build.
- Run all commands from the repository root unless a section explicitly changes directory.
- FoldX 5.1 and Rosetta are separately licensed and are not included in this repository.

<details>
<summary><strong>Environment installation commands</strong></summary>

### ProAffinity

```bash
conda env create -f environments/proaffinity.yml
conda activate proaffinity

python -m pip install \
  torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements/proaffinity.txt

python -c "import torch, torch_geometric, transformers, pandas, scipy; print('ProAffinity environment PASS'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Graphomer

```bash
conda env create -f environments/graphomer.yml
conda activate ppi-graphomer

python -m pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl

python -m pip install -r requirements/graphomer.txt

python -c "import torch, torch_scatter, esm, Bio, numpy, pandas; print('Graphomer environment PASS'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Graphomer is trained from scratch. `fair-esm` is used only to extract its ESM2 and ESM-IF1 input features.

### GearNet-Res and GearNet-Atom

```bash
conda env create -f environments/gearnet.yml
conda activate gearnet

python -m pip install \
  torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install \
  torch-scatter==2.0.8 torch-cluster==1.5.9 \
  -f https://data.pyg.org/whl/torch-1.8.1+cu111.html

python -m pip install -r requirements/gearnet.txt

python -c "import torch, torch_scatter, torch_cluster, torchdrug, rdkit, pandas; print('GearNet environment PASS'); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Frozen ESM2-3B

```bash
conda env create -f environments/esm2.yml
conda activate esm2

python -m pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements/esm2.txt

python -c "import torch, transformers; print('ESM2 environment PASS'); print(torch.__version__, transformers.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### Shared preprocessing and MMseqs2

```bash
conda env create -f environments/preprocess.yml
conda activate ppi-preprocess

obabel -V
mmseqs version
```

### FoldX and Rosetta

Their orchestration and analysis scripts use only the Python standard library, so both physical baselines share one environment. The external binaries are still required.

```bash
conda env create -f environments/physics.yml
conda activate ppi-physics

python --version
test -x software/foldx/foldx5/foldx_20261231 && echo "FoldX executable PASS"
test -x software/rosetta/current/source/bin/InterfaceAnalyzer.static.linuxgccrelease && echo "Rosetta executable PASS"
test -d software/rosetta/current/database && echo "Rosetta database PASS"
```

</details>

## Complete execution workflow

<details>
<summary><strong>1. Shared preprocessing</strong></summary>

Expected source layout:

```text
data/MCGLPPI_RawData/
├── PDBBINDdimer_strict_index.csv
└── pdbs/m2_pdbbind_dimer_strict/
```

Run the four canonical preprocessing scripts:

```bash
python3 data/preprocess/01_prepare_labels.py
python3 data/preprocess/02_collect_pdbs.py
python3 data/preprocess/03_pdb_to_pdbqt.py --workers 8
python3 data/preprocess/04_pdbqt_to_partner_fasta.py
```

The final script directly writes the ProAffinity-compatible indexes; no separate export notebook is required.

Expected outputs:

```text
data/metadata/ppi_index_labeled.csv
data/metadata/pdb_collection_manifest.csv
data/local/pdbs/<PDB>.pdb
data/local/pdbqt/<PDB>_atom_processed.pdbqt
data/mmseqs/fasta/<PDB>_1.fasta
data/mmseqs/fasta/<PDB>_2.fasta
baselines/Proaffinity/ProAffinity_Test/ProAffinity-GNN/data/chain_index.txt
baselines/Proaffinity/ProAffinity_Test/ProAffinity-GNN/data/PPIdataindex.txt
baselines/Proaffinity/ProAffinity_Test/ProAffinity-GNN/data/PPIdataindex_kd.txt
```

The expected checkpoints are 1,270 labeled/source PDB entries and 1,245 successfully converted partner pairs before the MMseqs2 filter.

</details>

<details>
<summary><strong>2. Generate the five MMseqs2 splits</strong></summary>

Run the scripts in order:

```bash
cd data/mmseqs

python3 01_check_fasta.py
python3 02_merge_fasta.py
python3 03_filter_unclusterable.py
bash 04_run_mmseqs_all_vs_all.sh
python3 05_build_ppi_groups.py
bash 09_generate_multiple_splits.sh

# Validate label generation without writing.
python3 10_add_proaffinity_labels.py

# Write train_split.csv, val_split.csv, and test_split.csv.
python3 10_add_proaffinity_labels.py --apply

cd ../..
```

Scripts `06`, `07`, and `08` are called automatically by script `09`. To regenerate selected partitions only:

```bash
cd data/mmseqs
bash 09_generate_multiple_splits.sh 0 42
cd ../..
```

Successful output:

```text
data/mmseqs_seeds_splits/
├── seed_0/
├── seed_1/
├── seed_42/
├── seed_142/
└── seed_4242/
```

Every seed directory must contain `train_split.csv`, `val_split.csv`, and `test_split.csv` with 994, 124, and 125 rows. Do not continue if a cross-check reports any cross-split hit.

</details>

<details>
<summary><strong>3. ProAffinity</strong></summary>

```bash
cd baselines/Proaffinity/ProAffinity_Test/ProAffinity-GNN

python3 pdb2graph.py
python3 graph_construct.py
python3 pdb2graph_individual.py
python3 graph_construct_indi.py

nohup python -u train_model.py > train_model.log 2>&1 &
tail -f train_model.log
```

The five MMseqs2 partitions are read from `../../../../data/mmseqs_seeds_splits`. Graphs are stored under `data/graph/`, while checkpoints and curves are stored under `model/seed_*/`.

</details>

<details>
<summary><strong>4. Graphomer</strong></summary>

Keep the author's `ppi-graphomer/data/hetatm_list.npy` in place, then run:

```bash
# Supply the model-specific copy of the canonical index.
cp \
  baselines/Proaffinity/ProAffinity_Test/ProAffinity-GNN/data/PPIdataindex.txt \
  baselines/Graphomer/Graphomer_Test/PPIdataindex.txt

cd baselines/Graphomer/Graphomer_Test/ppi-graphomer

python -u preprocess_cpu.py \
  --workers 16 \
  --save_dir ./preprocess_cpu_data \
  --pdb_folder ../../../../data/local/pdbs

python -u preprocess_gpu.py \
  --workers 1 \
  --save_dir ./preprocess_gpu_data \
  --pdb_folder ../../../../data/local/pdbs \
  --single_process True

python -u data_check.py \
  --cpu_path ./preprocess_cpu_data \
  --gpu_path ./preprocess_gpu_data \
  --save_folder ./checked_data

nohup python -u train_final.py > train_final.log 2>&1 &
tail -f train_final.log
```

`train_final.py` loops over all five MMseqs2 partitions and calls `generate_batch.py`. Formal outputs are written to `runs/5seeds_results/`. The obsolete `I. pdbs.ipynb` and `II. merge.ipynb` are not used: shared preprocessing replaces the first, and `data_check.py` performs the alignment formerly handled by the second.

</details>

<details>
<summary><strong>5. GearNet-Res and GearNet-Atom</strong></summary>

Both variants use the same cleaned PDB directory and MMseqs2 split CSVs.

```bash
cd baselines/GearNet-Res/GearNetRes_Test/GearNet

nohup python -u run_gearnet_res.py > gearnet_res.log 2>&1 &
nohup python -u run_gearnet_atom.py > gearnet_atom.log 2>&1 &

tail -f gearnet_res.log
# or
tail -f gearnet_atom.log
```

The runners generate seed-specific temporary YAML files, call `script/downstream.py`, select the best validation checkpoint, evaluate the test split, and aggregate the five results. The current formal runs train from scratch with fixed model seed `1024`; the five reported runs differ by MMseqs2 partition. Temporary checkpoints may be deleted after their metrics are safely written.

</details>

<details>
<summary><strong>6. Frozen ESM2-3B + MLP head</strong></summary>

Validate inputs:

```bash
python -u baselines/ESM2/check_inputs.py \
  --split-root data/mmseqs_seeds_splits \
  --fasta-dir data/mmseqs/fasta
```

Extract one frozen embedding for every partner sequence:

```bash
mkdir -p software/huggingface results/runtime/esm2/cache results/runtime/esm2/logs
export HF_HOME="$(pwd)/software/huggingface"

nohup python -u baselines/ESM2/extract_esm2_embeddings.py \
  --model facebook/esm2_t36_3B_UR50D \
  --fasta-dir data/mmseqs/fasta \
  --split-root data/mmseqs_seeds_splits \
  --output results/runtime/esm2/cache/esm2_t36_3B_embeddings.pt \
  --device cuda:0 \
  --dtype float16 \
  --checkpoint-every 25 \
  --seed 1024 \
  > results/runtime/esm2/logs/extract_esm2_3b.log 2>&1 &

tail -f results/runtime/esm2/logs/extract_esm2_3b.log
```

Validate the completed cache:

```bash
python -u baselines/ESM2/check_inputs.py \
  --split-root data/mmseqs_seeds_splits \
  --fasta-dir data/mmseqs/fasta \
  --embedding-cache results/runtime/esm2/cache/esm2_t36_3B_embeddings.pt
```

Run the formal MLP-head experiment:

```bash
mkdir -p results/runtime/esm2/head results/runtime/esm2/logs

nohup python -u baselines/ESM2/run_five_splits.py \
  --split-root data/mmseqs_seeds_splits \
  --embeddings results/runtime/esm2/cache/esm2_t36_3B_embeddings.pt \
  --output-root results/runtime/esm2/head \
  --model-seed 1024 \
  --head mlp \
  --hidden-dims 512 128 \
  --dropout 0.2 \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --scheduler-factor 0.5 \
  --scheduler-patience 10 \
  --early-stop-patience 30 \
  --device cuda:0 \
  > results/runtime/esm2/logs/esm2_head_console.log 2>&1 &

tail -f results/runtime/esm2/logs/esm2_head_console.log
```

Resume after interruption by repeating the command with `--resume`. A completed seed is skipped; an incomplete seed restarts from epoch 1. The reported benchmark uses the MLP-head result, not the linear diagnostic run.

</details>

<details>
<summary><strong>7. FoldX</strong></summary>

Expected local installation:

```text
software/foldx/foldx5/
├── foldx_20261231
└── molecules/
```

```bash
chmod +x software/foldx/foldx5/foldx_20261231

# Three-structure smoke test.
python -u baselines/FoldX/run_foldx_batch.py --workers 3 --limit 3

# Full resumable batch.
mkdir -p results/runtime/foldx results/summaries/foldx
nohup python -u baselines/FoldX/run_foldx_batch.py \
  --workers 10 \
  > results/runtime/foldx/foldx_batch.log 2>&1 &
echo $! > results/runtime/foldx/foldx_batch.pid

tail -f results/runtime/foldx/foldx_batch.log
python baselines/FoldX/run_foldx_batch.py --status-only
```

The runner performs `RepairPDB` followed by `AnalyseComplex` with chains `A,B`. It stores one durable `DONE.json` per structure, so the same full command safely resumes an interrupted run.

After all 1,243 structures finish:

```bash
python -u baselines/FoldX/analyze_foldx_mmseqs.py \
  2>&1 | tee results/summaries/foldx/foldx_analysis.log
```

</details>

<details>
<summary><strong>8. Rosetta InterfaceAnalyzer</strong></summary>

Expected local installation:

```text
software/rosetta/current/
├── database/
└── source/bin/InterfaceAnalyzer.static.linuxgccrelease
```

```bash
chmod +x software/rosetta/current/source/bin/InterfaceAnalyzer.static.linuxgccrelease

# Three-structure smoke test.
python -u baselines/Rosetta/run_rosetta_batch.py --workers 3 --limit 3

# Full resumable batch.
mkdir -p results/runtime/rosetta results/summaries/rosetta
nohup python -u baselines/Rosetta/run_rosetta_batch.py \
  --workers 10 \
  --job-timeout 3600 \
  > results/runtime/rosetta/rosetta_batch.log 2>&1 &
echo $! > results/runtime/rosetta/rosetta_batch.pid

tail -f results/runtime/rosetta/rosetta_batch.log
python baselines/Rosetta/run_rosetta_batch.py --status-only
```

The default interface is `A_B`. The runner records the validated `4FZV` special case as `A_bB`. Rosetta uses `ref2015`, `pack_input=true`, and `pack_separated=true`. Completed structures have durable `DONE.json` markers and are skipped on resume.

After all 1,243 structures finish:

```bash
python -u baselines/Rosetta/analyze_rosetta_mmseqs.py \
  2>&1 | tee results/summaries/rosetta/rosetta_analysis.log
```

</details>

## Interpreting the results

The following are **plausible explanations**, not conclusions established by the present benchmark. Confirming them requires controlled ablations, repeated model initializations, and per-complex error analysis.

| Method | Observed behavior | Plausible interpretation |
|---|---|---|
| ProAffinity | Modest correlation and the largest mean errors and variability | Its graph/sequence pipeline may be sensitive to preprocessing and may benefit less once close sequence relationships are separated. |
| Graphomer | Best mean value for all four metrics | Pretrained features plus global, structure-biased attention may offer the best bias–capacity balance for this dataset. |
| GearNet-Res | Modest correlation with relatively high error | Residue graphs are compact, but training a large geometric network from scratch on 994 labels may limit generalization. |
| GearNet-Atom | Weakest correlations, but lower RMSE/MAE than GearNet-Res | Dense atom graphs may lose ranking signal through oversmoothing or pooling while predictions shrink toward the label mean. |
| Frozen ESM2-3B + MLP | Second-best correlations without explicit 3D geometry | Large-scale sequence pretraining transfers well and may compensate for the small supervised dataset. |
| FoldX | Modest Pearson but comparatively useful rank correlation | The energy contains real interface signal, although its scale is not a direct heterogeneous-assay pKd scale. |
| Rosetta InterfaceAnalyzer | Positive but weaker correlation with many extreme or positive energies | Static separated-state scoring and repacking may be more sensitive to structural preparation and interface-specific artifacts. |

### Overall comparison

1. **Graphomer performs best.** Its pretrained sequence features, structural encodings, and global attention may provide a useful balance between prior biological knowledge and complex-level geometry. The result does not prove that global attention itself is the cause.
2. **Frozen ESM2-3B ranks second despite using no explicit atomic interface model.** This suggests that sequence and protein-family information remains strongly predictive even after the 30% MMseqs2 split. It may also mean the structural models are not extracting their extra geometric information efficiently from only 994 training complexes.
3. **All methods vary across partitions.** The five seeds change group membership and test-set difficulty, not merely model randomness. Therefore, a large standard deviation can reflect heterogeneous protein families and label distributions rather than unstable optimization alone.

### Why might GearNet-Atom underperform GearNet-Res?

- Atom graphs contain far more nodes and edges. Spatial and KNN edges can make them very dense, increasing optimization difficulty and encouraging oversmoothing.
- Complex-level mean pooling can dilute a sparse binding-interface signal with thousands of non-interface atoms. Residue graphs provide a smaller and less noisy representation.
- Atom-level detail increases sample complexity, but the dataset still provides only 994 training labels per split. More resolution does not automatically create more supervision.
- The atom model was trained from scratch. Hyperparameters, normalization, receptive fields, and pooling that are acceptable for residue graphs may not be appropriate for atom graphs.
- Atom has **lower correlations but better RMSE/MAE than GearNet-Res**. This pattern is consistent with prediction shrinkage toward the training mean: absolute errors improve while ranking information is lost. Prediction variance should be checked before claiming that the atom representation is uniformly worse.

Useful follow-ups are interface-only atom graphs, attention or weighted pooling, sparser edges, removal of graph-level BatchNorm at tiny batch sizes, prediction-distribution plots, and a pretrained atom encoder.

### Why are FoldX and Rosetta correlations modest?

- Experimental pKd reflects solvent, entropy, protonation, ions, conformational changes, temperature, and assay conditions. A single static PDB energy cannot represent all of these effects.
- FoldX and Rosetta score functions were designed mainly for structural energetics and relative comparisons, not direct absolute pKd prediction across heterogeneous complexes.
- Processed structures, missing residues, unusual `UNK/HETATM` records, interface definitions, and repacking choices can introduce structure-specific errors.
- Raw energies depend on interface size and composition. A single affine calibration can correct scale and offset, but not nonlinear or protein-family-specific bias.
- Extreme scores have substantial leverage. FoldX produced nine global three-IQR energy extremes; Rosetta produced more and also returned many positive `dG_separated` values.

The physics baselines are nevertheless not signal-free: FoldX reaches `Rs = 0.2468`, higher than several learned baselines in this experiment. FoldX and Rosetta values are also not directly comparable because FoldX reports kcal/mol-like interaction energy while Rosetta reports Rosetta Energy Units.

### What can and cannot be concluded

- On these five MMseqs2 partitions, Graphomer has the strongest mean result and GearNet-Atom the weakest correlations.
- The experiment compares complete pipelines, not isolated architectures. Pretraining, preprocessing, parameter count, pooling, optimization, and input resolution all differ.
- Five partitions are too few to establish statistical superiority from the mean alone. Paired per-complex tests, bootstrap confidence intervals, and controlled ablations are needed for stronger claims.
- These MMseqs2 results must not be compared with earlier random-split numbers as if the evaluation difficulty were identical.

## Reproducibility rules

- Select checkpoints using validation data only.
- Fit FoldX/Rosetta calibration using training data only.
- Never use test labels for tuning, checkpoint selection, filtering, or calibration.
- Keep the five MMseqs2 split CSVs unchanged when comparing methods.
- Preserve final logs, predictions, split hashes, and compact summaries in `results/summaries/`.
- Keep raw structures, embeddings, checkpoints, job folders, licensed binaries, and large software caches out of Git.
