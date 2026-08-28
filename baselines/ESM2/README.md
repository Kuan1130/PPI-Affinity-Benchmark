# Frozen ESM2 + regression head for PPI affinity

This project implements a sequence-only baseline for protein-protein binding affinity:

1. encode each partner independently with a frozen ESM2 model;
2. mean-pool residue embeddings, excluding BOS/EOS/padding;
3. build an order-invariant pair representation;
4. train only a small regression head;
5. select the checkpoint by validation Pearson correlation;
6. evaluate the test split only after model selection.

The default checkpoint is
[`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D),
loaded through the official Transformers ESM interface. The checkpoint is configurable
and may also be a local directory.

## Fixed protocol

For partner embeddings `h1` and `h2`, the PPI representation is:

```text
[h1 + h2, abs(h1 - h2), h1 * h2]
```

This representation is invariant to exchanging partner 1 and partner 2. The default
head is:

```text
Linear(3d, 512) -> LayerNorm -> GELU -> Dropout(0.2)
Linear(512, 128) -> LayerNorm -> GELU -> Dropout(0.2)
Linear(128, 1)
```

ESM2 is never updated. Labels are standardized with the training split's mean and
population standard deviation only. The head is trained with MSE, while the best
checkpoint is chosen by validation Pearson. `ReduceLROnPlateau` explicitly uses
`mode=max`. The test split is not evaluated during training.

Sequences longer than 1,022 residues are split into non-overlapping chunks. Residue
embeddings are summed within each chunk, then divided by the total sequence length.
This produces one length-weighted mean vector without silently truncating the protein.
Record this policy when reporting results.

## Expected data layout

The paths can be anywhere; they are passed explicitly on the command line.

```text
FASTA/
├── 1abc_1.fasta
├── 1abc_2.fasta
├── 2def_1.fasta
└── 2def_2.fasta

mmseqs_seeds_splits/
├── seed_0/
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── seed_1/
├── seed_42/
├── seed_142/
└── seed_4242/
```

Every CSV must contain at least:

```text
pdb_code,proaffinity_label
```

By default, `pdb_code=1abc` resolves to `1abc_1.fasta` and `1abc_2.fasta`.

## Installation

Use a separate environment or an environment that already has a compatible PyTorch:

```bash
cd /path/to/esm2_ppi_head
python -m pip install -r requirements.txt
```

The 650M model is downloaded on first use unless `--model` points to a local model
directory. To require an already-cached/local model, add `--local-files-only` during
embedding extraction. The extractor records the resolved model commit in its metadata;
for a formally frozen experiment, `--revision` may also be set to an explicit commit.

## 1. Validate CSV and FASTA inputs

```bash
python check_inputs.py \
  --split-root /path/to/mmseqs_seeds_splits \
  --fasta-dir /path/to/FASTA
```

The default expected counts are `994 / 124 / 125`. For another dataset size, use
`--skip-expected-counts` or supply `--expected-counts TRAIN VAL TEST`.

## 2. Extract frozen ESM2 embeddings once

Do not run this concurrently with GearNet on the same GPU unless there is enough free
memory.

```bash
python -u extract_esm2_embeddings.py \
  --fasta-dir /path/to/FASTA \
  --split-root /path/to/mmseqs_seeds_splits \
  --output esm2_t33_650m_embeddings.pt \
  --device cuda:0
```

The extractor saves progress every 25 new partner IDs and resumes automatically from
the same output file. Identical sequences are embedded once and reused. If a FASTA
changes, its SHA-256 changes and that record is recomputed.

If the default model does not fit, a smaller checkpoint can be selected explicitly,
but it is then a different experimental condition and must use a different cache and
result directory:

```bash
python -u extract_esm2_embeddings.py \
  --fasta-dir /path/to/FASTA \
  --split-root /path/to/mmseqs_seeds_splits \
  --model facebook/esm2_t30_150M_UR50D \
  --output esm2_t30_150m_embeddings.pt \
  --device cuda:0
```

After extraction, validate that the cache exactly matches the current FASTAs:

```bash
python check_inputs.py \
  --split-root /path/to/mmseqs_seeds_splits \
  --fasta-dir /path/to/FASTA \
  --embedding-cache esm2_t33_650m_embeddings.pt
```

## 3. Smoke-test one split

The following short run checks mapping, dimensions, training, checkpoint selection,
and output generation. It is not a result to report.

```bash
python train_esm2_head.py \
  --train-csv /path/to/mmseqs_seeds_splits/seed_42/train_split.csv \
  --val-csv /path/to/mmseqs_seeds_splits/seed_42/val_split.csv \
  --test-csv /path/to/mmseqs_seeds_splits/seed_42/test_split.csv \
  --embeddings esm2_t33_650m_embeddings.pt \
  --output-dir smoke_seed_42 \
  --epochs 3 \
  --device cuda:0
```

## 4. Run all five MMseqs splits

The split seed changes dataset membership; the head initialization/training seed is
fixed at `1024` across all five runs.

```bash
nohup python -u run_five_splits.py \
  --split-root /path/to/mmseqs_seeds_splits \
  --embeddings esm2_t33_650m_embeddings.pt \
  --output-root esm2_head_results \
  --device cuda:0 \
  > esm2_head_console.log 2>&1 &
```

Monitor progress:

```bash
tail -f esm2_head_console.log
```

Resume after interruption:

```bash
nohup python -u run_five_splits.py \
  --split-root /path/to/mmseqs_seeds_splits \
  --embeddings esm2_t33_650m_embeddings.pt \
  --output-root esm2_head_results \
  --device cuda:0 \
  --resume \
  > esm2_head_resume.log 2>&1 &
```

## Outputs

Each seed directory contains:

```text
best_head.pt
history.csv
val_predictions.csv
test_predictions.csv
summary.json
train.log
```

The aggregate files are:

```text
esm2_head_results/esm2_head_results.csv
esm2_head_results/esm2_head_summary.json
```

The CSV reports test Pearson, Spearman, RMSE, and MAE for all five split seeds. The
JSON reports their mean and population standard deviation, matching `numpy.std` with
its default `ddof=0` convention.

## Important reporting language

A concise method description is:

> We independently encoded both protein partners using frozen ESM2
> (`esm2_t33_650M_UR50D`) and mean-pooled non-special residue representations.
> We formed an order-invariant pair representation by concatenating the sum,
> absolute difference, and Hadamard product of partner embeddings, then trained a
> two-layer MLP regression head with MSE. Labels were standardized using training
> statistics only, and checkpoints were selected by validation Pearson correlation.

This is a sequence-only trained baseline. It is separate from the zero-shot FoldX and
Rosetta interface-score baselines.
