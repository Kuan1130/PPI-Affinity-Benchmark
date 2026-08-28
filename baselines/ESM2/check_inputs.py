#!/usr/bin/env python3
"""Validate MMseqs split CSVs, partner FASTAs, labels, and optional ESM2 cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from esmppi.io import (
    DEFAULT_SPLIT_SEEDS,
    discover_fastas,
    read_seed_split,
    read_single_fasta,
    required_partner_ids,
    sequence_sha256,
    validate_disjoint_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument("--partner-suffixes", nargs=2, default=("_1", "_2"))
    parser.add_argument(
        "--expected-counts", nargs=3, type=int, default=(994, 124, 125)
    )
    parser.add_argument("--skip-expected-counts", action="store_true")
    return parser.parse_args()


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to inspect --embedding-cache") from error
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    args = parse_args()
    fasta_mapping = discover_fastas(args.fasta_dir)
    expected = dict(zip(("train", "val", "test"), args.expected_counts))
    all_frames = []
    labels_by_pdb: dict[str, float] = {}

    print("Split summary")
    for seed in args.seeds:
        frames = read_seed_split(args.split_root, seed)
        validate_disjoint_splits(frames, context=seed)
        counts = {name: len(frame) for name, frame in frames.items()}
        unique = len(set().union(*(set(frame["pdb_code"]) for frame in frames.values())))
        print(
            f"  {seed}: train={counts['train']}, val={counts['val']}, "
            f"test={counts['test']}, unique={unique}"
        )
        if not args.skip_expected_counts and counts != expected:
            raise ValueError(f"{seed}: expected {expected}, found {counts}")
        for frame in frames.values():
            all_frames.append(frame)
            for row in frame.itertuples(index=False):
                old = labels_by_pdb.get(row.pdb_code)
                label = float(row.proaffinity_label)
                if old is not None and not np.isclose(old, label, rtol=0, atol=1e-10):
                    raise ValueError(
                        f"Inconsistent label for {row.pdb_code}: {old} vs {label}"
                    )
                labels_by_pdb[row.pdb_code] = label

    suffixes = tuple(args.partner_suffixes)
    partner_ids = required_partner_ids(all_frames, suffixes)
    missing_fastas = sorted(partner_ids - set(fasta_mapping))
    if missing_fastas:
        raise FileNotFoundError(
            f"Missing {len(missing_fastas)} required partner FASTAs, e.g. {missing_fastas[:10]}"
        )

    sequences: dict[str, str] = {}
    lengths = []
    for partner_id in sorted(partner_ids):
        sequence = read_single_fasta(fasta_mapping[partner_id])
        sequences[partner_id] = sequence
        lengths.append(len(sequence))
    print(
        f"FASTA summary: partners={len(partner_ids)}, min_len={min(lengths)}, "
        f"median_len={float(np.median(lengths)):.1f}, max_len={max(lengths)}"
    )

    if args.embedding_cache:
        if not args.embedding_cache.is_file():
            raise FileNotFoundError(f"Missing embedding cache: {args.embedding_cache}")
        cache = safe_torch_load(args.embedding_cache)
        records = cache.get("records", {})
        missing_embeddings = sorted(partner_ids - set(records))
        if missing_embeddings:
            raise ValueError(
                f"Embedding cache misses {len(missing_embeddings)} partners, "
                f"e.g. {missing_embeddings[:10]}"
            )
        dimensions = set()
        for partner_id in partner_ids:
            record = records[partner_id]
            digest = sequence_sha256(sequences[partner_id])
            if record.get("sequence_sha256") != digest:
                raise ValueError(f"Stale embedding for {partner_id}: sequence hash differs")
            embedding = record["embedding"]
            if embedding.ndim != 1:
                raise ValueError(
                    f"{partner_id}: expected a 1-D pooled embedding, found {embedding.shape}"
                )
            dimensions.add(int(embedding.numel()))
            if not bool(embedding.isfinite().all()):
                raise ValueError(f"{partner_id}: embedding contains NaN/inf")
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent embedding dimensions: {sorted(dimensions)}")
        print(
            f"Embedding summary: model={cache.get('model')}, "
            f"partners={len(partner_ids)}, dimension={dimensions.pop()}"
        )

    print(
        f"PASS: {len(labels_by_pdb)} unique PPI samples have consistent labels and "
        "complete partner inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

