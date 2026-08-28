#!/usr/bin/env python3
"""Extract one frozen ESM2 mean-pooled embedding per partner FASTA file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from esmppi.io import (
    DEFAULT_SPLIT_SEEDS,
    atomic_json_dump,
    discover_fastas,
    read_seed_split,
    read_single_fasta,
    required_partner_ids,
    sequence_sha256,
    set_global_seed,
)


FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("esm2_embeddings.pt"))
    parser.add_argument(
        "--model",
        default="facebook/esm2_t36_3B_UR50D",
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face model revision; the resolved commit is recorded in metadata",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        help="Optional five-seed split root; when set, extract only required partner IDs",
    )
    parser.add_argument(
        "--seeds", nargs="+", default=list(DEFAULT_SPLIT_SEEDS)
    )
    parser.add_argument("--partner-suffixes", nargs=2, default=("_1", "_2"))
    parser.add_argument(
        "--max-residues",
        type=int,
        default=1022,
        help="Maximum residues per ESM2 forward pass; longer proteins are chunked",
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing cache instead of resuming it",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    result = getattr(torch, requested)
    if device.type == "cpu" and result == torch.float16:
        raise ValueError("float16 is not supported reliably on CPU; use float32 or bfloat16")
    return result


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "model": str(args.model),
        "requested_revision": str(args.revision),
        "resolved_revision": None,
        "pooling": "mean_residue_excluding_special_tokens",
        "long_sequence_policy": "non_overlapping_chunks_then_length_weighted_mean",
        "max_residues": int(args.max_residues),
        "partner_suffixes": list(args.partner_suffixes),
        "hidden_size": None,
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "records": {},
    }


def load_or_initialize_cache(args: argparse.Namespace) -> dict[str, Any]:
    if not args.output.exists() or args.overwrite:
        return build_cache(args)
    cache = safe_torch_load(args.output)
    required = {
        "format_version": FORMAT_VERSION,
        "model": str(args.model),
        "requested_revision": str(args.revision),
        "max_residues": int(args.max_residues),
    }
    mismatches = {
        key: (cache.get(key), value)
        for key, value in required.items()
        if cache.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing cache {args.output} is incompatible: {mismatches}. "
            "Use a different --output path or pass --overwrite."
        )
    if not isinstance(cache.get("records"), dict):
        raise ValueError(f"Existing cache {args.output} has no valid records dictionary")
    return cache


def select_fastas(args: argparse.Namespace) -> dict[str, Path]:
    mapping = discover_fastas(args.fasta_dir)
    if args.split_root is None:
        return mapping

    frames = []
    for seed in args.seeds:
        frames.extend(read_seed_split(args.split_root, seed).values())
    required = required_partner_ids(frames, tuple(args.partner_suffixes))
    missing = sorted(required - set(mapping))
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} FASTA partners required by split CSVs, e.g. {missing[:10]}"
        )
    return {key: mapping[key] for key in sorted(required)}


def chunk_sequence(sequence: str, max_residues: int) -> list[str]:
    if max_residues <= 0:
        raise ValueError("max_residues must be positive")
    return [
        sequence[start : start + max_residues]
        for start in range(0, len(sequence), max_residues)
    ]


@torch.inference_mode()
def embed_sequence(
    sequence: str,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_residues: int,
) -> torch.Tensor:
    weighted_sum: torch.Tensor | None = None
    total_residues = 0

    for chunk in chunk_sequence(sequence, max_residues):
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
            truncation=False,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded).last_hidden_state
        residue_mask = encoded["attention_mask"].bool() & ~special_tokens_mask.bool()
        residue_count = int(residue_mask.sum().item())
        if residue_count != len(chunk):
            raise RuntimeError(
                "Tokenizer did not produce exactly one non-special token per residue: "
                f"expected {len(chunk)}, found {residue_count}"
            )
        chunk_sum = output[residue_mask].float().sum(dim=0).cpu()
        weighted_sum = chunk_sum if weighted_sum is None else weighted_sum + chunk_sum
        total_residues += residue_count

    if weighted_sum is None or total_residues == 0:
        raise RuntimeError("Cannot embed an empty sequence")
    return weighted_sum / total_residues


def write_metadata(cache: dict[str, Any], output: Path) -> None:
    records = cache["records"]
    lengths = [int(record["length"]) for record in records.values()]
    metadata = {
        key: value for key, value in cache.items() if key != "records"
    }
    metadata.update(
        {
            "num_records": len(records),
            "min_sequence_length": min(lengths) if lengths else None,
            "max_sequence_length": max(lengths) if lengths else None,
            "cache_file": str(output.resolve()),
        }
    )
    atomic_json_dump(metadata, output.with_suffix(output.suffix + ".json"))


def main() -> int:
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    fasta_mapping = select_fastas(args)
    cache = load_or_initialize_cache(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hidden_size = int(model.config.hidden_size)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    old_resolved_revision = cache.get("resolved_revision")
    if (
        old_resolved_revision is not None
        and resolved_revision is not None
        and old_resolved_revision != resolved_revision
    ):
        raise ValueError(
            "The model revision resolved to a different commit than the existing cache: "
            f"{old_resolved_revision} vs {resolved_revision}. Use a new cache path."
        )
    cache["resolved_revision"] = resolved_revision or old_resolved_revision
    old_hidden_size = cache.get("hidden_size")
    if old_hidden_size not in (None, hidden_size):
        raise ValueError(
            f"Cache hidden size {old_hidden_size} does not match model hidden size {hidden_size}"
        )
    cache["hidden_size"] = hidden_size

    # Existing identical sequences can be reused even when multiple partner IDs share them.
    by_hash = {
        record["sequence_sha256"]: record["embedding"]
        for record in cache["records"].values()
        if record.get("sequence_sha256") and "embedding" in record
    }

    processed_since_save = 0
    skipped = reused = computed = 0
    for partner_id, fasta_path in tqdm(fasta_mapping.items(), desc="ESM2 embeddings"):
        sequence = read_single_fasta(fasta_path)
        digest = sequence_sha256(sequence)
        existing = cache["records"].get(partner_id)
        if existing and existing.get("sequence_sha256") == digest:
            skipped += 1
            continue

        if digest in by_hash:
            embedding = by_hash[digest].clone()
            reused += 1
        else:
            embedding = embed_sequence(
                sequence, tokenizer, model, device, args.max_residues
            )
            by_hash[digest] = embedding
            computed += 1

        cache["records"][partner_id] = {
            "embedding": embedding.to(dtype=torch.float32, device="cpu"),
            "sequence_sha256": digest,
            "length": len(sequence),
            "source_fasta": str(fasta_path.resolve()),
        }
        processed_since_save += 1
        if processed_since_save >= args.checkpoint_every:
            atomic_torch_save(cache, args.output)
            write_metadata(cache, args.output)
            processed_since_save = 0

    atomic_torch_save(cache, args.output)
    write_metadata(cache, args.output)
    print(
        f"Saved {len(cache['records'])} partner embeddings to {args.output} "
        f"(computed={computed}, reused_by_sequence={reused}, already_cached={skipped}, "
        f"hidden_size={hidden_size}, device={device}, dtype={dtype})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
