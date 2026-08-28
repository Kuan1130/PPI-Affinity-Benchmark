#!/usr/bin/env python3
"""Create train, validation, and test FASTAs for one split directory."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


EXPECTED_COUNTS = {"train": 1988, "validation": 248, "test": 250}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--split-dir", type=Path, required=True)
    return parser.parse_args()


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                if current in sequences:
                    raise ValueError(f"Duplicate FASTA ID: {current}")
                sequences[current] = "".join(parts)
            current = line[1:].split()[0]
            parts = []
        else:
            if current is None:
                raise ValueError(f"Sequence before FASTA header: {path}")
            parts.append(line)
    if current is not None:
        if current in sequences:
            raise ValueError(f"Duplicate FASTA ID: {current}")
        sequences[current] = "".join(parts)
    return sequences


def fasta_bytes(records: list[tuple[str, str]]) -> bytes:
    lines: list[str] = []
    for sequence_id, sequence in sorted(records):
        lines.append(f">{sequence_id}")
        lines.extend(sequence[start : start + 80] for start in range(0, len(sequence), 80))
    return ("\n".join(lines) + "\n").encode("utf-8")


def publish_consistently(outputs: dict[Path, bytes]) -> None:
    conflicts = [
        path
        for path, content in outputs.items()
        if path.exists()
        and path.read_text(encoding="utf-8").splitlines()
        != content.decode("utf-8").splitlines()
    ]
    if conflicts:
        raise RuntimeError(
            "Existing split FASTA conflicts with deterministic regeneration. "
            "No file was changed: " + ", ".join(str(path) for path in conflicts)
        )
    for path, content in outputs.items():
        if path.exists():
            print(f"UNCHANGED: {path}")
            continue
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"WROTE: {path}")


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    split_dir = args.split_dir
    if not split_dir.is_absolute():
        split_dir = root / "data/mmseqs_seeds_splits" / split_dir
    assignment_path = split_dir / "split_assignments.csv"
    fasta_path = root / "data/mmseqs/all_proteins.fasta"

    pdb_split: dict[str, str] = {}
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb = row["pdb_code"].strip().lower()
            split = row["split"].strip()
            if pdb in pdb_split:
                raise ValueError(f"Duplicate PDB assignment: {pdb}")
            if split not in EXPECTED_COUNTS:
                raise ValueError(f"Unknown split name for {pdb}: {split}")
            pdb_split[pdb] = split

    sequences = read_fasta(fasta_path)
    split_sequences: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for sequence_id, sequence in sequences.items():
        pdb = sequence_id.rsplit("_", 1)[0].lower()
        if pdb not in pdb_split:
            raise ValueError(f"No split assignment for sequence: {sequence_id}")
        split_sequences[pdb_split[pdb]].append((sequence_id, sequence))

    observed = {split: len(split_sequences[split]) for split in EXPECTED_COUNTS}
    if observed != EXPECTED_COUNTS:
        raise RuntimeError(f"Split FASTA counts={observed}; expected={EXPECTED_COUNTS}")

    outputs = {
        split_dir / f"{split}.fasta": fasta_bytes(split_sequences[split])
        for split in EXPECTED_COUNTS
    }
    publish_consistently(outputs)
    print(
        "Train / validation / test sequences: "
        f"{observed['train']} / {observed['validation']} / {observed['test']}"
    )
    print("SPLIT FASTA PASS")


if __name__ == "__main__":
    main()
