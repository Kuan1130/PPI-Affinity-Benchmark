#!/usr/bin/env python3
"""Create train, validation, and test FASTA files for one split seed."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FASTA_PATH = SCRIPT_DIR / "all_proteins.fasta"
EXPECTED_SEQUENCE_COUNTS = {
    "train": 1988,
    "validation": 248,
    "test": 250,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Split directory, normally data/mmseqs_seeds_splits/seed_<seed>.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    current_sequence: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    if current_id in sequences:
                        raise ValueError(f"Duplicate FASTA header: {current_id}")
                    sequences[current_id] = "".join(current_sequence)
                current_id = line[1:].split()[0]
                current_sequence = []
            else:
                if current_id is None:
                    raise ValueError("Sequence text appeared before the first FASTA header.")
                current_sequence.append(line)

    if current_id is not None:
        if current_id in sequences:
            raise ValueError(f"Duplicate FASTA header: {current_id}")
        sequences[current_id] = "".join(current_sequence)
    return sequences


def main() -> None:
    args = parse_args()
    split_dir = resolve_path(args.split_dir)
    assignment_path = split_dir / "split_assignments.csv"

    if not FASTA_PATH.is_file():
        raise FileNotFoundError(f"Missing merged FASTA: {FASTA_PATH}")
    if not assignment_path.is_file():
        raise FileNotFoundError(f"Missing split assignment: {assignment_path}")

    pdb_split: dict[str, str] = {}
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"pdb_code", "split"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Missing pdb_code or split column in {assignment_path}")
        for row in reader:
            pdb = row["pdb_code"].strip().lower()
            split = row["split"].strip()
            if pdb in pdb_split:
                raise ValueError(f"Duplicate pdb_code in assignment: {pdb}")
            if split not in EXPECTED_SEQUENCE_COUNTS:
                raise ValueError(f"Unknown split value for {pdb}: {split}")
            pdb_split[pdb] = split

    sequences = read_fasta(FASTA_PATH)
    split_sequences: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

    for sequence_id, sequence in sequences.items():
        pdb = sequence_id.rsplit("_", 1)[0].lower()
        if pdb not in pdb_split:
            raise ValueError(f"No split assignment for sequence: {sequence_id}")
        split_sequences[pdb_split[pdb]].append((sequence_id, sequence))

    for split in ("train", "validation", "test"):
        observed_count = len(split_sequences[split])
        expected_count = EXPECTED_SEQUENCE_COUNTS[split]
        if observed_count != expected_count:
            raise ValueError(
                f"{split} has {observed_count} sequences; expected {expected_count}."
            )

        output_path = split_dir / f"{split}.fasta"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sequence_id, sequence in sorted(split_sequences[split]):
                handle.write(f">{sequence_id}\n")
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")
        print(f"{split}: {observed_count} sequences -> {output_path}")


if __name__ == "__main__":
    main()
