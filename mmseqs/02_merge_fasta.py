#!/usr/bin/env python3
"""Merge partner FASTA files and exclude PPIs with missing or empty inputs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
DEFAULT_INDEX = DATA_DIR / "MCGLPPI_RawData" / "PDBBINDdimer_strict_index.csv"
DEFAULT_FASTA_DIR = SCRIPT_DIR / "fasta"
DEFAULT_OUTPUT_FASTA = SCRIPT_DIR / "all_proteins.fasta"
DEFAULT_OUTPUT_INDEX = SCRIPT_DIR / "usable_index.csv"
DEFAULT_EXCLUDED_INDEX = SCRIPT_DIR / "excluded_missing_fasta.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--fasta-dir", type=Path, default=DEFAULT_FASTA_DIR)
    parser.add_argument("--output-fasta", type=Path, default=DEFAULT_OUTPUT_FASTA)
    parser.add_argument("--output-index", type=Path, default=DEFAULT_OUTPUT_INDEX)
    parser.add_argument(
        "--excluded-index", type=Path, default=DEFAULT_EXCLUDED_INDEX
    )
    return parser.parse_args()


def read_fasta_sequence(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    sequence = "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith(">")
    ).upper()
    return re.sub(r"\s+", "", sequence)


def main() -> None:
    args = parse_args()
    index_path = args.index.resolve()
    fasta_dir = args.fasta_dir.resolve()
    output_fasta = args.output_fasta.resolve()
    output_index = args.output_index.resolve()
    excluded_index = args.excluded_index.resolve()

    if not index_path.is_file():
        raise FileNotFoundError(f"Index CSV not found: {index_path}")
    if not fasta_dir.is_dir():
        raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")

    fasta_files = {path.name.lower(): path for path in fasta_dir.glob("*.fasta")}

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column in {index_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    usable_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    sequences: list[tuple[str, str]] = []

    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        partner_paths = [
            fasta_files.get(f"{pdb}_1.fasta"),
            fasta_files.get(f"{pdb}_2.fasta"),
        ]
        missing = [
            f"{pdb}_{partner}.fasta"
            for partner, path in enumerate(partner_paths, start=1)
            if path is None
        ]

        if missing:
            excluded_row = dict(row)
            excluded_row["exclusion_reason"] = "missing FASTA: " + ", ".join(
                missing
            )
            excluded_rows.append(excluded_row)
            continue

        seq_1 = read_fasta_sequence(partner_paths[0])
        seq_2 = read_fasta_sequence(partner_paths[1])
        if not seq_1 or not seq_2:
            excluded_row = dict(row)
            excluded_row["exclusion_reason"] = "empty FASTA sequence"
            excluded_rows.append(excluded_row)
            continue

        usable_rows.append(row)
        sequences.extend([(f"{pdb}_1", seq_1), (f"{pdb}_2", seq_2)])

    for path in (output_fasta, output_index, excluded_index):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Canonical headers replace any legacy "Fake Protein" FASTA headers.
    with output_fasta.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence_id, sequence in sequences:
            handle.write(f">{sequence_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

    with output_index.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(usable_rows)

    with excluded_index.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames + ["exclusion_reason"]
        )
        writer.writeheader()
        writer.writerows(excluded_rows)

    print(f"Raw PPI rows: {len(rows)}")
    print(f"Retained PPI rows: {len(usable_rows)}")
    print(f"Excluded PPI rows: {len(excluded_rows)}")
    print(f"Merged FASTA sequences: {len(sequences)}")
    print("\nOutput files:")
    print(output_fasta)
    print(output_index)
    print(excluded_index)


if __name__ == "__main__":
    main()
