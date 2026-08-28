#!/usr/bin/env python3
"""Check whether each PPI row has two valid partner FASTA files.

Default paths are derived from this script's location:

    data/mmseqs/01_check_fasta.py
    data/mmseqs/fasta/
    data/MCGLPPI_RawData/PDBBINDdimer_strict_index.csv

No file is modified by this check.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
DEFAULT_INDEX = DATA_DIR / "MCGLPPI_RawData" / "PDBBINDdimer_strict_index.csv"
DEFAULT_FASTA_DIR = SCRIPT_DIR / "fasta"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--fasta-dir", type=Path, default=DEFAULT_FASTA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.index.resolve()
    fasta_dir = args.fasta_dir.resolve()

    if not index_path.is_file():
        raise FileNotFoundError(f"Index CSV not found: {index_path}")
    if not fasta_dir.is_dir():
        raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")

    # Use a case-insensitive index so the same data works on Windows and WSL.
    fasta_files: dict[str, Path] = {}
    duplicate_names: list[tuple[str, Path, Path]] = []
    for path in fasta_dir.glob("*.fasta"):
        key = path.name.lower()
        if key in fasta_files:
            duplicate_names.append((key, fasta_files[key], path))
        else:
            fasta_files[key] = path

    missing: list[str] = []
    invalid: list[tuple[str, list[str]]] = []
    empty: list[str] = []
    lengths: list[tuple[str, int, int]] = []
    matched_files: set[str] = set()

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column in {index_path}")
        rows = list(reader)

    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        for partner in (1, 2):
            expected_name = f"{pdb}_{partner}.fasta"
            path = fasta_files.get(expected_name.lower())
            if path is None:
                missing.append(expected_name)
                continue

            matched_files.add(path.name.lower())
            lines = path.read_text(encoding="utf-8").splitlines()
            sequence = "".join(
                line.strip()
                for line in lines
                if line.strip() and not line.lstrip().startswith(">")
            ).upper()
            sequence = re.sub(r"\s+", "", sequence)

            if not sequence:
                empty.append(path.name)
                continue

            bad_chars = sorted(set(sequence) - VALID_AA)
            if bad_chars:
                invalid.append((path.name, bad_chars))
            lengths.append((pdb, partner, len(sequence)))

    extra = sorted(set(fasta_files) - matched_files)

    print(f"Index CSV: {index_path}")
    print(f"FASTA directory: {fasta_dir}")
    print(f"PPI rows: {len(rows)}")
    print(f"Expected FASTA files: {len(rows) * 2}")
    print(f"FASTA files found: {len(fasta_files)}")
    print(f"Matched FASTA files: {len(matched_files)}")
    print(f"Missing FASTA files: {len(missing)}")
    print(f"Empty FASTA sequences: {len(empty)}")
    print(f"Files with invalid characters: {len(invalid)}")
    print(f"Extra FASTA files not used by the index: {len(extra)}")
    print(f"Case-insensitive duplicate filenames: {len(duplicate_names)}")

    if lengths:
        values = [item[2] for item in lengths]
        print(f"Shortest sequence: {min(values)}")
        print(f"Longest sequence: {max(values)}")
        print(f"Mean sequence length: {sum(values) / len(values):.2f}")

    print("\nFirst 20 missing files:")
    print(missing[:20])
    print("\nFirst 20 files with invalid characters:")
    print(invalid[:20])
    print("\nFirst 20 extra files:")
    print(extra[:20])

    print("\nAll missing FASTA files:")
    for name in missing:
        print(name)

    print("\nThirty shortest sequences:")
    for pdb, partner, length in sorted(lengths, key=lambda item: item[2])[:30]:
        print(f"{pdb}_{partner}.fasta\tlength={length}")

    missing_pdb = sorted({name.rsplit("_", 1)[0].lower() for name in missing})
    print(f"\nPDB entries with missing FASTA files ({len(missing_pdb)}):")
    print(",".join(missing_pdb))

    if duplicate_names:
        raise RuntimeError(
            "Case-insensitive duplicate FASTA filenames were detected; "
            "resolve them before continuing."
        )


if __name__ == "__main__":
    main()
