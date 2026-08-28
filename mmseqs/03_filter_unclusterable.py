#!/usr/bin/env python3
"""Remove the two known PPIs whose partner sequences cannot be clustered."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_PATH = SCRIPT_DIR / "usable_index.csv"
FASTA_PATH = SCRIPT_DIR / "all_proteins.fasta"
INDEX_BACKUP = SCRIPT_DIR / "usable_index_before_unknown_filter.csv"
FASTA_BACKUP = SCRIPT_DIR / "all_proteins_before_unknown_filter.fasta"
EXCLUDED_OUTPUT = SCRIPT_DIR / "excluded_unclusterable_sequence.csv"

EXCLUSIONS = {
    "3kv4": "partner 2 sequence is X (length 1)",
    "4ft4": "partner 2 sequence is ATRX (length 4, contains X)",
}
EXPECTED_PPI_COUNT = 1243
EXPECTED_SEQUENCE_COUNT = 2486


def read_fasta_records(path: Path) -> list[tuple[str, list[str]]]:
    records: list[tuple[str, list[str]]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, current_lines))
                current_id = line[1:].strip().split()[0]
                current_lines = []
            elif current_id is not None:
                current_lines.append(line)

    if current_id is not None:
        records.append((current_id, current_lines))
    return records


def main() -> None:
    if not INDEX_PATH.is_file() or not FASTA_PATH.is_file():
        raise FileNotFoundError(
            "Run 02_merge_fasta.py first; usable_index.csv or "
            "all_proteins.fasta is missing."
        )

    # Preserve the pre-filter inputs once. Re-running this script is idempotent.
    if not INDEX_BACKUP.exists():
        shutil.copy2(INDEX_PATH, INDEX_BACKUP)
    if not FASTA_BACKUP.exists():
        shutil.copy2(FASTA_PATH, FASTA_BACKUP)

    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column in {INDEX_PATH}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    present_exclusions = {
        row["pdb_code"].strip().lower()
        for row in rows
        if row["pdb_code"].strip().lower() in EXCLUSIONS
    }
    if present_exclusions not in (set(), set(EXCLUSIONS)):
        raise RuntimeError(
            "Only part of the expected exclusion set is present: "
            f"{sorted(present_exclusions)}"
        )

    kept_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        if pdb in EXCLUSIONS:
            excluded_row = dict(row)
            excluded_row["exclusion_reason"] = EXCLUSIONS[pdb]
            excluded_rows.append(excluded_row)
        else:
            kept_rows.append(row)

    if not present_exclusions:
        kept_rows = rows
        if not EXCLUDED_OUTPUT.is_file():
            raise RuntimeError(
                "The index is already filtered, but the exclusion audit CSV is missing."
            )

    if len(kept_rows) != EXPECTED_PPI_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PPI_COUNT} retained PPIs, found {len(kept_rows)}."
        )

    if present_exclusions:
        temporary_index = SCRIPT_DIR / "usable_index.filtered.tmp.csv"
        with temporary_index.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        temporary_index.replace(INDEX_PATH)

        with EXCLUDED_OUTPUT.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames + ["exclusion_reason"]
            )
            writer.writeheader()
            writer.writerows(excluded_rows)

    records = read_fasta_records(FASTA_PATH)
    kept_records = [
        (sequence_id, sequence_lines)
        for sequence_id, sequence_lines in records
        if sequence_id.rsplit("_", 1)[0].lower() not in EXCLUSIONS
    ]
    if len(kept_records) != EXPECTED_SEQUENCE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SEQUENCE_COUNT} retained sequences, "
            f"found {len(kept_records)}."
        )

    if len(kept_records) != len(records):
        temporary_fasta = SCRIPT_DIR / "all_proteins.filtered.tmp.fasta"
        with temporary_fasta.open("w", encoding="utf-8", newline="\n") as handle:
            for sequence_id, sequence_lines in kept_records:
                handle.write(f">{sequence_id}\n")
                for line in sequence_lines:
                    handle.write(line + "\n")
        temporary_fasta.replace(FASTA_PATH)

    print(f"Excluded PPIs in this run: {len(excluded_rows)}")
    print(f"Retained PPIs: {len(kept_rows)}")
    print(f"Retained FASTA sequences: {len(kept_records)}")
    print(f"Exclusion audit: {EXCLUDED_OUTPUT}")
    print(f"Pre-filter index backup: {INDEX_BACKUP}")
    print(f"Pre-filter FASTA backup: {FASTA_BACKUP}")


if __name__ == "__main__":
    main()
