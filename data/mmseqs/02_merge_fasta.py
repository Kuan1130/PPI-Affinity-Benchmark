#!/usr/bin/env python3
"""Merge complete partner FASTAs and record PPI entries missing either partner."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-raw-ppi", type=int, default=1270)
    parser.add_argument("--expected-usable-ppi", type=int, default=1245)
    return parser.parse_args()


def read_sequence(path: Path) -> str:
    return "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()


def csv_content(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def fasta_content(records: list[tuple[str, str]]) -> bytes:
    lines: list[str] = []
    for sequence_id, sequence in records:
        lines.append(f">{sequence_id}")
        lines.extend(sequence[start : start + 80] for start in range(0, len(sequence), 80))
    return ("\n".join(lines) + "\n").encode("utf-8")


def atomic_publish(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return "UNCHANGED"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "WROTE"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    metadata = root / "data/metadata/ppi_index_labeled.csv"
    fasta_dir = root / "data/mmseqs/fasta"
    work_dir = root / "data/mmseqs"
    output_fasta = work_dir / "all_proteins_before_unknown_filter.fasta"
    output_index = work_dir / "usable_index_before_unknown_filter.csv"
    excluded_path = work_dir / "excluded_missing_fasta.csv"

    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {metadata}")
        fields = list(reader.fieldnames)
        rows = list(reader)
    if len(rows) != args.expected_raw_ppi:
        raise ValueError(f"Raw PPI rows={len(rows)}; expected {args.expected_raw_ppi}")
    if "proaffinity_label" not in fields:
        raise ValueError("Canonical metadata is missing proaffinity_label")

    fasta_index: dict[str, Path] = {}
    for path in fasta_dir.glob("*.fasta"):
        key = path.name.casefold()
        if key in fasta_index:
            raise ValueError(f"Case-insensitive duplicate FASTA: {path.name}")
        fasta_index[key] = path

    usable: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    sequences: list[tuple[str, str]] = []
    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        paths = [fasta_index.get(f"{pdb}_{partner}.fasta".casefold()) for partner in (1, 2)]
        missing = [f"{pdb}_{i}.fasta" for i, path in enumerate(paths, 1) if path is None]
        if missing:
            excluded.append({**row, "exclusion_reason": "missing FASTA: " + ", ".join(missing)})
            continue
        partner_sequences = [read_sequence(path) for path in paths if path is not None]
        if any(not sequence for sequence in partner_sequences):
            excluded.append({**row, "exclusion_reason": "empty FASTA sequence"})
            continue
        usable.append(row)
        sequences.extend(
            [(f"{pdb}_1", partner_sequences[0]), (f"{pdb}_2", partner_sequences[1])]
        )

    if len(usable) != args.expected_usable_ppi or len(sequences) != args.expected_usable_ppi * 2:
        raise RuntimeError(
            f"Usable PPI/sequences={len(usable)}/{len(sequences)}; expected "
            f"{args.expected_usable_ppi}/{args.expected_usable_ppi * 2}. No output changed."
        )

    excluded_fields = fields + ["exclusion_reason"]
    publications = [
        (output_fasta, fasta_content(sequences)),
        (output_index, csv_content(fields, usable)),
        (excluded_path, csv_content(excluded_fields, excluded)),
    ]
    for path, content in publications:
        print(f"{atomic_publish(path, content)}: {path}")

    print(f"Raw / usable / excluded PPI: {len(rows)} / {len(usable)} / {len(excluded)}")
    print(f"Merged sequences: {len(sequences)}")
    print("FASTA MERGE PASS")


if __name__ == "__main__":
    main()
