#!/usr/bin/env python3
"""Audit partner FASTA coverage before building the MMseqs database."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-raw-ppi", type=int, default=1270)
    parser.add_argument("--expected-usable-ppi", type=int, default=1245)
    return parser.parse_args()


def fasta_sequence(path: Path) -> str:
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    return re.sub(r"\s+", "", sequence)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    metadata = root / "data/metadata/ppi_index_labeled.csv"
    fasta_dir = root / "data/mmseqs/fasta"
    audit_path = root / "data/mmseqs/fasta_check.json"

    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column: {metadata}")
        rows = list(reader)
    if len(rows) != args.expected_raw_ppi:
        raise ValueError(f"Raw PPI rows={len(rows)}; expected {args.expected_raw_ppi}")

    fasta_index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in fasta_dir.glob("*.fasta"):
        key = path.name.casefold()
        if key in fasta_index:
            duplicates.append(f"{fasta_index[key].name} / {path.name}")
        fasta_index[key] = path
    if duplicates:
        raise ValueError("Case-insensitive duplicate FASTAs: " + "; ".join(duplicates))

    missing: list[str] = []
    empty: list[str] = []
    invalid: list[dict[str, object]] = []
    matched: set[str] = set()
    lengths: list[int] = []
    usable_pdbs = 0

    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        pair_ok = True
        for partner in (1, 2):
            name = f"{pdb}_{partner}.fasta"
            path = fasta_index.get(name.casefold())
            if path is None:
                missing.append(name)
                pair_ok = False
                continue
            matched.add(path.name.casefold())
            sequence = fasta_sequence(path)
            if not sequence:
                empty.append(path.name)
                pair_ok = False
                continue
            bad = sorted(set(sequence) - VALID_AA)
            if bad:
                invalid.append({"file": path.name, "characters": bad})
                pair_ok = False
            lengths.append(len(sequence))
        if pair_ok:
            usable_pdbs += 1

    extra = sorted(set(fasta_index) - matched)
    missing_pdbs = sorted({name.rsplit("_", 1)[0] for name in missing})
    payload: dict[str, object] = {
        "raw_ppi_rows": len(rows),
        "expected_fasta_files": len(rows) * 2,
        "observed_fasta_files": len(fasta_index),
        "matched_fasta_files": len(matched),
        "usable_ppi_pairs": usable_pdbs,
        "missing_files": missing,
        "missing_pdb_codes": missing_pdbs,
        "empty_files": empty,
        "invalid_files": invalid,
        "extra_files": extra,
        "minimum_length": min(lengths) if lengths else None,
        "maximum_length": max(lengths) if lengths else None,
    }
    atomic_json(audit_path, payload)

    print(f"Raw PPI rows: {len(rows)}")
    print(f"Usable PPI FASTA pairs: {usable_pdbs}")
    print(f"Matched FASTAs: {len(matched)}")
    print(f"PDBs missing at least one FASTA: {len(missing_pdbs)}")
    print(f"Empty / invalid / extra: {len(empty)} / {len(invalid)} / {len(extra)}")
    print(f"Audit: {audit_path}")

    expected_files = args.expected_usable_ppi * 2
    if (
        usable_pdbs != args.expected_usable_ppi
        or len(matched) != expected_files
        or empty
        or invalid
        or extra
    ):
        raise SystemExit("FASTA CHECK FAILED; inspect the JSON audit")
    print("FASTA CHECK PASS")


if __name__ == "__main__":
    main()
