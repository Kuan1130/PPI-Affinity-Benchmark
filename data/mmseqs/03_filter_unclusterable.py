#!/usr/bin/env python3
"""Remove PPI entries containing non-standard partner sequences."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_EXCLUSIONS = {"3kv4", "4ft4"}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-final-ppi", type=int, default=1243)
    parser.add_argument("--expected-final-sequences", type=int, default=2486)
    return parser.parse_args()


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(parts).upper()))
            current_id = line[1:].split()[0]
            parts = []
        else:
            if current_id is None:
                raise ValueError(f"Sequence before first FASTA header: {path}")
            parts.append(line)
    if current_id is not None:
        records.append((current_id, "".join(parts).upper()))
    if len(records) != len({record[0] for record in records}):
        raise ValueError(f"Duplicate FASTA IDs: {path}")
    return records


def csv_bytes(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def fasta_bytes(records: list[tuple[str, str]]) -> bytes:
    lines: list[str] = []
    for sequence_id, sequence in records:
        lines.append(f">{sequence_id}")
        lines.extend(sequence[start : start + 80] for start in range(0, len(sequence), 80))
    return ("\n".join(lines) + "\n").encode("utf-8")


def atomic_publish(path: Path, content: bytes) -> str:
    if path.exists() and path.read_bytes() == content:
        return "UNCHANGED"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "WROTE"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    work_dir = root / "data/mmseqs"
    source_index = work_dir / "usable_index_before_unknown_filter.csv"
    source_fasta = work_dir / "all_proteins_before_unknown_filter.fasta"
    final_index = work_dir / "usable_index.csv"
    final_fasta = work_dir / "all_proteins.fasta"
    excluded_path = work_dir / "excluded_unclusterable_sequence.csv"

    with source_index.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source_index}")
        fields = list(reader.fieldnames)
        rows = list(reader)
    records = read_fasta(source_fasta)
    sequence_by_id = {sequence_id.lower(): sequence for sequence_id, sequence in records}

    detected: dict[str, list[str]] = {}
    for sequence_id, sequence in records:
        bad = sorted(set(sequence) - STANDARD_AA)
        if bad:
            pdb = sequence_id.rsplit("_", 1)[0].lower()
            detected.setdefault(pdb, []).append(
                f"{sequence_id}: non-standard residues {''.join(bad)}"
            )
    if set(detected) != EXPECTED_EXCLUSIONS:
        raise RuntimeError(
            f"Detected unclusterable PPI set={sorted(detected)}; expected "
            f"{sorted(EXPECTED_EXCLUSIONS)}. No output changed."
        )

    kept_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        if pdb in detected:
            excluded_rows.append({**row, "exclusion_reason": "; ".join(detected[pdb])})
        else:
            kept_rows.append(row)
    kept_records = [
        record
        for record in records
        if record[0].rsplit("_", 1)[0].lower() not in detected
    ]

    if len(kept_rows) != args.expected_final_ppi:
        raise RuntimeError(f"Final PPI rows={len(kept_rows)}; expected {args.expected_final_ppi}")
    if len(kept_records) != args.expected_final_sequences:
        raise RuntimeError(
            f"Final sequences={len(kept_records)}; expected {args.expected_final_sequences}"
        )
    for row in kept_rows:
        pdb = row["pdb_code"].strip().lower()
        if f"{pdb}_1" not in sequence_by_id or f"{pdb}_2" not in sequence_by_id:
            raise ValueError(f"Final index references missing sequence pair: {pdb}")

    publications = [
        (final_index, csv_bytes(fields, kept_rows)),
        (final_fasta, fasta_bytes(kept_records)),
        (excluded_path, csv_bytes(fields + ["exclusion_reason"], excluded_rows)),
    ]
    for path, content in publications:
        print(f"{atomic_publish(path, content)}: {path}")

    print(f"Excluded PPI: {sorted(detected)}")
    print(f"Final PPI / sequences: {len(kept_rows)} / {len(kept_records)}")
    print("UNCLUSTERABLE FILTER PASS")


if __name__ == "__main__":
    main()
