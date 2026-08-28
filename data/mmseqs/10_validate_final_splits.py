#!/usr/bin/env python3
"""Validate final labeled MMseqs split artifacts without modifying any file."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


CANONICAL_SEEDS = [0, 1, 42, 142, 4242]
CSV_SPECS = {
    "train": ("train_split.csv", 994),
    "validation": ("val_split.csv", 124),
    "test": ("test_split.csv", 125),
}
FASTA_SPECS = {"train": 1988, "validation": 248, "test": 250}
UNIT_TO_MOLAR = {
    "fM": 1e-15, "pM": 1e-12, "nM": 1e-9, "uM": 1e-6,
    "µM": 1e-6, "μM": 1e-6, "mM": 1e-3, "M": 1.0,
}
AFFINITY_PATTERN = re.compile(
    r"([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*"
    r"(fM|pM|nM|uM|µM|μM|mM|M)"
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS)
    parser.add_argument("--expected-total", type=int, default=1243)
    return parser.parse_args()


def calculate_pkd(value: str) -> float:
    match = AFFINITY_PATTERN.search("" if value is None else str(value).strip())
    if match is None:
        raise ValueError(f"cannot parse binding_affinity={value!r}")
    number = float(match.group(1))
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid binding affinity={value!r}")
    return -math.log10(number * UNIT_TO_MOLAR[match.group(2)])


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_labels(path: Path, rows: list[dict[str, str]]) -> dict[str, float]:
    labels: dict[str, float] = {}
    for row_number, row in enumerate(rows, start=2):
        pdb = row.get("pdb_code", "").strip().lower()
        if not pdb or pdb in labels:
            raise ValueError(f"{path}:{row_number}: empty or duplicate pdb_code={pdb!r}")
        try:
            observed = float(row.get("proaffinity_label", ""))
        except ValueError as error:
            raise ValueError(f"{path}:{row_number}: invalid proaffinity_label") from error
        expected = calculate_pkd(row.get("binding_affinity", ""))
        if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(
                f"{path}:{row_number}: label mismatch for {pdb}: {observed} != {expected}"
            )
        labels[pdb] = observed
    return labels


def fasta_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing FASTA: {path}")
    return sum(line.startswith(">") for line in path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    source_path = root / "data/mmseqs/usable_index.csv"
    source_fields, source_rows = read_csv(source_path)
    required = {"pdb_code", "binding_affinity", "proaffinity_label"}
    if not required.issubset(source_fields):
        raise ValueError(f"{source_path} is missing columns: {sorted(required - set(source_fields))}")
    if len(source_rows) != args.expected_total:
        raise ValueError(f"Source rows={len(source_rows)}; expected {args.expected_total}")
    source_labels = validate_labels(source_path, source_rows)
    source_ids = set(source_labels)

    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Duplicate seed argument")
    for seed in args.seeds:
        if seed < 0:
            raise ValueError(f"Seed must be non-negative: {seed}")
        split_dir = root / f"data/mmseqs_seeds_splits/seed_{seed}"
        split_ids: dict[str, set[str]] = {}
        print(f"Checking seed_{seed} ...")

        for split, (filename, expected_rows) in CSV_SPECS.items():
            path = split_dir / filename
            fields, rows = read_csv(path)
            if fields != source_fields:
                raise ValueError(f"Columns differ from usable_index.csv: {path}")
            if len(rows) != expected_rows:
                raise ValueError(f"{path}: rows={len(rows)}; expected {expected_rows}")
            labels = validate_labels(path, rows)
            for pdb, label in labels.items():
                if pdb not in source_labels or not math.isclose(
                    label, source_labels[pdb], rel_tol=1e-10, abs_tol=1e-10
                ):
                    raise ValueError(f"Source label mismatch for {pdb} in {path}")
            split_ids[split] = set(labels)

        if (
            split_ids["train"] & split_ids["validation"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["validation"] & split_ids["test"]
        ):
            raise ValueError(f"PDB overlap detected in seed_{seed}")
        if set().union(*split_ids.values()) != source_ids:
            raise ValueError(f"Split membership does not cover the source in seed_{seed}")

        assignment_path = split_dir / "split_assignments.csv"
        assignment_fields, assignment_rows = read_csv(assignment_path)
        if not {"ppi_group", "group_size", "split"}.issubset(assignment_fields):
            raise ValueError(f"Assignment columns missing: {assignment_path}")
        if len(assignment_rows) != args.expected_total:
            raise ValueError(f"Assignment row count is incorrect: {assignment_path}")
        assignment_seen: set[str] = set()
        group_splits: dict[str, set[str]] = {}
        for row in assignment_rows:
            pdb = row["pdb_code"].strip().lower()
            split = row["split"].strip()
            group = row["ppi_group"].strip()
            if pdb in assignment_seen or pdb not in split_ids.get(split, set()):
                raise ValueError(f"Assignment mismatch for {pdb}: {assignment_path}")
            assignment_seen.add(pdb)
            group_splits.setdefault(group, set()).add(split)
        if assignment_seen != source_ids:
            raise ValueError(f"Assignment membership is incomplete: {assignment_path}")
        broken_groups = {group: splits for group, splits in group_splits.items() if len(splits) != 1}
        if broken_groups:
            raise ValueError(f"PPI groups cross splits: {broken_groups}")

        leakage_path = split_dir / "cross_split_leakage.tsv"
        with leakage_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            leakage_rows = list(reader)
        if leakage_rows:
            raise ValueError(f"Direct leakage audit is non-empty: {leakage_path}")

        for split, expected_count in FASTA_SPECS.items():
            observed = fasta_count(split_dir / f"{split}.fasta")
            if observed != expected_count:
                raise ValueError(
                    f"seed_{seed} {split}.fasta sequences={observed}; expected {expected_count}"
                )

        crosscheck_dir = split_dir / "crosscheck"
        crosscheck_files = sorted(crosscheck_dir.glob("*_vs_*.tsv"))
        if len(crosscheck_files) != 6:
            raise ValueError(f"seed_{seed}: crosscheck files={len(crosscheck_files)}; expected 6")
        nonempty = [path for path in crosscheck_files if path.stat().st_size != 0]
        if nonempty:
            raise ValueError(
                f"seed_{seed}: non-empty crosscheck files: "
                + ", ".join(path.name for path in nonempty)
            )

        print("  CSV 994/124/125; FASTA 1988/248/250; leakage 0: PASS")

    print(f"Validated seeds: {', '.join('seed_' + str(seed) for seed in args.seeds)}")
    print("FINAL SPLIT VALIDATION PASS")


if __name__ == "__main__":
    main()
