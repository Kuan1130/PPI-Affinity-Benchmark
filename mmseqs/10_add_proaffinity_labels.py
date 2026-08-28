#!/usr/bin/env python3
"""Add pKd-style proaffinity labels to the final MMseqs split CSVs.

This script never changes split membership. It reads the split files produced
by ``06_make_group_split.py`` (``train.csv``, ``validation.csv``, and
``test.csv``), then
writes labeled model-input files named ``train_split.csv``, ``val_split.csv``,
and ``test_split.csv``. The original three split CSVs are left unchanged.

It computes

    proaffinity_label = -log10(binding_affinity in mol/L)

for the final usable index and every train/validation/test CSV, validates that
all 1,243 PDB IDs are present exactly once per split set, and only writes after
the entire dry-run validation succeeds.

Dry run (recommended first):

    python3 data/mmseqs/10_add_proaffinity_labels.py

Apply changes, with automatic one-time backups:

    python3 data/mmseqs/10_add_proaffinity_labels.py --apply

Process only selected split directories:

    python3 data/mmseqs/10_add_proaffinity_labels.py --apply \
        --split-dirs ../mmseqs_seeds_splits/seed_0 ../mmseqs_seeds_splits/seed_4242
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
SPLIT_ROOT = DATA_DIR / "mmseqs_seeds_splits"
DEFAULT_SOURCE = SCRIPT_DIR / "usable_index.csv"
DEFAULT_SEEDS = (0, 1, 42, 142, 4242)
DEFAULT_SPLIT_DIRS = [SPLIT_ROOT / f"seed_{seed}" for seed in DEFAULT_SEEDS]

SPLIT_FILE_SPECS = [
    # logical name, source filename, labeled output filename, expected rows
    ("train", "train.csv", "train_split.csv", 994),
    ("validation", "validation.csv", "val_split.csv", 124),
    ("test", "test.csv", "test_split.csv", 125),
]

UNIT_TO_MOLAR = {
    "fM": 1e-15,
    "pM": 1e-12,
    "nM": 1e-9,
    "uM": 1e-6,
    "µM": 1e-6,
    "μM": 1e-6,
    "mM": 1e-3,
    "M": 1.0,
}

AFFINITY_PATTERN = re.compile(
    r"([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*"
    r"(fM|pM|nM|uM|µM|μM|mM|M)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and add proaffinity_label without changing MMseqs "
            "split membership."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Final usable metadata CSV (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--split-dirs",
        type=Path,
        nargs="+",
        default=DEFAULT_SPLIT_DIRS,
        help="Split directories to label and validate.",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=1243,
        help="Expected final PPI count (default: 1243).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write validated labels. Without this flag, perform a dry run.",
    )
    return parser.parse_args()


def resolve_relative_to_script(path: Path) -> Path:
    """Resolve command-line relative paths from data/mmseqs, not the CWD."""
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def calculate_proaffinity_label(value: str) -> float:
    text = "" if value is None else str(value).strip()
    match = AFFINITY_PATTERN.search(text)

    if match is None:
        raise ValueError(f"cannot parse binding_affinity={text!r}")

    numeric_value = float(match.group(1))
    unit = match.group(2)

    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError(f"affinity must be positive and finite: {text!r}")

    molar_value = numeric_value * UNIT_TO_MOLAR[unit]
    return -math.log10(molar_value)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames

        if fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        rows = list(reader)

    required = {"pdb_code", "binding_affinity"}
    missing_columns = sorted(required - set(fieldnames))

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {missing_columns}"
        )

    return list(fieldnames), rows


def label_rows(
    path: Path,
) -> tuple[list[str], list[dict[str, str]], dict[str, float]]:
    fieldnames, rows = read_csv(path)
    labeled_rows: list[dict[str, str]] = []
    labels_by_pdb: dict[str, float] = {}
    errors: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        pdb = str(row.get("pdb_code", "")).strip().lower()

        if not pdb:
            errors.append(f"{path}:{row_number}: empty pdb_code")
            continue

        if pdb in labels_by_pdb:
            errors.append(f"{path}:{row_number}: duplicate pdb_code={pdb}")
            continue

        try:
            computed_label = calculate_proaffinity_label(
                row.get("binding_affinity", "")
            )
        except ValueError as error:
            errors.append(f"{path}:{row_number} ({pdb}): {error}")
            continue

        existing_text = str(row.get("proaffinity_label", "")).strip()

        if existing_text:
            try:
                existing_label = float(existing_text)
            except ValueError:
                errors.append(
                    f"{path}:{row_number} ({pdb}): invalid existing "
                    f"proaffinity_label={existing_text!r}"
                )
                continue

            if not math.isclose(
                existing_label,
                computed_label,
                rel_tol=1e-10,
                abs_tol=1e-10,
            ):
                errors.append(
                    f"{path}:{row_number} ({pdb}): existing label "
                    f"{existing_label} != computed {computed_label}"
                )
                continue

        labeled_row = dict(row)
        labeled_row["proaffinity_label"] = format(computed_label, ".12g")
        labeled_rows.append(labeled_row)
        labels_by_pdb[pdb] = computed_label

    if errors:
        preview = "\n".join(errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n... {len(errors) - 30} more"
        raise ValueError(
            f"Label validation failed with {len(errors)} error(s):\n"
            f"{preview}{suffix}"
        )

    if "proaffinity_label" not in fieldnames:
        fieldnames.append("proaffinity_label")

    return fieldnames, labeled_rows, labels_by_pdb


def backup_path(path: Path) -> Path:
    return path.with_name(
        f"{path.stem}_before_proaffinity_label{path.suffix}"
    )


def write_csv_atomically(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    backup = backup_path(path)

    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)

    temporary = path.with_name(f"{path.name}.labeling.tmp")

    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_path = resolve_relative_to_script(args.source)

    # Phase 1: validate every input and prepare all outputs in memory.
    write_plan: dict[Path, tuple[list[str], list[dict[str, str]]]] = {}

    source_fields, source_rows, source_labels = label_rows(source_path)
    source_ids = set(source_labels)

    if len(source_rows) != args.expected_total:
        raise ValueError(
            f"{source_path} has {len(source_rows)} rows; "
            f"expected {args.expected_total}"
        )

    write_plan[source_path] = (source_fields, source_rows)

    print(f"Source: {source_path} ({len(source_rows)} rows)")

    for split_dir_value in args.split_dirs:
        split_dir = resolve_relative_to_script(split_dir_value)
        split_ids: dict[str, set[str]] = {}

        print(f"\nChecking {split_dir} ...")

        for logical_name, source_filename, output_filename, expected_count in (
            SPLIT_FILE_SPECS
        ):
            source_split_path = split_dir / source_filename
            output_split_path = split_dir / output_filename

            # Always prefer the original split.py output as the source of
            # membership. If it has been archived later, fall back to an
            # already-created labeled output for idempotent validation.
            if source_split_path.exists():
                input_path = source_split_path
            elif output_split_path.exists():
                input_path = output_split_path
            else:
                raise FileNotFoundError(
                    f"missing both {source_split_path} and {output_split_path}"
                )

            fields, rows, labels = label_rows(input_path)
            ids = set(labels)

            if len(rows) != expected_count:
                raise ValueError(
                    f"{input_path} has {len(rows)} rows; "
                    f"expected {expected_count}"
                )

            for pdb, label in labels.items():
                if pdb not in source_labels:
                    raise ValueError(
                        f"{input_path}: unknown pdb_code={pdb}"
                    )

                if not math.isclose(
                    label,
                    source_labels[pdb],
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        f"{input_path}: label mismatch for pdb_code={pdb}"
                    )

            split_ids[logical_name] = ids
            write_plan[output_split_path] = (fields, rows)
            print(
                f"  {source_filename} -> {output_filename}: "
                f"{len(rows)} rows"
            )

        train_ids = split_ids["train"]
        val_ids = split_ids["validation"]
        test_ids = split_ids["test"]

        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError(f"{split_dir}: train/val/test PDB overlap detected")

        combined_ids = train_ids | val_ids | test_ids

        if combined_ids != source_ids:
            missing = sorted(source_ids - combined_ids)
            extra = sorted(combined_ids - source_ids)
            raise ValueError(
                f"{split_dir}: split membership differs from source; "
                f"missing={missing[:20]}, extra={extra[:20]}"
            )

        assignment_path = split_dir / "split_assignments.csv"

        if assignment_path.exists():
            fields, rows, labels = label_rows(assignment_path)

            if len(rows) != args.expected_total or set(labels) != source_ids:
                raise ValueError(
                    f"{assignment_path}: assignment membership/count mismatch"
                )

            write_plan[assignment_path] = (fields, rows)

        print("  membership: PASS; no overlap; all source PDBs present")

    print(
        f"\nValidation complete: {len(write_plan)} CSV files are consistent."
    )

    if not args.apply:
        print("DRY RUN ONLY: no files were changed.")
        print("Run again with --apply to write labels and create backups.")
        return

    # Phase 2: write only after all files have passed phase 1.
    for path, (fieldnames, rows) in write_plan.items():
        write_csv_atomically(path, fieldnames, rows)
        if backup_path(path).exists():
            print(f"WROTE: {path} (backup: {backup_path(path)})")
        else:
            print(f"CREATED: {path}")

    print("\nPASS: labels added; row order and split membership unchanged.")


if __name__ == "__main__":
    main()
