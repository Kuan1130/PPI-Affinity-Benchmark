#!/usr/bin/env python3
"""Create the canonical affinity metadata table used by every baseline."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


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


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-count", type=int, default=1270)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated table when its content differs.",
    )
    return parser.parse_args()


def calculate_pkd(value: str) -> float:
    text = "" if value is None else str(value).strip()
    match = AFFINITY_PATTERN.search(text)
    if match is None:
        raise ValueError(f"cannot parse binding_affinity={text!r}")

    number = float(match.group(1))
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"affinity must be positive and finite: {text!r}")

    return -math.log10(number * UNIT_TO_MOLAR[match.group(2)])


def csv_bytes(fieldnames: list[str], rows: list[dict[str, str]]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def publish(path: Path, content: bytes, overwrite: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return "UNCHANGED"
        if not overwrite:
            raise FileExistsError(
                f"Generated output differs from existing file: {path}\n"
                "Inspect it first, then rerun with --overwrite if replacement is intended."
            )

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "WROTE"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    source = root / "data/MCGLPPI_RawData/PDBBINDdimer_strict_index.csv"
    output = root / "data/metadata/ppi_index_labeled.csv"

    if not source.is_file():
        raise FileNotFoundError(f"Missing raw metadata CSV: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required = {"pdb_code", "binding_affinity"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")
    if len(rows) != args.expected_count:
        raise ValueError(
            f"Raw metadata has {len(rows)} rows; expected {args.expected_count}."
        )

    seen: set[str] = set()
    errors: list[str] = []
    labeled_rows: list[dict[str, str]] = []

    for row_number, row in enumerate(rows, start=2):
        pdb = str(row.get("pdb_code", "")).strip().lower()
        if not pdb:
            errors.append(f"row {row_number}: empty pdb_code")
            continue
        if pdb in seen:
            errors.append(f"row {row_number}: duplicate pdb_code={pdb}")
            continue
        seen.add(pdb)

        try:
            label = calculate_pkd(row.get("binding_affinity", ""))
        except ValueError as error:
            errors.append(f"row {row_number} ({pdb}): {error}")
            continue

        labeled = dict(row)
        labeled["pdb_code"] = pdb
        labeled["proaffinity_label"] = format(label, ".12g")
        labeled_rows.append(labeled)

    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(
            f"Label preparation found {len(errors)} error(s):\n{preview}"
        )

    if "proaffinity_label" not in fieldnames:
        fieldnames.append("proaffinity_label")

    status = publish(
        output,
        csv_bytes(fieldnames, labeled_rows),
        overwrite=args.overwrite,
    )
    print(f"Label rows: {len(labeled_rows)}")
    print(f"{status}: {output}")
    print("LABEL PREPARATION PASS")


if __name__ == "__main__":
    main()
