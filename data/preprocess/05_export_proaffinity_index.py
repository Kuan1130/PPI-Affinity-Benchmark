#!/usr/bin/env python3
"""Export ProAffinity-compatible index files from canonical labeled metadata.

Outputs (without headers):

* ``PPIdataindex.txt`` uses a tab delimiter.
* ``PPIdataindex_kd.txt`` uses a single-space delimiter.

The script never changes split membership or affinity labels.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


def repo_root_from_script() -> Path:
    # Expected location: <repo>/data/preprocess/05_export_proaffinity_index.py
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PPIdataindex files required by ProAffinity."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root_from_script(),
        help="Repository root inferred from this script by default.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Canonical labeled metadata CSV. Default: "
            "<repo>/data/metadata/ppi_index_labeled.csv"
        ),
    )
    parser.add_argument(
        "--proaffinity-data-dir",
        type=Path,
        default=None,
        help=(
            "Destination ProAffinity data directory. Default: "
            "<repo>/baselines/Proaffinity/ProAffinity_Test/"
            "ProAffinity-GNN/data"
        ),
    )
    parser.add_argument("--expected-count", type=int, default=1270)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting existing index files after validation.",
    )
    return parser.parse_args()


def render(rows: list[tuple[str, str]], delimiter: str) -> bytes:
    return (
        "".join(f"{pdb}{delimiter}{label}\n" for pdb, label in rows)
    ).encode("utf-8")


def publish(path: Path, content: bytes, overwrite: bool) -> str:
    if path.exists():
        if path.read_bytes() == content:
            return "UNCHANGED"
        if not overwrite:
            raise FileExistsError(
                f"Existing output differs: {path}\n"
                "No file was changed. Inspect it, then rerun with --overwrite "
                "if replacement is intended."
            )

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "WROTE"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    source = (
        args.source.resolve()
        if args.source is not None
        else root / "data/metadata/ppi_index_labeled.csv"
    )
    destination = (
        args.proaffinity_data_dir.resolve()
        if args.proaffinity_data_dir is not None
        else root
        / "baselines/Proaffinity/ProAffinity_Test/ProAffinity-GNN/data"
    )

    if not source.is_file():
        raise FileNotFoundError(f"Missing canonical labeled metadata: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        required = {"pdb_code", "proaffinity_label"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Missing required columns in {source}: {missing}")
        source_rows = list(reader)

    if len(source_rows) != args.expected_count:
        raise ValueError(
            f"Metadata rows={len(source_rows)}; expected {args.expected_count}"
        )

    exported_rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(source_rows, start=2):
        pdb = str(row.get("pdb_code", "")).strip().lower()
        label_text = str(row.get("proaffinity_label", "")).strip()
        if not pdb:
            raise ValueError(f"{source}:{row_number}: empty pdb_code")
        if pdb in seen:
            raise ValueError(f"{source}:{row_number}: duplicate pdb_code={pdb}")
        seen.add(pdb)

        try:
            label = float(label_text)
        except ValueError as error:
            raise ValueError(
                f"{source}:{row_number}: invalid proaffinity_label={label_text!r}"
            ) from error
        if not math.isfinite(label):
            raise ValueError(
                f"{source}:{row_number}: non-finite proaffinity_label={label_text!r}"
            )

        # Preserve the canonical decimal text instead of reformatting the label.
        exported_rows.append((pdb, label_text))

    if not destination.is_dir():
        raise FileNotFoundError(
            f"ProAffinity data directory does not exist: {destination}\n"
            "Move ProAffinity-GNN to its final repository location first, or "
            "pass --proaffinity-data-dir explicitly."
        )

    outputs = {
        destination / "PPIdataindex.txt": render(exported_rows, "\t"),
        destination / "PPIdataindex_kd.txt": render(exported_rows, " "),
    }

    # Validate every conflict before publishing either file.
    conflicts = [
        path
        for path, content in outputs.items()
        if path.exists() and path.read_bytes() != content
    ]
    if conflicts and not args.overwrite:
        raise FileExistsError(
            "Existing ProAffinity index files differ. No file was changed: "
            + ", ".join(str(path) for path in conflicts)
            + "\nInspect them, then rerun with --overwrite if replacement is intended."
        )

    for path, content in outputs.items():
        print(f"{publish(path, content, overwrite=args.overwrite)}: {path}")

    print(f"Exported rows: {len(exported_rows)}")
    print("PROAFFINITY INDEX EXPORT PASS")


if __name__ == "__main__":
    main()
