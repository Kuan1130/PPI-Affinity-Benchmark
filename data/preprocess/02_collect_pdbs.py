#!/usr/bin/env python3
"""Collect one deterministic all-atom PDB per PPI into shared local storage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-count", type=int, default=1270)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_source(folder: Path, pdb: str) -> Path:
    candidates = sorted(
        path
        for path in folder.glob("*.pdb")
        if "-cg" not in path.name.lower()
    )
    exact = [path for path in candidates if path.stem.lower() == pdb]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no all-atom PDB in {folder}")
    raise RuntimeError(
        f"ambiguous all-atom PDB files in {folder}: "
        + ", ".join(path.name for path in candidates)
    )


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = ["pdb_code", "source", "destination", "sha256", "status", "message"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    index_path = root / "data/metadata/ppi_index_labeled.csv"
    raw_root = root / "data/MCGLPPI_RawData/pdbs/m2_pdbbind_dimer_strict"
    output_dir = root / "data/local/pdbs"
    manifest = root / "data/metadata/pdb_collection_manifest.csv"

    if not index_path.is_file():
        raise FileNotFoundError(f"Missing labeled metadata: {index_path}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw PDB directory: {raw_root}")

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != args.expected_count:
        raise ValueError(f"Metadata rows={len(rows)}; expected {args.expected_count}")

    folder_index = {path.name.lower(): path for path in raw_root.iterdir() if path.is_dir()}
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    failures = 0

    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        source_text = destination_text = digest = ""
        try:
            folder = folder_index.get(pdb)
            if folder is None:
                raise FileNotFoundError(f"missing raw PDB folder for {pdb}")
            source = find_source(folder, pdb)
            destination = output_dir / f"{pdb}.pdb"
            source_text = str(source.relative_to(root))
            destination_text = str(destination.relative_to(root))
            source_hash = sha256(source)

            if destination.exists():
                destination_hash = sha256(destination)
                if destination_hash == source_hash:
                    status = "unchanged"
                    digest = destination_hash
                elif not args.overwrite:
                    raise FileExistsError(
                        f"existing destination differs: {destination}; use --overwrite"
                    )
                else:
                    atomic_copy(source, destination)
                    status = "replaced"
                    digest = source_hash
            else:
                atomic_copy(source, destination)
                status = "copied"
                digest = source_hash
            message = ""
        except Exception as error:  # Continue so successful files remain reusable.
            failures += 1
            status = "failed"
            message = str(error)

        records.append(
            {
                "pdb_code": pdb,
                "source": source_text,
                "destination": destination_text,
                "sha256": digest,
                "status": status,
                "message": message,
            }
        )

    write_manifest(manifest, records)
    complete = sum(row["status"] != "failed" for row in records)
    print(f"PDBs complete: {complete}/{len(records)}")
    print(f"Failures: {failures}")
    print(f"Manifest: {manifest}")
    if failures:
        raise SystemExit("PDB COLLECTION INCOMPLETE; rerun after fixing the failures")
    print("PDB COLLECTION PASS")


if __name__ == "__main__":
    main()
