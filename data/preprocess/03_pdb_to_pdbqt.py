#!/usr/bin/env python3
"""Convert shared PDB files to PDBQT safely and resumably with Open Babel."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--expected-count", type=int, default=1270)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_pdbqt(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return any(line.startswith(("ATOM", "HETATM")) for line in handle)


def convert_one(
    pdb_path: Path,
    output_dir: Path,
    timeout: int,
    overwrite: bool,
) -> dict[str, str]:
    pdb = pdb_path.stem.lower()
    output = output_dir / f"{pdb}_atom_processed.pdbqt"
    if valid_pdbqt(output) and not overwrite:
        return {"pdb_code": pdb, "status": "unchanged", "message": ""}

    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.{threading.get_ident()}.tmp.pdbqt"
    )
    temporary.unlink(missing_ok=True)
    command = [
        "obabel",
        "-ipdb",
        str(pdb_path),
        "-opdbqt",
        "-O",
        str(temporary),
        "-p",
        "7.4",
    ]

    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Open Babel exit={completed.returncode}: {completed.stderr[-1000:]}"
            )
        if not valid_pdbqt(temporary):
            raise RuntimeError("Open Babel produced an empty or invalid PDBQT")
        temporary.replace(output)
        return {"pdb_code": pdb, "status": "converted", "message": ""}
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return {"pdb_code": pdb, "status": "failed", "message": str(error)}


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    input_dir = root / "data/local/pdbs"
    output_dir = root / "data/local/pdbqt"
    manifest = root / "data/metadata/pdbqt_conversion_manifest.csv"

    if shutil.which("obabel") is None:
        raise SystemExit("Open Babel executable 'obabel' was not found in PATH")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    pdb_files = sorted(input_dir.glob("*.pdb"), key=lambda path: path.name.lower())
    if len(pdb_files) != args.expected_count:
        raise ValueError(f"Input PDB count={len(pdb_files)}; expected {args.expected_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                convert_one, path, output_dir, args.timeout, args.overwrite
            ): path
            for path in pdb_files
        }
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if number % 50 == 0 or number == len(futures):
                print(f"Processed: {number}/{len(futures)}")

    results.sort(key=lambda row: row["pdb_code"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pdb_code", "status", "message"]
        )
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(manifest)

    failures = [row for row in results if row["status"] == "failed"]
    valid_count = sum(
        valid_pdbqt(path) for path in output_dir.glob("*_atom_processed.pdbqt")
    )
    print(f"Valid PDBQT files: {valid_count}/{args.expected_count}")
    print(f"Failures this run: {len(failures)}")
    print(f"Manifest: {manifest}")
    if failures or valid_count != args.expected_count:
        raise SystemExit("PDBQT CONVERSION INCOMPLETE; rerun to resume failed files")
    print("PDBQT CONVERSION PASS")


if __name__ == "__main__":
    main()
