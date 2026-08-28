#!/usr/bin/env python3
"""Resumable parallel FoldX RepairPDB + AnalyseComplex runner for PhysicsPPI_Test."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_NAMES = ("train_split.csv", "val_split.csv", "validation_split.csv", "test_split.csv")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def discover_codes(split_dir: Path) -> list[str]:
    files = [split_dir / name for name in SPLIT_NAMES if (split_dir / name).is_file()]
    if not files:
        raise FileNotFoundError(f"No split CSV files found in {split_dir}")

    codes: set[str] = set()
    for path in files:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"Empty CSV header: {path}")
            lookup = {name.strip().lower(): name for name in reader.fieldnames}
            column = next((lookup[x] for x in ("pdb_code", "pdb", "complex", "id") if x in lookup), None)
            if column is None:
                raise ValueError(f"Cannot find PDB code column in {path}; columns={reader.fieldnames}")
            for row in reader:
                code = (row.get(column) or "").strip().upper()
                if code:
                    codes.add(code)
    if not codes:
        raise ValueError("No PDB codes were read from the split CSV files")
    return sorted(codes)


def pdb_index(pdb_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in pdb_dir.rglob("*.pdb"):
        key = path.stem.upper()
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path
    if duplicates:
        sample = "; ".join(f"{k}: {v}" for k, v in list(duplicates.items())[:5])
        raise ValueError(f"Duplicate PDB basenames found: {sample}")
    return index


def parse_interaction(path: Path) -> float:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) >= 6 and fields[1:3] == ["A", "B"]:
                try:
                    rows.append(float(fields[5]))
                except ValueError:
                    pass
    if len(rows) != 1:
        raise ValueError(f"Expected one A/B energy row in {path}, found {len(rows)}")
    return rows[0]


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"FoldX returned exit code {result.returncode}; see {log_path}")


def run_one(code: str, source_pdb: Path, foldx_bin: Path, molecules: Path, jobs_dir: Path) -> dict:
    final_dir = jobs_dir / code
    done_path = final_dir / "DONE.json"
    if done_path.is_file():
        with done_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        result["state"] = "skipped_complete"
        return result

    attempts = final_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = now_iso()

    try:
        local_pdb = attempt / f"{code}.pdb"
        shutil.copy2(source_pdb, local_pdb)
        os.symlink(molecules, attempt / "molecules", target_is_directory=True)

        run_command(
            [str(foldx_bin), "--command=RepairPDB", f"--pdb={local_pdb.name}"],
            attempt,
            attempt / "repair.log",
        )
        repaired = attempt / f"{code}_Repair.pdb"
        if not repaired.is_file() or repaired.stat().st_size == 0:
            raise FileNotFoundError(f"FoldX did not create {repaired.name}")

        run_command(
            [str(foldx_bin), "--command=AnalyseComplex", f"--pdb={repaired.name}", "--analyseComplexChains=A,B"],
            attempt,
            attempt / "analyse_complex.log",
        )
        interaction_files = list(attempt.glob("Interaction_*.fxout"))
        if len(interaction_files) != 1:
            raise ValueError(f"Expected one Interaction_*.fxout, found {len(interaction_files)}")
        energy = parse_interaction(interaction_files[0])

        # Copy final artifacts before DONE.json. DONE.json is the commit marker.
        for src in (
            repaired,
            attempt / "repair.log",
            attempt / "analyse_complex.log",
            interaction_files[0],
        ):
            destination = final_dir / src.name
            tmp_destination = final_dir / f".{src.name}.tmp"
            shutil.copy2(src, tmp_destination)
            os.replace(tmp_destination, destination)

        result = {
            "pdb_code": code,
            "interaction_energy_kcal_mol": energy,
            "foldx_affinity_score": -energy,
            "source_pdb": str(source_pdb),
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "state": "complete",
            "attempt_dir": str(attempt),
        }
        atomic_json(done_path, result)
        return result
    except Exception as exc:
        failure = {
            "pdb_code": code,
            "state": "failed",
            "started_at": started_at,
            "failed_at": now_iso(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "attempt_dir": str(attempt),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        atomic_json(attempt / "FAILED.json", failure)
        return failure


def collect_results(codes: list[str], jobs_dir: Path) -> tuple[list[dict], list[str]]:
    complete, missing = [], []
    for code in codes:
        path = jobs_dir / code / "DONE.json"
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as handle:
                    complete.append(json.load(handle))
            except (OSError, json.JSONDecodeError):
                missing.append(code)
        else:
            missing.append(code)
    complete.sort(key=lambda row: row["pdb_code"])
    return complete, missing


def export_csv(path: Path, complete: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    columns = ["pdb_code", "interaction_energy_kcal_mol", "foldx_affinity_score", "elapsed_seconds", "finished_at"]
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in complete:
            writer.writerow({key: row.get(key, "") for key in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--pdb-dir", type=Path)
    parser.add_argument("--foldx-bin", type=Path)
    parser.add_argument("--molecules-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N pending structures (for testing)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    root = args.project_root.resolve()

    split_dir = (
        args.split_dir
        or root / "data/mmseqs_seeds_splits/seed_0"
    ).resolve()

    pdb_dir = (
        args.pdb_dir
        or root / "data/local/pdbs"
    ).resolve()

    foldx_bin = (
        args.foldx_bin
        or root / "software/foldx/foldx5/foldx_20261231"
    ).resolve()

    molecules = (
        args.molecules_dir
        or foldx_bin.parent / "molecules"
    ).resolve()

    output_root = (
        args.output_dir
        or root / "results/runtime/foldx"
    ).resolve()
    jobs_dir = output_root / "jobs"
    csv_path = output_root / "foldx_scores.csv"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Prevent two accidental batch launches from processing the same structures.
    lock_handle = None
    if not args.status_only:
        lock_handle = (output_root / "batch.lock").open("w")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another FoldX batch process is already running. "
                "Do not launch a second copy; inspect foldx_batch.log instead."
            )
        lock_handle.write(f"pid={os.getpid()} started={now_iso()}\n")
        lock_handle.flush()

    codes = discover_codes(split_dir)
    index = pdb_index(pdb_dir)
    absent = [code for code in codes if code not in index]
    if absent:
        raise FileNotFoundError(f"Missing {len(absent)} PDB files; examples: {absent[:20]}")

    complete, pending = collect_results(codes, jobs_dir)
    export_csv(csv_path, complete)
    print(f"Dataset: total={len(codes)}, complete={len(complete)}, pending={len(pending)}", flush=True)
    print(f"Durable output: {output_root}", flush=True)
    if args.status_only:
        return 0 if not pending else 2
    if not foldx_bin.is_file() or not os.access(foldx_bin, os.X_OK):
        raise FileNotFoundError(f"FoldX executable missing or not executable: {foldx_bin}")
    if not molecules.is_dir():
        raise FileNotFoundError(f"molecules directory missing: {molecules}")

    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print("Nothing to run; all structures are already complete.")
        return 0

    manifest = {
        "created_at": now_iso(),
        "project_root": str(root),
        "workers": args.workers,
        "dataset_total": len(codes),
        "submitted_this_run": len(pending),
        "codes": pending,
    }
    atomic_json(output_root / "last_run_manifest.json", manifest)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, code, index[code], foldx_bin, molecules, jobs_dir): code
            for code in pending
        }
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["state"] == "failed":
                failures += 1
                print(f"[{number}/{len(pending)}] FAIL {result['pdb_code']}: {result['error']}", flush=True)
            else:
                print(
                    f"[{number}/{len(pending)}] OK {result['pdb_code']} "
                    f"energy={result['interaction_energy_kcal_mol']:.6g} "
                    f"time={result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            # Rebuild the summary after every completion; os.replace makes it atomic.
            current, _ = collect_results(codes, jobs_dir)
            export_csv(csv_path, current)

    complete, pending_after = collect_results(codes, jobs_dir)
    export_csv(csv_path, complete)
    print(f"Final: total={len(codes)}, complete={len(complete)}, pending={len(pending_after)}, failures_this_run={failures}")
    print(f"Scores: {csv_path}")
    return 0 if not pending_after else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed jobs remain saved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
