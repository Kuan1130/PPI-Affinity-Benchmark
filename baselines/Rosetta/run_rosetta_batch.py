#!/usr/bin/env python3
"""Resumable parallel Rosetta InterfaceAnalyzer runner for PhysicsPPI_Test.

The protein partners are PDB chains A and B.  The interface is specified
explicitly so that nonstandard polymer fragments can be assigned safely.  In
the processed 4FZV structure, residues b:94-119 are the unresolved N-terminal
fragment of partner B, so that structure uses interface ``A_bB``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
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
SCORE_FUNCTION = "ref2015"
DEFAULT_INTERFACE = "A_B"
SPECIAL_INTERFACES = {"4FZV": "A_bB"}
REQUIRED_SCORE_FIELDS = (
    "dG_separated",
    "dSASA_int",
    "dG_separated/dSASAx100",
    "nres_int",
)
OPTIONAL_SCORE_FIELDS = (
    "dG_cross",
    "dG_cross/dSASAx100",
    "delta_unsatHbonds",
    "hbonds_int",
    "sc_value",
    "complex_normalized",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
            columns = {name.strip().lower(): name for name in reader.fieldnames}
            code_column = next(
                (columns[name] for name in ("pdb_code", "pdb", "complex", "id") if name in columns),
                None,
            )
            if code_column is None:
                raise ValueError(f"Cannot find PDB code column in {path}; columns={reader.fieldnames}")
            for row in reader:
                code = (row.get(code_column) or "").strip().upper()
                if code:
                    codes.add(code)
    if not codes:
        raise ValueError("No PDB codes were read from the split CSV files")
    return sorted(codes)


def index_pdbs(pdb_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in pdb_dir.rglob("*.pdb"):
        code = path.stem.upper()
        if code in index:
            duplicates.setdefault(code, [index[code]]).append(path)
        else:
            index[code] = path
    if duplicates:
        examples = "; ".join(f"{code}: {paths}" for code, paths in list(duplicates.items())[:5])
        raise ValueError(f"Duplicate PDB basenames found: {examples}")
    return index


def atom_chains(path: Path) -> tuple[str, ...]:
    chains: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM") and len(line) > 21:
                chain = line[21].strip()
                if chain and chain not in chains:
                    chains.append(chain)
    return tuple(chains)


def pdb_residue_count(path: Path, record: str, chain: str, residue_name: str) -> int:
    residues: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if (
                line.startswith(record)
                and len(line) > 26
                and line[21] == chain
                and line[17:20].strip() == residue_name
            ):
                residues.add((line[22:26], line[26]))
    return len(residues)


def deterministic_seed(code: str) -> int:
    digest = hashlib.sha256(code.encode("ascii")).digest()
    return int.from_bytes(digest[:4], "big") % 2_000_000_000 + 1


def interface_for_code(code: str) -> str:
    return SPECIAL_INTERFACES.get(code, DEFAULT_INTERFACE)


def parse_scorefile(path: Path) -> dict[str, float | str]:
    header: list[str] | None = None
    data_rows: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0] != "SCORE:":
                continue
            values = fields[1:]
            if "description" in values and "dG_separated" in values:
                header = values
                data_rows = []
            elif header is not None:
                data_rows.append(values)

    if header is None:
        raise ValueError(f"No InterfaceAnalyzer header found in {path}")
    if len(data_rows) != 1:
        raise ValueError(f"Expected one InterfaceAnalyzer data row in {path}, found {len(data_rows)}")
    if len(data_rows[0]) != len(header):
        raise ValueError(
            f"Score column mismatch in {path}: header={len(header)}, values={len(data_rows[0])}"
        )

    raw = dict(zip(header, data_rows[0]))
    parsed: dict[str, float | str] = {"description": raw.get("description", "")}
    for name in REQUIRED_SCORE_FIELDS + OPTIONAL_SCORE_FIELDS:
        if name not in raw:
            if name in REQUIRED_SCORE_FIELDS:
                raise ValueError(f"Required score field {name!r} is missing from {path}")
            continue
        try:
            value = float(raw[name])
        except ValueError as exc:
            raise ValueError(f"Non-numeric {name}={raw[name]!r} in {path}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {name}={value} in {path}")
        parsed[name] = value

    if float(parsed["dSASA_int"]) <= 0 or float(parsed["nres_int"]) <= 0:
        raise ValueError(
            f"Empty interface in {path}: dSASA_int={parsed['dSASA_int']}, nres_int={parsed['nres_int']}"
        )
    return parsed


def run_command(command: list[str], cwd: Path, log_path: Path, timeout_seconds: int) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.flush()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Rosetta exceeded {timeout_seconds} seconds; see {log_path}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"Rosetta returned exit code {result.returncode}; see {log_path}")


def valid_done(path: Path, code: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("pdb_code") != code or result.get("state") != "complete":
            return None
        for field in ("dG_separated_reu", "rosetta_affinity_score"):
            if not math.isfinite(float(result[field])):
                return None
        if not math.isclose(
            float(result["rosetta_affinity_score"]),
            -float(result["dG_separated_reu"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            return None
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def run_one(
    code: str,
    source_pdb: Path,
    rosetta_bin: Path,
    database: Path,
    jobs_dir: Path,
    timeout_seconds: int,
) -> dict:
    final_dir = jobs_dir / code
    final_dir.mkdir(parents=True, exist_ok=True)
    done_path = final_dir / "DONE.json"
    existing = valid_done(done_path, code)
    if existing is not None:
        existing["state"] = "skipped_complete"
        return existing

    attempts_dir = final_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt = attempts_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = now_iso()
    seed = deterministic_seed(code)
    interface_definition = interface_for_code(code)

    try:
        local_pdb = attempt / f"{code}.pdb"
        shutil.copy2(source_pdb, local_pdb)
        score_path = attempt / "score.sc"
        log_path = attempt / "interface_analyzer.log"
        command = [
            str(rosetta_bin),
            "-database",
            str(database),
            "-s",
            local_pdb.name,
            "-interface",
            interface_definition,
            "-score:weights",
            SCORE_FUNCTION,
            "-pack_input",
            "true",
            "-pack_separated",
            "true",
            "-compute_packstat",
            "false",
            "-tracer_data_print",
            "false",
            "-out:file:score_only",
            score_path.name,
            "-out:overwrite",
            "true",
            "-constant_seed",
            "-jran",
            str(seed),
        ]
        run_command(command, attempt, log_path, timeout_seconds)
        if not score_path.is_file() or score_path.stat().st_size == 0:
            raise FileNotFoundError(f"Rosetta did not create {score_path.name}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if f"{code}_0001 reported success" not in log_text:
            raise ValueError("Rosetta exited without the expected successful-job marker")
        scores = parse_scorefile(score_path)

        for source in (score_path, log_path):
            destination = final_dir / source.name
            temporary = final_dir / f".{source.name}.{os.getpid()}.tmp"
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)

        d_g = float(scores["dG_separated"])
        result = {
            "pdb_code": code,
            "dG_separated_reu": d_g,
            "rosetta_affinity_score": -d_g,
            "dSASA_int_A2": float(scores["dSASA_int"]),
            "dG_separated_per_dSASA_x100": float(scores["dG_separated/dSASAx100"]),
            "nres_int": float(scores["nres_int"]),
            "dG_cross_reu": scores.get("dG_cross"),
            "dG_cross_per_dSASA_x100": scores.get("dG_cross/dSASAx100"),
            "delta_unsatHbonds": scores.get("delta_unsatHbonds"),
            "hbonds_int": scores.get("hbonds_int"),
            "shape_complementarity": scores.get("sc_value"),
            "complex_normalized": scores.get("complex_normalized"),
            "interface_definition": interface_definition,
            "score_function": SCORE_FUNCTION,
            "rosetta_seed": seed,
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
    complete: list[dict] = []
    missing: list[str] = []
    for code in codes:
        result = valid_done(jobs_dir / code / "DONE.json", code)
        if result is None:
            missing.append(code)
        else:
            complete.append(result)
    complete.sort(key=lambda row: row["pdb_code"])
    return complete, missing


def export_csv(path: Path, complete: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    columns = [
        "pdb_code",
        "dG_separated_reu",
        "rosetta_affinity_score",
        "dSASA_int_A2",
        "dG_separated_per_dSASA_x100",
        "nres_int",
        "dG_cross_reu",
        "dG_cross_per_dSASA_x100",
        "delta_unsatHbonds",
        "hbonds_int",
        "shape_complementarity",
        "complex_normalized",
        "interface_definition",
        "score_function",
        "rosetta_seed",
        "elapsed_seconds",
        "finished_at",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in complete:
            output_row = {name: row.get(name, "") for name in columns}
            if not output_row["interface_definition"]:
                output_row["interface_definition"] = interface_for_code(str(row["pdb_code"]))
            writer.writerow(output_row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--pdb-dir", type=Path)
    parser.add_argument("--rosetta-bin", type=Path)
    parser.add_argument("--database-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--job-timeout", type=int, default=3600)
    parser.add_argument("--expected-samples", type=int, default=1243)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N pending structures")
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("--workers must be >= 1")
    if options.job_timeout < 1:
        parser.error("--job-timeout must be >= 1")
    if options.limit is not None and options.limit < 1:
        parser.error("--limit must be >= 1")

    root = options.project_root.resolve()

    split_dir = (
        options.split_dir
        or root / "data/mmseqs_seeds_splits/seed_0"
    ).resolve()

    pdb_dir = (
        options.pdb_dir
        or root / "data/local/pdbs"
    ).resolve()

    rosetta_root = root / "software/rosetta/current"

    rosetta_bin = (
        options.rosetta_bin
        or rosetta_root / "source/bin/InterfaceAnalyzer.static.linuxgccrelease"
    ).resolve()

    database = (
        options.database_dir
        or rosetta_root / "database"
    ).resolve()

    output_root = (
        options.output_dir
        or root / "results/runtime/rosetta"
    ).resolve()
    jobs_dir = output_root / "jobs"
    csv_path = output_root / "rosetta_scores.csv"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    lock_handle = None
    if not options.status_only:
        lock_handle = (output_root / "batch.lock").open("w")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another Rosetta batch process is already running; inspect rosetta_batch.log instead"
            ) from exc
        lock_handle.write(f"pid={os.getpid()} started={now_iso()}\n")
        lock_handle.flush()

    codes = discover_codes(split_dir)
    if len(codes) != options.expected_samples:
        raise ValueError(f"Expected {options.expected_samples} unique PDB codes, found {len(codes)}")
    index = index_pdbs(pdb_dir)
    absent = [code for code in codes if code not in index]
    if absent:
        raise FileNotFoundError(f"Missing {len(absent)} PDB files; examples: {absent[:20]}")

    bad_chains = []
    for code in codes:
        chains = atom_chains(index[code])
        if chains != ("A", "B"):
            bad_chains.append((code, chains))
    if bad_chains:
        raise ValueError(f"Expected exactly ATOM chains A,B; examples: {bad_chains[:20]}")
    if "4FZV" in index:
        unknown_fragment_size = pdb_residue_count(index["4FZV"], "HETATM", "b", "UNK")
        if unknown_fragment_size != 26:
            raise ValueError(
                "4FZV special interface A_bB requires exactly 26 HETATM UNK residues "
                f"in lowercase chain b; found {unknown_fragment_size}"
            )

    complete, pending = collect_results(codes, jobs_dir)
    complete_by_code = {row["pdb_code"]: row for row in complete}
    export_csv(csv_path, complete)
    print(
        f"Dataset: total={len(codes)}, complete={len(complete)}, pending={len(pending)}; chain_check=PASS",
        flush=True,
    )
    print(f"Durable output: {output_root}", flush=True)
    if options.status_only:
        return 0 if not pending else 2
    if not rosetta_bin.is_file() or not os.access(rosetta_bin, os.X_OK):
        raise FileNotFoundError(f"Rosetta executable missing or not executable: {rosetta_bin}")
    if not database.is_dir():
        raise FileNotFoundError(f"Rosetta database missing: {database}")

    submitted = pending[: options.limit] if options.limit is not None else pending
    if not submitted:
        print("Nothing to run; all structures are already complete.")
        return 0

    atomic_json(
        output_root / "last_run_manifest.json",
        {
            "created_at": now_iso(),
            "project_root": str(root),
            "rosetta_executable": str(rosetta_bin.resolve()),
            "database": str(database.resolve()),
            "score_function": SCORE_FUNCTION,
            "default_interface": DEFAULT_INTERFACE,
            "special_interfaces": SPECIAL_INTERFACES,
            "workers": options.workers,
            "job_timeout_seconds": options.job_timeout,
            "dataset_total": len(codes),
            "submitted_this_run": len(submitted),
            "codes": submitted,
        },
    )

    failures = 0
    with ThreadPoolExecutor(max_workers=options.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                code,
                index[code],
                rosetta_bin,
                database,
                jobs_dir,
                options.job_timeout,
            ): code
            for code in submitted
        }
        for number, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                print(f"[{number}/{len(submitted)}] FAIL {code}: unexpected {type(exc).__name__}: {exc}", flush=True)
            else:
                if result["state"] == "failed":
                    failures += 1
                    print(f"[{number}/{len(submitted)}] FAIL {code}: {result['error']}", flush=True)
                else:
                    complete_by_code[code] = result
                    print(
                        f"[{number}/{len(submitted)}] OK {code} "
                        f"dG={float(result['dG_separated_reu']):.6g} "
                        f"dSASA={float(result['dSASA_int_A2']):.6g} "
                        f"time={float(result['elapsed_seconds']):.1f}s",
                        flush=True,
                    )
            export_csv(csv_path, sorted(complete_by_code.values(), key=lambda row: row["pdb_code"]))

    complete, pending_after = collect_results(codes, jobs_dir)
    export_csv(csv_path, complete)
    print(
        f"Final: total={len(codes)}, complete={len(complete)}, "
        f"pending={len(pending_after)}, failures_this_run={failures}"
    )
    print(f"Scores: {csv_path}")
    if options.limit is not None:
        return 0 if failures == 0 else 1
    return 0 if not pending_after else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed jobs remain saved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
