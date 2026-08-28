#!/usr/bin/env python3
"""Create one deterministic PPI-group-disjoint split with final labeled CSVs."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from io import StringIO
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=1243)
    return parser.parse_args()


def csv_bytes(fields: list[str], rows: list[dict[str, object]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")


def tsv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def normalized_lines(content: bytes) -> list[str]:
    return content.decode("utf-8-sig").splitlines()


def publish_consistently(outputs: dict[Path, bytes]) -> None:
    conflicts = [
        path
        for path, content in outputs.items()
        if path.exists()
        and path.read_text(encoding="utf-8-sig").splitlines()
        != normalized_lines(content)
    ]
    if conflicts:
        raise RuntimeError(
            "Existing split files conflict with deterministic regeneration. "
            "No file was changed: " + ", ".join(str(path) for path in conflicts)
        )
    for path, content in outputs.items():
        if path.exists():
            print(f"UNCHANGED: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        print(f"WROTE: {path}")


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    root = args.repo_root.resolve()
    work_dir = root / "data/mmseqs"
    index_path = work_dir / "usable_index.csv"
    groups_path = work_dir / "mmseqs_out/ppi_groups.tsv"
    hits_path = work_dir / "mmseqs_out/all_vs_all.tsv"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / "data/mmseqs_seeds_splits" / output_dir

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {index_path}")
        original_fields = list(reader.fieldnames)
        rows = list(reader)
    if len(rows) != args.expected_total:
        raise ValueError(f"Usable PPI rows={len(rows)}; expected {args.expected_total}")
    if "proaffinity_label" not in original_fields:
        raise ValueError("usable_index.csv is missing proaffinity_label")

    rows_by_pdb: dict[str, dict[str, str]] = {}
    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        if pdb in rows_by_pdb:
            raise ValueError(f"Duplicate PDB in usable index: {pdb}")
        rows_by_pdb[pdb] = row

    group_to_pdbs: defaultdict[str, list[str]] = defaultdict(list)
    pdb_to_group: dict[str, str] = {}
    reported_sizes: dict[str, int] = {}
    with groups_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pdb = row["pdb_code"].strip().lower()
            group = row["ppi_group"].strip()
            if pdb in pdb_to_group:
                raise ValueError(f"Duplicate PDB in ppi_groups.tsv: {pdb}")
            pdb_to_group[pdb] = group
            group_to_pdbs[group].append(pdb)
            reported_sizes[group] = int(row["group_size"])

    if set(rows_by_pdb) != set(pdb_to_group):
        raise ValueError("usable_index.csv and ppi_groups.tsv contain different PDB sets")
    for group, members in group_to_pdbs.items():
        if len(members) != reported_sizes[group]:
            raise ValueError(f"Incorrect reported size for {group}")

    total = len(rows)
    targets = {
        "train": round(total * 0.80),
        "validation": round(total * 0.10),
    }
    targets["test"] = total - targets["train"] - targets["validation"]

    groups = list(group_to_pdbs.items())
    random.Random(args.seed).shuffle(groups)
    val_target = targets["validation"]
    test_target = targets["test"]
    backpointer: dict[
        tuple[int, int], tuple[tuple[int, int], str, str] | None
    ] = {(0, 0): None}
    exact = False

    for group, members in groups:
        size = len(members)
        additions: dict[tuple[int, int], tuple[tuple[int, int], str, str]] = {}
        for val_count, test_count in list(backpointer):
            candidates = [
                ((val_count + size, test_count), "validation"),
                ((val_count, test_count + size), "test"),
            ]
            for state, split in candidates:
                if state[0] > val_target or state[1] > test_target:
                    continue
                if state not in backpointer and state not in additions:
                    additions[state] = ((val_count, test_count), group, split)
        backpointer.update(additions)
        if (val_target, test_target) in backpointer:
            exact = True
            break

    if exact:
        goal = (val_target, test_target)
    else:
        goal = min(
            backpointer,
            key=lambda state: (
                abs(state[0] - val_target) + abs(state[1] - test_target),
                max(abs(state[0] - val_target), abs(state[1] - test_target)),
            ),
        )

    group_split: dict[str, str] = {}
    state = goal
    while state != (0, 0):
        entry = backpointer[state]
        if entry is None:
            raise RuntimeError("Broken dynamic-programming backpointer")
        previous, group, split = entry
        group_split[group] = split
        state = previous
    for group in group_to_pdbs:
        group_split.setdefault(group, "train")

    pdb_split = {
        pdb: group_split[group]
        for group, members in group_to_pdbs.items()
        for pdb in members
    }
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [], "validation": [], "test": []
    }
    assignment_rows: list[dict[str, object]] = []
    for pdb, original in rows_by_pdb.items():
        group = pdb_to_group[pdb]
        split = pdb_split[pdb]
        split_rows[split].append(original)
        assignment_rows.append(
            {
                **original,
                "ppi_group": group,
                "group_size": len(group_to_pdbs[group]),
                "split": split,
            }
        )

    observed_counts = {split: len(split_rows[split]) for split in split_rows}
    if observed_counts != targets:
        raise RuntimeError(f"Split counts={observed_counts}; expected={targets}")
    if not exact:
        raise RuntimeError("Exact 80/10/10 target was not reached")

    observed_group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for pdb, split in pdb_split.items():
        observed_group_splits[pdb_to_group[pdb]].add(split)
    broken = {group: splits for group, splits in observed_group_splits.items() if len(splits) > 1}
    if broken:
        raise RuntimeError(f"PPI groups were split: {broken}")

    leakage_header = [
        "query", "target", "fident", "qcov", "tcov", "query_split", "target_split"
    ]
    leakages: list[list[object]] = []
    with hits_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split()
            if len(fields) != 8:
                raise ValueError(f"Invalid MMseqs row at line {line_number}")
            query, target = fields[0], fields[1]
            fident, qcov, tcov = float(fields[2]), float(fields[4]), float(fields[5])
            if query == target:
                continue
            query_pdb = query.rsplit("_", 1)[0].lower()
            target_pdb = target.rsplit("_", 1)[0].lower()
            if (
                fident >= 0.30
                and qcov >= 0.80
                and tcov >= 0.80
                and pdb_split[query_pdb] != pdb_split[target_pdb]
            ):
                leakages.append(
                    [
                        query, target, fident, qcov, tcov,
                        pdb_split[query_pdb], pdb_split[target_pdb],
                    ]
                )
    if leakages:
        raise RuntimeError(f"Detected {len(leakages)} cross-split homology hits")

    assignment_fields = original_fields + ["ppi_group", "group_size", "split"]
    outputs = {
        output_dir / "split_assignments.csv": csv_bytes(assignment_fields, assignment_rows),
        output_dir / "train_split.csv": csv_bytes(original_fields, split_rows["train"]),
        output_dir / "val_split.csv": csv_bytes(original_fields, split_rows["validation"]),
        output_dir / "test_split.csv": csv_bytes(original_fields, split_rows["test"]),
        output_dir / "cross_split_leakage.tsv": tsv_bytes(leakage_header, leakages),
    }
    publish_consistently(outputs)

    print(f"Seed: {args.seed}")
    print(f"Train / validation / test: {targets['train']} / {targets['validation']} / {targets['test']}")
    print("Broken PPI groups: 0")
    print("Cross-split MMseqs hits: 0")
    print(f"Output: {output_dir}")
    print("GROUP SPLIT PASS")


if __name__ == "__main__":
    main()
