#!/usr/bin/env python3
"""Create one deterministic, PPI-group-disjoint 80/10/10 split."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_PATH = SCRIPT_DIR / "usable_index.csv"
GROUPS_PATH = SCRIPT_DIR / "mmseqs_out" / "ppi_groups.tsv"
HITS_PATH = SCRIPT_DIR / "mmseqs_out" / "all_vs_all.tsv"
EXPECTED_PPI_COUNT = 1243
MIN_SEQUENCE_IDENTITY = 0.30
MIN_QUERY_COVERAGE = 0.80
MIN_TARGET_COVERAGE = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory, normally data/mmseqs_seeds_splits/seed_<seed>.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed must be a non-negative integer")

    for path in (INDEX_PATH, GROUPS_PATH, HITS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "split_assignments.csv"
    leakage_path = output_dir / "cross_split_leakage.tsv"

    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column in {INDEX_PATH}")
        original_fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if len(rows) != EXPECTED_PPI_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PPI_COUNT} PPI rows, found {len(rows)}."
        )

    rows_by_pdb: dict[str, dict[str, str]] = {}
    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        if pdb in rows_by_pdb:
            raise ValueError(f"Duplicate pdb_code: {pdb}")
        rows_by_pdb[pdb] = row

    sample_count = len(rows)
    train_target = round(sample_count * 0.80)
    validation_target = round(sample_count * 0.10)
    test_target = sample_count - train_target - validation_target

    print(f"Split seed: {args.seed}")
    print(f"Total PPI samples: {sample_count}")
    print(
        "Target counts: "
        f"train={train_target}, validation={validation_target}, test={test_target}"
    )

    group_to_pdbs: defaultdict[str, list[str]] = defaultdict(list)
    pdb_to_group: dict[str, str] = {}
    reported_group_sizes: dict[str, int] = {}

    with GROUPS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pdb = row["pdb_code"].strip().lower()
            group_id = row["ppi_group"].strip()
            group_size = int(row["group_size"])
            if pdb in pdb_to_group:
                raise ValueError(f"Duplicate PDB in ppi_groups.tsv: {pdb}")
            pdb_to_group[pdb] = group_id
            group_to_pdbs[group_id].append(pdb)
            reported_group_sizes[group_id] = group_size

    if set(rows_by_pdb) != set(pdb_to_group):
        missing = sorted(set(rows_by_pdb) - set(pdb_to_group))
        extra = sorted(set(pdb_to_group) - set(rows_by_pdb))
        raise ValueError(
            "PDB mapping mismatch between usable_index.csv and ppi_groups.tsv; "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    for group_id, pdbs in group_to_pdbs.items():
        if len(pdbs) != reported_group_sizes[group_id]:
            raise ValueError(
                f"Group-size mismatch for {group_id}: "
                f"observed={len(pdbs)}, reported={reported_group_sizes[group_id]}"
            )

    # The initial single-seed implementation was generalized here: the same
    # deterministic algorithm is run independently for every requested seed.
    groups = list(group_to_pdbs.items())
    random.Random(args.seed).shuffle(groups)

    # Dynamic programming chooses non-overlapping validation and test groups.
    # Every group not selected during backtracking is assigned to training.
    backpointer: dict[
        tuple[int, int], tuple[tuple[int, int], str, str] | None
    ] = {(0, 0): None}
    exact_found = False

    for group_id, pdbs in groups:
        group_size = len(pdbs)
        new_states: dict[tuple[int, int], tuple[tuple[int, int], str, str]] = {}

        for validation_count, test_count in list(backpointer):
            validation_state = (validation_count + group_size, test_count)
            if (
                validation_state[0] <= validation_target
                and validation_state not in backpointer
                and validation_state not in new_states
            ):
                new_states[validation_state] = (
                    (validation_count, test_count),
                    group_id,
                    "validation",
                )

            test_state = (validation_count, test_count + group_size)
            if (
                test_state[1] <= test_target
                and test_state not in backpointer
                and test_state not in new_states
            ):
                new_states[test_state] = (
                    (validation_count, test_count),
                    group_id,
                    "test",
                )

        backpointer.update(new_states)
        if (validation_target, test_target) in backpointer:
            exact_found = True
            break

    if exact_found:
        goal_state = (validation_target, test_target)
    else:
        goal_state = min(
            backpointer,
            key=lambda state: (
                abs(state[0] - validation_target) + abs(state[1] - test_target),
                max(
                    abs(state[0] - validation_target),
                    abs(state[1] - test_target),
                ),
            ),
        )

    group_split: dict[str, str] = {}
    state = goal_state
    while state != (0, 0):
        pointer = backpointer[state]
        if pointer is None:
            raise RuntimeError(f"Broken DP backpointer at state {state}")
        previous_state, group_id, split = pointer
        group_split[group_id] = split
        state = previous_state

    for group_id in group_to_pdbs:
        group_split.setdefault(group_id, "train")

    pdb_split: dict[str, str] = {}
    for group_id, pdbs in group_to_pdbs.items():
        for pdb in pdbs:
            pdb_split[pdb] = group_split[group_id]

    master_fieldnames = original_fieldnames + ["ppi_group", "group_size", "split"]
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    with master_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=master_fieldnames)
        writer.writeheader()
        for pdb, original_row in rows_by_pdb.items():
            group_id = pdb_to_group[pdb]
            split = pdb_split[pdb]
            master_row = dict(original_row)
            master_row["ppi_group"] = group_id
            master_row["group_size"] = len(group_to_pdbs[group_id])
            master_row["split"] = split
            writer.writerow(master_row)
            split_rows[split].append(original_row)

    for split, filename in (
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ):
        with (output_dir / filename).open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=original_fieldnames)
            writer.writeheader()
            writer.writerows(split_rows[split])

    observed_group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for pdb, split in pdb_split.items():
        observed_group_splits[pdb_to_group[pdb]].add(split)
    broken_groups = {
        group_id: splits
        for group_id, splits in observed_group_splits.items()
        if len(splits) > 1
    }
    if broken_groups:
        raise RuntimeError(f"PPI groups were split: {broken_groups}")

    leakages: list[list[object]] = []
    with HITS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split()
            query, target = fields[0], fields[1]
            if query == target:
                continue
            fident = float(fields[2])
            qcov = float(fields[4])
            tcov = float(fields[5])
            query_pdb = query.rsplit("_", 1)[0].lower()
            target_pdb = target.rsplit("_", 1)[0].lower()
            query_split = pdb_split[query_pdb]
            target_split = pdb_split[target_pdb]

            if (
                fident >= MIN_SEQUENCE_IDENTITY
                and qcov >= MIN_QUERY_COVERAGE
                and tcov >= MIN_TARGET_COVERAGE
                and query_split != target_split
            ):
                leakages.append(
                    [
                        query,
                        target,
                        fident,
                        qcov,
                        tcov,
                        query_split,
                        target_split,
                    ]
                )

    with leakage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "query",
                "target",
                "fident",
                "qcov",
                "tcov",
                "query_split",
                "target_split",
            ]
        )
        writer.writerows(leakages)

    print("\nActual split:")
    for split in ("train", "validation", "test"):
        split_groups = {
            pdb_to_group[pdb]
            for pdb, assigned_split in pdb_split.items()
            if assigned_split == split
        }
        group_sizes = sorted(
            [len(group_to_pdbs[group_id]) for group_id in split_groups], reverse=True
        )
        print(
            f"{split}: {len(split_rows[split])} samples, "
            f"{len(split_groups)} groups, largest groups={group_sizes[:10]}"
        )

    print(f"\nExact target reached: {exact_found}")
    print(f"Broken PPI groups: {len(broken_groups)}")
    print(f"Cross-split MMseqs2 hits: {len(leakages)}")
    print(f"Output directory: {output_dir}")
    print(f"Assignment table: {master_path}")
    print(f"Leakage audit: {leakage_path}")


if __name__ == "__main__":
    main()
