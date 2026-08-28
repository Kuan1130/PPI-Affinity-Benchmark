#!/usr/bin/env python3
"""Build protein clusters and indivisible PPI groups from locked MMseqs hits."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-sequences", type=int, default=2486)
    parser.add_argument("--expected-directed-hits", type=int, default=10890)
    parser.add_argument("--expected-nonself-hits", type=int, default=8404)
    parser.add_argument("--expected-protein-clusters", type=int, default=1651)
    parser.add_argument("--expected-ppi-groups", type=int, default=582)
    return parser.parse_args()


def fasta_ids(path: Path) -> list[str]:
    ids = [
        line[1:].strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate FASTA header in {path}")
    return ids


def tsv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def atomic_publish(path: Path, content: bytes) -> str:
    if path.exists() and path.read_bytes() == content:
        return "UNCHANGED"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return "WROTE"


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    work_dir = root / "data/mmseqs"
    fasta_path = work_dir / "all_proteins.fasta"
    hits_path = work_dir / "mmseqs_out/all_vs_all.tsv"
    index_path = work_dir / "usable_index.csv"
    protein_output = work_dir / "mmseqs_out/protein_clusters.tsv"
    ppi_output = work_dir / "mmseqs_out/ppi_groups.tsv"

    sequence_ids = fasta_ids(fasta_path)
    if len(sequence_ids) != args.expected_sequences:
        raise ValueError(
            f"FASTA sequences={len(sequence_ids)}; expected {args.expected_sequences}"
        )
    sequence_set = set(sequence_ids)
    protein_uf = UnionFind(sequence_ids)
    directed_hits = 0
    nonself_hits = 0
    query_counts: Counter[str] = Counter()
    homology_edges: list[tuple[str, str]] = []

    with hits_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split()
            if len(fields) != 8:
                raise ValueError(f"{hits_path}:{line_number}: expected 8 columns")
            query, target = fields[0], fields[1]
            if query not in sequence_set or target not in sequence_set:
                raise ValueError(f"Unknown sequence in hit: {query}, {target}")
            try:
                fident = float(fields[2])
                qcov = float(fields[4])
                tcov = float(fields[5])
            except ValueError as error:
                raise ValueError(f"Invalid numeric hit at line {line_number}") from error
            directed_hits += 1
            query_counts[query] += 1
            if query != target and fident >= 0.30 and qcov >= 0.80 and tcov >= 0.80:
                nonself_hits += 1
                protein_uf.union(query, target)
                homology_edges.append((query, target))

    if directed_hits != args.expected_directed_hits:
        raise RuntimeError(
            f"Directed hits={directed_hits}; expected {args.expected_directed_hits}"
        )
    if nonself_hits != args.expected_nonself_hits:
        raise RuntimeError(
            f"Qualifying non-self hits={nonself_hits}; expected {args.expected_nonself_hits}"
        )

    protein_components: defaultdict[str, list[str]] = defaultdict(list)
    for sequence_id in sequence_ids:
        protein_components[protein_uf.find(sequence_id)].append(sequence_id)
    sorted_proteins = sorted(
        protein_components.values(), key=lambda members: (-len(members), min(members))
    )
    if len(sorted_proteins) != args.expected_protein_clusters:
        raise RuntimeError(
            f"Protein clusters={len(sorted_proteins)}; expected {args.expected_protein_clusters}"
        )

    protein_rows: list[list[object]] = []
    for number, members in enumerate(sorted_proteins, start=1):
        cluster = f"PC{number:04d}"
        protein_rows.extend([cluster, member, len(members)] for member in sorted(members))

    ppi_uf = UnionFind(sequence_ids)
    for query, target in homology_edges:
        ppi_uf.union(query, target)

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column: {index_path}")
        index_rows = list(reader)
    pdb_codes: list[str] = []
    seen: set[str] = set()
    for row in index_rows:
        pdb = row["pdb_code"].strip().lower()
        if pdb in seen:
            raise ValueError(f"Duplicate PDB in usable index: {pdb}")
        seen.add(pdb)
        partners = (f"{pdb}_1", f"{pdb}_2")
        if any(partner not in sequence_set for partner in partners):
            raise ValueError(f"Missing sequence pair for {pdb}")
        ppi_uf.union(*partners)
        pdb_codes.append(pdb)

    ppi_components: defaultdict[str, list[str]] = defaultdict(list)
    for pdb in pdb_codes:
        ppi_components[ppi_uf.find(f"{pdb}_1")].append(pdb)
    sorted_ppis = sorted(
        ppi_components.values(), key=lambda members: (-len(members), min(members))
    )
    if len(sorted_ppis) != args.expected_ppi_groups:
        raise RuntimeError(f"PPI groups={len(sorted_ppis)}; expected {args.expected_ppi_groups}")

    ppi_rows: list[list[object]] = []
    for number, members in enumerate(sorted_ppis, start=1):
        group = f"PG{number:04d}"
        ppi_rows.extend([pdb, group, len(members)] for pdb in sorted(members))

    publications = [
        (
            protein_output,
            tsv_bytes(["protein_cluster", "sequence_id", "cluster_size"], protein_rows),
        ),
        (ppi_output, tsv_bytes(["pdb_code", "ppi_group", "group_size"], ppi_rows)),
    ]
    for path, content in publications:
        print(f"{atomic_publish(path, content)}: {path}")

    protein_sizes = [len(members) for members in sorted_proteins]
    ppi_sizes = [len(members) for members in sorted_ppis]
    print(f"Directed / qualifying non-self hits: {directed_hits} / {nonself_hits}")
    print(
        f"Protein clusters / singletons / largest: {len(protein_sizes)} / "
        f"{sum(size == 1 for size in protein_sizes)} / {max(protein_sizes)}"
    )
    print(
        f"PPI groups / singletons / largest: {len(ppi_sizes)} / "
        f"{sum(size == 1 for size in ppi_sizes)} / {max(ppi_sizes)}"
    )
    print(f"Queries without output: {len(sequence_set - set(query_counts))}")
    print("PPI GROUP BUILD PASS")


if __name__ == "__main__":
    main()
