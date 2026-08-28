#!/usr/bin/env python3
"""Build protein clusters and indivisible PPI groups from MMseqs2 hits."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FASTA_PATH = SCRIPT_DIR / "all_proteins.fasta"
HITS_PATH = SCRIPT_DIR / "mmseqs_out" / "all_vs_all.tsv"
INDEX_PATH = SCRIPT_DIR / "usable_index.csv"
PROTEIN_OUTPUT = SCRIPT_DIR / "mmseqs_out" / "protein_clusters.tsv"
PPI_OUTPUT = SCRIPT_DIR / "mmseqs_out" / "ppi_groups.tsv"

MIN_SEQUENCE_IDENTITY = 0.30
MIN_QUERY_COVERAGE = 0.80
MIN_TARGET_COVERAGE = 0.80
EXPECTED_SEQUENCE_COUNT = 2486
EXPECTED_PPI_COUNT = 1243


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

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]


def main() -> None:
    for path in (FASTA_PATH, HITS_PATH, INDEX_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    PROTEIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Read every canonical FASTA identifier.
    sequence_ids: list[str] = []
    with FASTA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                sequence_ids.append(line[1:].strip().split()[0])

    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Duplicate FASTA headers were detected.")
    if len(sequence_ids) != EXPECTED_SEQUENCE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SEQUENCE_COUNT} FASTA sequences, "
            f"found {len(sequence_ids)}."
        )

    sequence_set = set(sequence_ids)
    protein_uf = UnionFind(sequence_ids)
    directed_hits = 0
    qualifying_non_self_hits = 0
    query_hit_counts: Counter[str] = Counter()
    homology_edges: list[tuple[str, str]] = []

    # Recheck all three thresholds before an edge is allowed into the graph.
    with HITS_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split()
            if len(fields) != 8:
                raise ValueError(
                    f"{HITS_PATH}:{line_number}: expected 8 columns, "
                    f"found {len(fields)}: {line.rstrip()}"
                )

            query, target = fields[0], fields[1]
            fident = float(fields[2])
            qcov = float(fields[4])
            tcov = float(fields[5])

            if query not in sequence_set or target not in sequence_set:
                raise ValueError(
                    f"Unknown sequence in MMseqs2 hit: {query}, {target}"
                )

            directed_hits += 1
            query_hit_counts[query] += 1
            if (
                query != target
                and fident >= MIN_SEQUENCE_IDENTITY
                and qcov >= MIN_QUERY_COVERAGE
                and tcov >= MIN_TARGET_COVERAGE
            ):
                qualifying_non_self_hits += 1
                protein_uf.union(query, target)
                homology_edges.append((query, target))

    if directed_hits == 0:
        raise ValueError(f"No MMseqs2 hits were read from {HITS_PATH}")

    # Convert connected protein components into stable, sorted cluster IDs.
    protein_components: defaultdict[str, list[str]] = defaultdict(list)
    for sequence_id in sequence_ids:
        protein_components[protein_uf.find(sequence_id)].append(sequence_id)

    sorted_protein_components = sorted(
        protein_components.values(), key=lambda members: (-len(members), min(members))
    )

    with PROTEIN_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["protein_cluster", "sequence_id", "cluster_size"])
        for number, members in enumerate(sorted_protein_components, start=1):
            cluster_id = f"PC{number:04d}"
            for sequence_id in sorted(members):
                writer.writerow([cluster_id, sequence_id, len(members)])

    # A PPI group is indivisible: homologous partners are joined first, then the
    # two partners of every PPI are joined so neither relation can cross splits.
    ppi_uf = UnionFind(sequence_ids)
    for query, target in homology_edges:
        ppi_uf.union(query, target)

    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column in {INDEX_PATH}")
        rows = list(reader)

    if len(rows) != EXPECTED_PPI_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PPI_COUNT} PPI rows, found {len(rows)}."
        )

    pdb_codes: list[str] = []
    seen_pdb: set[str] = set()
    for row in rows:
        pdb = row["pdb_code"].strip().lower()
        if pdb in seen_pdb:
            raise ValueError(f"Duplicate pdb_code in usable_index.csv: {pdb}")
        seen_pdb.add(pdb)

        partner_1 = f"{pdb}_1"
        partner_2 = f"{pdb}_2"
        if partner_1 not in sequence_set or partner_2 not in sequence_set:
            raise ValueError(f"Missing partner sequence for PDB {pdb}")
        ppi_uf.union(partner_1, partner_2)
        pdb_codes.append(pdb)

    ppi_components: defaultdict[str, list[str]] = defaultdict(list)
    for pdb in pdb_codes:
        ppi_components[ppi_uf.find(f"{pdb}_1")].append(pdb)

    sorted_ppi_components = sorted(
        ppi_components.values(), key=lambda members: (-len(members), min(members))
    )

    with PPI_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["pdb_code", "ppi_group", "group_size"])
        for number, members in enumerate(sorted_ppi_components, start=1):
            group_id = f"PG{number:04d}"
            for pdb in sorted(members):
                writer.writerow([pdb, group_id, len(members)])

    protein_sizes = [len(members) for members in sorted_protein_components]
    ppi_sizes = [len(members) for members in sorted_ppi_components]
    max_hits = max(query_hit_counts.values(), default=0)
    queries_without_hits = len(sequence_set - set(query_hit_counts))

    print(f"FASTA sequences: {len(sequence_ids)}")
    print("\nMMseqs2 statistics:")
    print(f"Directed hits: {directed_hits}")
    print(f"Qualifying non-self directed hits: {qualifying_non_self_hits}")
    print(f"Maximum hits for one query: {max_hits}")
    print(f"Sequences with no output hits: {queries_without_hits}")
    print("\nProtein cluster statistics:")
    print(f"Protein clusters: {len(protein_sizes)}")
    print(f"Singleton protein clusters: {sum(size == 1 for size in protein_sizes)}")
    print(f"Largest 20 protein clusters: {protein_sizes[:20]}")
    print("\nPPI group statistics:")
    print(f"PPI samples: {len(pdb_codes)}")
    print(f"Indivisible PPI groups: {len(ppi_sizes)}")
    print(f"Singleton PPI groups: {sum(size == 1 for size in ppi_sizes)}")
    print(f"Largest 20 PPI groups: {ppi_sizes[:20]}")
    print("\nOutput files:")
    print(PROTEIN_OUTPUT)
    print(PPI_OUTPUT)


if __name__ == "__main__":
    main()
