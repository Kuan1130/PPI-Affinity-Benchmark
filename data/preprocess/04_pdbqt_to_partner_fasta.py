#!/usr/bin/env python3
"""Build the canonical two-partner FASTA files from shared PDBQT files.

The partner boundary follows the historical ProAffinity preprocessing rule:
the chain-prefix length closest to ``<pdb>-cg_A.itp`` defines partner 1. If
that file is unavailable, the final chain is used as partner 2 and the fallback
is recorded in the audit table.

The script never deletes the FASTA directory. Existing sequences are reused
when identical; conflicts stop the run before any FASTA is replaced.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path


AA3_TO_1 = {
    "GLY": "G", "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I",
    "PHE": "F", "TRP": "W", "TYR": "Y", "ASP": "D", "ASN": "N",
    "GLU": "E", "LYS": "K", "GLN": "Q", "MET": "M", "SER": "S",
    "THR": "T", "CYS": "C", "PRO": "P", "HIS": "H", "ARG": "R",
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "ASH": "D",
    "GLH": "E",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    parser.add_argument("--expected-input", type=int, default=1270)
    parser.add_argument("--expected-success", type=int, default=1245)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing FASTA only when its sequence differs.",
    )
    return parser.parse_args()


def read_index(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "pdb_code" not in reader.fieldnames:
            raise ValueError(f"Missing pdb_code column: {path}")
        codes = [row["pdb_code"].strip().lower() for row in reader]
    if len(codes) != len(set(codes)):
        raise ValueError(f"Duplicate pdb_code in {path}")
    return codes


def read_fasta_sequence(path: Path) -> str:
    return "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()


def casefold_file_index(directory: Path, pattern: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in directory.glob(pattern):
        key = path.name.casefold()
        if key in index:
            duplicates.append(f"{index[key].name} / {path.name}")
        index[key] = path
    if duplicates:
        raise ValueError(
            "Case-insensitive duplicate files detected: " + "; ".join(duplicates)
        )
    return index


def parse_pdbqt(path: Path) -> tuple[list[str], dict[str, str]]:
    residues: OrderedDict[str, OrderedDict[tuple[str, str], str]] = OrderedDict()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 27:
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21]
            residue_name = line[17:21].strip().upper()
            residue_key = (line[22:26].strip(), line[26].strip())
            residues.setdefault(chain, OrderedDict()).setdefault(
                residue_key, AA3_TO_1.get(residue_name, "X")
            )

    order = list(residues)
    sequences = {
        chain: "".join(chain_residues.values())
        for chain, chain_residues in residues.items()
    }
    return order, sequences


def get_itp_length(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    length = 0
    in_atoms = False
    last_residue = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[ atoms ]"):
                in_atoms = True
                continue
            if line.startswith("[") and in_atoms:
                break
            if in_atoms:
                fields = line.split()
                if len(fields) >= 4 and fields[2] != last_residue:
                    length += 1
                    last_residue = fields[2]
    return length


def find_raw_folder(raw_root: Path, pdb: str) -> Path | None:
    direct = raw_root / pdb
    if direct.is_dir():
        return direct
    matches = [path for path in raw_root.iterdir() if path.is_dir() and path.name.lower() == pdb]
    return matches[0] if len(matches) == 1 else None


def find_itp(raw_root: Path, pdb: str) -> Path | None:
    folder = find_raw_folder(raw_root, pdb)
    if folder is None:
        return None
    expected = f"{pdb}-cg_a.itp"
    matches = [path for path in folder.glob("*.itp") if path.name.lower() == expected]
    return matches[0] if len(matches) == 1 else None


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_csv_atomic(
    path: Path,
    fields: list[str],
    rows: list[dict[str, object]],
    *,
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    metadata = root / "data/metadata/ppi_index_labeled.csv"
    raw_root = root / "data/MCGLPPI_RawData/pdbs/m2_pdbbind_dimer_strict"
    pdbqt_dir = root / "data/local/pdbqt"
    fasta_dir = root / "data/mmseqs/fasta"
    chain_index = (
        root
        / "baselines/Proaffinity/ProAffinity_Test/"
          "ProAffinity-GNN/data/chain_index.txt"
    )
    audit_path = root / "data/metadata/fasta_generation_audit.csv"

    codes = read_index(metadata)
    if len(codes) != args.expected_input:
        raise ValueError(f"Metadata rows={len(codes)}; expected {args.expected_input}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw PDB root: {raw_root}")
    if not pdbqt_dir.is_dir():
        raise FileNotFoundError(f"Missing PDBQT directory: {pdbqt_dir}")

    pdbqt_index = casefold_file_index(pdbqt_dir, "*_atom_processed.pdbqt")
    desired: dict[str, tuple[str, str]] = {}
    audit_rows: list[dict[str, object]] = []
    chain_rows: list[dict[str, object]] = []

    for pdb in codes:
        key = f"{pdb}_atom_processed.pdbqt".casefold()
        pdbqt = pdbqt_index.get(key)
        row: dict[str, object] = {
            "pdb_code": pdb,
            "status": "failed",
            "method": "",
            "chain_order": "",
            "partner_1_chains": "",
            "partner_2_chains": "",
            "partner_1_length": "",
            "partner_2_length": "",
            "itp_A_length": "",
            "length_difference": "",
            "message": "",
        }
        try:
            if pdbqt is None:
                raise FileNotFoundError("missing PDBQT")
            chain_order, sequences = parse_pdbqt(pdbqt)
            if len(chain_order) < 2:
                raise ValueError(f"only {len(chain_order)} surviving chain(s)")

            itp_length = get_itp_length(find_itp(raw_root, pdb))
            if itp_length > 0:
                cumulative = 0
                candidates: list[tuple[int, int]] = []
                for index in range(len(chain_order) - 1):
                    cumulative += len(sequences[chain_order[index]])
                    candidates.append((abs(cumulative - itp_length), index + 1))
                difference, split_at = min(candidates)
                method = "itp_A_length"
            else:
                split_at = len(chain_order) - 1
                difference = ""
                method = "last_chain_fallback"

            group_1 = chain_order[:split_at]
            group_2 = chain_order[split_at:]
            sequence_1 = "".join(sequences[chain] for chain in group_1)
            sequence_2 = "".join(sequences[chain] for chain in group_2)
            if not sequence_1 or not sequence_2:
                raise ValueError("empty partner sequence")

            header_1 = f">{pdb.upper()}_1|{','.join(group_1)}|Fake Protein|Fake Species"
            header_2 = f">{pdb.upper()}_2|{','.join(group_2)}|Fake Protein|Fake Species"
            desired[f"{pdb}_1.fasta"] = (sequence_1, f"{header_1}\n{sequence_1}\n")
            desired[f"{pdb}_2.fasta"] = (sequence_2, f"{header_2}\n{sequence_2}\n")

            row.update(
                status="ready",
                method=method,
                chain_order=",".join(chain_order),
                partner_1_chains=",".join(group_1),
                partner_2_chains=",".join(group_2),
                partner_1_length=len(sequence_1),
                partner_2_length=len(sequence_2),
                itp_A_length=itp_length or "",
                length_difference=difference,
            )
            chain_rows.append(
                {
                    "pdb_code": pdb,
                    "partner_1_chains": ",".join(group_1),
                    "partner_2_chains": ",".join(group_2),
                    "method": method,
                }
            )
        except Exception as error:
            row["message"] = str(error)
        audit_rows.append(row)

    success = sum(row["status"] == "ready" for row in audit_rows)
    if success != args.expected_success:
        write_csv_atomic(audit_path, list(audit_rows[0]), audit_rows)
        raise RuntimeError(
            f"Partner FASTA candidates={success}; expected {args.expected_success}. "
            f"No FASTA was changed. Inspect {audit_path}."
        )

    fasta_dir.mkdir(parents=True, exist_ok=True)
    existing = casefold_file_index(fasta_dir, "*.fasta")
    desired_keys = {name.casefold() for name in desired}
    dangerous_extras = sorted(
        path.name for key, path in existing.items() if key not in desired_keys
    )
    if dangerous_extras:
        raise RuntimeError(
            "Unexpected FASTA files could change the usable dataset. Move them out "
            "and rerun. Examples: " + ", ".join(dangerous_extras[:20])
        )

    conflicts: list[str] = []
    for name, (sequence, _content) in desired.items():
        current = existing.get(name.casefold())
        if current is not None and read_fasta_sequence(current) != sequence:
            conflicts.append(current.name)
    if conflicts and not args.overwrite:
        raise RuntimeError(
            f"{len(conflicts)} existing FASTA sequence(s) differ. No file was changed. "
            "Inspect them or rerun with --overwrite. Examples: "
            + ", ".join(conflicts[:20])
        )

    wrote = unchanged = replaced = 0
    for name, (sequence, content) in sorted(desired.items()):
        current = existing.get(name.casefold())
        if current is None:
            atomic_write(fasta_dir / name, content)
            wrote += 1
        elif read_fasta_sequence(current) == sequence:
            unchanged += 1
        else:
            atomic_write(current, content)
            replaced += 1

    write_csv_atomic(audit_path, list(audit_rows[0]), audit_rows)
    write_csv_atomic(
        chain_index,
        ["pdb_code", "partner_1_chains", "partner_2_chains", "method"],
        chain_rows,
        delimiter="\t",
    )

    print(f"PPI partner pairs: {success}")
    print(f"FASTA files: {success * 2}")
    print(f"New / unchanged / replaced: {wrote} / {unchanged} / {replaced}")
    print(f"Skipped PPI entries: {len(codes) - success}")
    print(f"Audit: {audit_path}")
    print("PARTNER FASTA PREPARATION PASS")


if __name__ == "__main__":
    main()
