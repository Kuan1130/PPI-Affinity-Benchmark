from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SPLIT_SEEDS = ("seed_0", "seed_1", "seed_42", "seed_142", "seed_4242")
REQUIRED_COLUMNS = ("pdb_code", "proaffinity_label")
FASTA_EXTENSIONS = (".fasta", ".fa", ".faa")


def canonical_id(value: object) -> str:
    """Return a stable PDB/sample identifier as read from a CSV."""
    text = str(value).strip()
    if not text:
        raise ValueError("Encountered an empty pdb_code")
    return text


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_single_fasta(path: str | Path) -> str:
    """Read exactly one FASTA record and return an uppercase sequence."""
    path = Path(path)
    records: list[str] = []
    current: list[str] = []
    saw_header = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header:
                    records.append("".join(current))
                    current = []
                saw_header = True
                continue
            if not saw_header:
                raise ValueError(f"{path}:{line_number}: sequence found before FASTA header")
            current.append("".join(line.split()).upper())

    if saw_header:
        records.append("".join(current))
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one FASTA record, found {len(records)}")
    sequence = records[0]
    if not sequence:
        raise ValueError(f"{path}: empty FASTA sequence")
    invalid = sorted(set(sequence) - set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if invalid:
        raise ValueError(f"{path}: invalid sequence characters {invalid}")
    return sequence


def discover_fastas(fasta_dir: str | Path) -> dict[str, Path]:
    """Map FASTA file stems to paths, rejecting ambiguous duplicate stems."""
    fasta_dir = Path(fasta_dir)
    if not fasta_dir.is_dir():
        raise FileNotFoundError(f"FASTA directory does not exist: {fasta_dir}")

    mapping: dict[str, Path] = {}
    for path in sorted(fasta_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FASTA_EXTENSIONS:
            continue
        key = path.stem
        if key in mapping:
            raise ValueError(
                f"Duplicate FASTA stem {key!r}: {mapping[key]} and {path}. "
                "Every partner ID must resolve to exactly one file."
            )
        mapping[key] = path
    if not mapping:
        raise FileNotFoundError(f"No FASTA files found below {fasta_dir}")
    return mapping


def read_split_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    frame = frame.copy()
    frame["pdb_code"] = frame["pdb_code"].map(canonical_id)
    frame["proaffinity_label"] = pd.to_numeric(
        frame["proaffinity_label"], errors="raise"
    ).astype(float)
    if not np.isfinite(frame["proaffinity_label"].to_numpy()).all():
        raise ValueError(f"{path}: labels contain NaN or infinite values")
    duplicated = frame.loc[frame["pdb_code"].duplicated(), "pdb_code"].tolist()
    if duplicated:
        raise ValueError(f"{path}: duplicate pdb_code values, e.g. {duplicated[:5]}")
    return frame


def read_seed_split(split_root: str | Path, seed: str) -> dict[str, pd.DataFrame]:
    seed_dir = Path(split_root) / seed
    return {
        "train": read_split_csv(seed_dir / "train_split.csv"),
        "val": read_split_csv(seed_dir / "val_split.csv"),
        "test": read_split_csv(seed_dir / "test_split.csv"),
    }


def validate_disjoint_splits(frames: dict[str, pd.DataFrame], context: str = "") -> None:
    sets = {name: set(frame["pdb_code"]) for name, frame in frames.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(sets[left] & sets[right])
        if overlap:
            prefix = f"{context}: " if context else ""
            raise ValueError(
                f"{prefix}{left}/{right} overlap contains {len(overlap)} samples, "
                f"e.g. {overlap[:5]}"
            )


def required_partner_ids(
    frames: Iterable[pd.DataFrame], suffixes: tuple[str, str] = ("_1", "_2")
) -> set[str]:
    result: set[str] = set()
    for frame in frames:
        for pdb_code in frame["pdb_code"]:
            result.add(f"{pdb_code}{suffixes[0]}")
            result.add(f"{pdb_code}{suffixes[1]}")
    return result


def metric_dict(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"Metric shape mismatch: {target.shape} vs {prediction.shape}")
    if target.size < 2:
        pearson = spearman = float("nan")
    elif np.std(target) == 0 or np.std(prediction) == 0:
        pearson = spearman = float("nan")
    else:
        pearson = float(stats.pearsonr(target, prediction).statistic)
        spearman = float(stats.spearmanr(target, prediction).statistic)
    error = prediction - target
    return {
        "pearsonr": pearson,
        "spearmanr": spearman,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def finite_selection_score(value: float) -> float:
    return value if math.isfinite(value) else float("-inf")


def atomic_json_dump(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (when installed) PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

