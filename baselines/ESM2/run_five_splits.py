#!/usr/bin/env python3
"""Run the same frozen-ESM2 head protocol on five fixed MMseqs splits."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from esmppi.io import DEFAULT_SPLIT_SEEDS, atomic_json_dump


METRICS = ("pearsonr", "spearmanr", "rmse", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("esm2_head_results")
    )
    parser.add_argument(
        "--seeds", nargs="+", default=list(DEFAULT_SPLIT_SEEDS)
    )
    parser.add_argument("--model-seed", type=int, default=1024)
    parser.add_argument("--partner-suffixes", nargs=2, default=("_1", "_2"))
    parser.add_argument("--head", choices=("mlp", "linear"), default="mlp")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=(512, 128))
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_success(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("status") == "success" else None


def build_command(args: argparse.Namespace, seed: str, output_dir: Path) -> list[str]:
    project_root = Path(__file__).resolve().parent
    seed_dir = args.split_root.resolve() / seed
    command = [
        sys.executable,
        "-u",
        str(project_root / "train_esm2_head.py"),
        "--train-csv",
        str(seed_dir / "train_split.csv"),
        "--val-csv",
        str(seed_dir / "val_split.csv"),
        "--test-csv",
        str(seed_dir / "test_split.csv"),
        "--embeddings",
        str(args.embeddings.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--partner-suffixes",
        *args.partner_suffixes,
        "--head",
        args.head,
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--scheduler-factor",
        str(args.scheduler_factor),
        "--scheduler-patience",
        str(args.scheduler_patience),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--seed",
        str(args.model_seed),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.head == "mlp":
        command.extend(["--hidden-dims", *[str(value) for value in args.hidden_dims]])
    return command


def run_and_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("Command: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def write_aggregate(args: argparse.Namespace) -> int:
    rows = []
    for seed in args.seeds:
        summary = load_success(args.output_root / seed / "summary.json")
        if summary is None:
            continue
        row = {
            "split_seed": seed,
            "model_seed": summary["model_seed"],
            "best_epoch": summary["best_epoch"],
        }
        row.update(summary["test"])
        rows.append(row)

    csv_path = args.output_root / "esm2_head_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["split_seed", "model_seed", "best_epoch", *METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {
        "status": "success" if len(rows) == len(args.seeds) else "incomplete",
        "completed": len(rows),
        "requested": len(args.seeds),
        "model_seed": args.model_seed,
        "split_seeds": list(args.seeds),
        "metrics": {},
    }
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        aggregate["metrics"][metric] = {
            "mean": statistics.fmean(values) if values else None,
            "population_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    atomic_json_dump(aggregate, args.output_root / "esm2_head_summary.json")

    print("\nFrozen ESM2 + head aggregate")
    print(f"Completed: {len(rows)}/{len(args.seeds)}")
    for metric in METRICS:
        item = aggregate["metrics"][metric]
        if item["mean"] is not None:
            print(
                f"{metric}: {item['mean']:.4f} +/- {item['population_std']:.4f}"
            )
    print(f"Results: {csv_path}")
    return len(rows)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for seed in args.seeds:
        output_dir = args.output_root / seed
        summary_path = output_dir / "summary.json"
        if args.resume and load_success(summary_path) is not None:
            print(f"[resume] Skip completed {seed}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, seed, output_dir)
        print("\n" + "=" * 64)
        print(f"Running {seed} with fixed model seed {args.model_seed}")
        print("=" * 64)
        return_code = run_and_tee(command, output_dir / "train.log")
        if return_code != 0 or load_success(summary_path) is None:
            failures.append(seed)
            print(f"[ERROR] {seed} failed; inspect {output_dir / 'train.log'}")

    completed = write_aggregate(args)
    if failures:
        print("Failed seeds:", ", ".join(failures))
        return 1
    return 0 if completed == len(args.seeds) else 2


if __name__ == "__main__":
    raise SystemExit(main())

