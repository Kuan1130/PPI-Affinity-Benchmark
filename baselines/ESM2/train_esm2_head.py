#!/usr/bin/env python3
"""Train one frozen-ESM2 PPI affinity regression head on a fixed split."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from esmppi.io import (
    atomic_json_dump,
    finite_selection_score,
    metric_dict,
    read_split_csv,
    set_global_seed,
    validate_disjoint_splits,
)
from esmppi.model import RegressionHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    return device


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_pair_arrays(
    frame: pd.DataFrame,
    records: dict[str, dict[str, Any]],
    suffixes: tuple[str, str],
    context: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    identifiers: list[str] = []
    pair_features: list[np.ndarray] = []
    labels: list[float] = []
    embedding_dim: int | None = None

    for row in frame.itertuples(index=False):
        first_id = f"{row.pdb_code}{suffixes[0]}"
        second_id = f"{row.pdb_code}{suffixes[1]}"
        missing = [key for key in (first_id, second_id) if key not in records]
        if missing:
            raise KeyError(f"{context}: {row.pdb_code} is missing embeddings {missing}")

        first = records[first_id]["embedding"].detach().cpu().float().numpy()
        second = records[second_id]["embedding"].detach().cpu().float().numpy()
        if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
            raise ValueError(
                f"{context}: invalid partner shapes for {row.pdb_code}: "
                f"{first.shape} and {second.shape}"
            )
        if embedding_dim is None:
            embedding_dim = int(first.shape[0])
        elif first.shape[0] != embedding_dim:
            raise ValueError(
                f"{context}: inconsistent embedding dimension for {row.pdb_code}"
            )
        pair = np.concatenate((first + second, np.abs(first - second), first * second))
        if not np.isfinite(pair).all():
            raise ValueError(f"{context}: non-finite pair feature for {row.pdb_code}")
        identifiers.append(str(row.pdb_code))
        pair_features.append(pair.astype(np.float32, copy=False))
        labels.append(float(row.proaffinity_label))

    if not pair_features:
        raise ValueError(f"{context}: split is empty")
    return identifiers, np.stack(pair_features), np.asarray(labels, dtype=np.float32)


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def predict_normalized(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = []
    targets = []
    squared_error_sum = 0.0
    count = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(features)
        squared_error_sum += float(torch.square(output - labels).sum().item())
        count += labels.numel()
        predictions.append(output.cpu().numpy())
        targets.append(labels.cpu().numpy())
    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        squared_error_sum / max(count, 1),
    )


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values.astype(np.float64) * std + mean


def write_predictions(
    identifiers: list[str], target: np.ndarray, prediction: np.ndarray, path: Path
) -> None:
    pd.DataFrame(
        {
            "pdb_code": identifiers,
            "target_proaffinity_label": target,
            "predicted_proaffinity_label": prediction,
        }
    ).to_csv(path, index=False)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value.resolve())
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.early_stop_patience <= 0 or args.scheduler_patience < 0:
        raise ValueError("Invalid patience setting")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)

    frames = {
        "train": read_split_csv(args.train_csv),
        "val": read_split_csv(args.val_csv),
        "test": read_split_csv(args.test_csv),
    }
    validate_disjoint_splits(frames)
    cache = safe_torch_load(args.embeddings)
    records = cache.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"{args.embeddings} does not contain an embedding records dictionary")

    suffixes = tuple(args.partner_suffixes)
    arrays = {
        name: build_pair_arrays(frame, records, suffixes, name)
        for name, frame in frames.items()
    }
    feature_dims = {values[1].shape[1] for values in arrays.values()}
    if len(feature_dims) != 1:
        raise ValueError(f"Pair feature dimensions differ across splits: {feature_dims}")
    input_dim = feature_dims.pop()

    train_raw = arrays["train"][2].astype(np.float64)
    label_mean = float(train_raw.mean())
    label_std = float(train_raw.std(ddof=0))
    if not math.isfinite(label_std) or label_std <= 0:
        raise ValueError(f"Training label standard deviation is invalid: {label_std}")

    normalized_labels = {
        name: ((values[2].astype(np.float64) - label_mean) / label_std).astype(np.float32)
        for name, values in arrays.items()
    }
    loaders = {
        name: make_loader(
            values[1],
            normalized_labels[name],
            args.batch_size,
            shuffle=(name == "train"),
            seed=args.seed,
            num_workers=args.num_workers,
        )
        for name, values in arrays.items()
    }

    hidden_dims = () if args.head == "linear" else tuple(args.hidden_dims)
    model = RegressionHead(input_dim, hidden_dims=hidden_dims, dropout=args.dropout).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    criterion = nn.MSELoss()
    checkpoint_path = args.output_dir / "best_head.pt"
    history_rows = []
    best_score = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_squared_error = 0.0
        train_count = 0
        for features, labels in loaders["train"]:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            train_squared_error += float(torch.square(prediction - labels).sum().item())
            train_count += labels.numel()

        val_target_norm, val_prediction_norm, val_loss = predict_normalized(
            model, loaders["val"], device
        )
        val_target = denormalize(val_target_norm, label_mean, label_std)
        val_prediction = denormalize(val_prediction_norm, label_mean, label_std)
        val_metrics = metric_dict(val_target, val_prediction)
        selection_score = finite_selection_score(val_metrics["pearsonr"])
        scheduler.step(selection_score)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_normalized_mse": train_squared_error / max(train_count, 1),
                "val_normalized_mse": val_loss,
                "val_pearsonr": val_metrics["pearsonr"],
                "val_spearmanr": val_metrics["spearmanr"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "learning_rate": current_lr,
            }
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dims": list(hidden_dims),
                    "dropout": args.dropout,
                    "head": args.head,
                    "best_epoch": best_epoch,
                    "best_val_pearsonr": best_score,
                    "label_mean": label_mean,
                    "label_std": label_std,
                    "embedding_model": cache.get("model"),
                    "embedding_pooling": cache.get("pooling"),
                    "pair_features": ["sum", "absdiff", "product"],
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0 or selection_score == best_score:
            print(
                f"epoch={epoch:03d} train_mse_norm={history_rows[-1]['train_normalized_mse']:.5f} "
                f"val_Rp={val_metrics['pearsonr']:.5f} val_Rs={val_metrics['spearmanr']:.5f} "
                f"val_RMSE={val_metrics['rmse']:.5f} lr={current_lr:.3g}"
            )
        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}; best validation Pearson was "
                f"{best_score:.6f} at epoch {best_epoch}."
            )
            break

    pd.DataFrame(history_rows).to_csv(args.output_dir / "history.csv", index=False)
    if best_epoch < 0 or not checkpoint_path.is_file():
        raise RuntimeError("No valid checkpoint was selected from validation Pearson")

    checkpoint = safe_torch_load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {}
    for name in ("val", "test"):
        target_norm, prediction_norm, normalized_mse = predict_normalized(
            model, loaders[name], device
        )
        target = denormalize(target_norm, label_mean, label_std)
        prediction = denormalize(prediction_norm, label_mean, label_std)
        final_metrics[name] = metric_dict(target, prediction)
        final_metrics[name]["normalized_mse"] = normalized_mse
        write_predictions(
            arrays[name][0],
            target,
            prediction,
            args.output_dir / f"{name}_predictions.csv",
        )

    summary = {
        "status": "success",
        "selection_metric": "validation pearsonr",
        "best_epoch": best_epoch,
        "model_seed": args.seed,
        "embedding_model": cache.get("model"),
        "embedding_requested_revision": cache.get("requested_revision"),
        "embedding_resolved_revision": cache.get("resolved_revision"),
        "embedding_hidden_size": cache.get("hidden_size"),
        "embedding_pooling": cache.get("pooling"),
        "long_sequence_policy": cache.get("long_sequence_policy"),
        "pair_features": ["sum", "absdiff", "product"],
        "head": args.head,
        "hidden_dims": list(hidden_dims),
        "label_normalization": {
            "source": "training split only",
            "mean": label_mean,
            "population_std": label_std,
        },
        "counts": {name: len(values[0]) for name, values in arrays.items()},
        "validation": final_metrics["val"],
        "test": final_metrics["test"],
        "arguments": serializable_args(args),
    }
    atomic_json_dump(summary, args.output_dir / "summary.json")
    print(json.dumps({"best_epoch": best_epoch, "test": final_metrics["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
