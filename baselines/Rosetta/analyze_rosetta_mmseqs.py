#!/usr/bin/env python3
"""Evaluate Rosetta InterfaceAnalyzer on five MMseqs splits.

The script intentionally uses only Python's standard library.  It reports two
pre-specified scores:

* primary:   -dG_separated
* secondary: -(dG_separated / dSASA * 100), an interface-size-normalized score

For each MMseqs split, an affine calibration is fitted on the training set
only and evaluated on the test set.  Full-test results remain the official
result.  A separate train-only 3-IQR sensitivity analysis checks whether score
outliers materially change the conclusion without using test labels or test
scores to choose the cutoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import fmean

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = ("seed_0", "seed_1", "seed_42", "seed_142", "seed_4242")
SPLIT_FILES = {
    "train": "train_split.csv",
    "validation": "val_split.csv",
    "test": "test_split.csv",
}
SCORE_VARIANTS = {
    "dG_separated": {
        "role": "primary",
        "column": "rosetta_affinity_score",
        "definition": "-dG_separated_reu",
    },
    "interface_density": {
        "role": "secondary",
        "column": "rosetta_interface_density_score",
        "definition": "-dG_separated_per_dSASA_x100",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument(
        "--scores",
        type=Path,
        help="rosetta_scores.csv; auto-detected if omitted",
    )
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-samples", type=int, default=1243)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    if "pdb_code" not in fields:
        raise ValueError(f"Missing pdb_code column: {path}")
    for row in rows:
        row["pdb_code"] = str(row.get("pdb_code", "")).strip().upper()
        if not row["pdb_code"]:
            raise ValueError(f"Blank pdb_code in {path}")
    return fields, rows


def number(row: dict, column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Bad {column} for {row.get('pdb_code', '?')} in {path}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite {column} for {row.get('pdb_code', '?')} in {path}"
        )
    return value


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            result[order[position]] = rank
        start = stop
    return result


def correlation(x: list[float], y: list[float], rank: bool = False) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Correlation requires equal vectors with at least two values")
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    mean_x, mean_y = fmean(x), fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0:
        raise ValueError("Correlation is undefined because a vector has zero variance")
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denominator


def fit_ols(x: list[float], y: list[float]) -> tuple[float, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("OLS requires equal vectors with at least two values")
    mean_x, mean_y = fmean(x), fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("Training Rosetta scores have zero variance")
    slope = sum(
        (score - mean_x) * (label - mean_y) for score, label in zip(x, y)
    ) / denominator
    return slope, mean_y - slope * mean_x


def errors(observed: list[float], predicted: list[float]) -> tuple[float, float]:
    residuals = [prediction - truth for prediction, truth in zip(predicted, observed)]
    return (
        math.sqrt(fmean(value * value for value in residuals)),
        fmean(abs(value) for value in residuals),
    )


def aggregate(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise ValueError("Cannot aggregate an empty metric")
    mean = fmean(finite)
    return {
        "mean": mean,
        "population_std": math.sqrt(fmean((value - mean) ** 2 for value in finite)),
    }


def three_iqr_bounds(values: list[float]) -> tuple[float, float, float, float]:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    return q1, q3, q1 - 3.0 * iqr, q3 + 3.0 * iqr


def validate_score_table(
    path: Path, expected_samples: int
) -> tuple[list[str], list[dict], dict[str, dict], dict]:
    original_fields, rows = read_csv(path)
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} Rosetta rows, found {len(rows)}")
    codes = [row["pdb_code"] for row in rows]
    if len(set(codes)) != len(codes):
        duplicates = sorted(code for code, count in Counter(codes).items() if count > 1)
        raise ValueError(f"Duplicate Rosetta pdb_code values: {duplicates[:20]}")

    lookup: dict[str, dict] = {}
    d_g_values: list[float] = []
    d_sasa_values: list[float] = []
    density_values: list[float] = []
    interface_counts: Counter[str] = Counter()
    for row in rows:
        code = row["pdb_code"]
        d_g = number(row, "dG_separated_reu", path)
        affinity = number(row, "rosetta_affinity_score", path)
        d_sasa = number(row, "dSASA_int_A2", path)
        density = number(row, "dG_separated_per_dSASA_x100", path)
        nres_int = number(row, "nres_int", path)
        if not math.isclose(affinity, -d_g, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Affinity sign mismatch for {code}")
        if d_sasa <= 0 or nres_int <= 0:
            raise ValueError(
                f"Empty Rosetta interface for {code}: dSASA={d_sasa}, nres={nres_int}"
            )
        calculated_density = d_g / d_sasa * 100.0
        if not math.isclose(density, calculated_density, rel_tol=2e-3, abs_tol=0.01):
            raise ValueError(
                f"dG/dSASA mismatch for {code}: exported={density}, "
                f"calculated={calculated_density}"
            )

        score_function = str(row.get("score_function", "ref2015")).strip() or "ref2015"
        if score_function != "ref2015":
            raise ValueError(f"Unexpected score function for {code}: {score_function}")
        expected_interface = "A_bB" if code == "4FZV" else "A_B"
        interface = str(row.get("interface_definition", expected_interface)).strip()
        interface = interface or expected_interface
        if interface != expected_interface:
            raise ValueError(
                f"Unexpected interface for {code}: {interface}; expected {expected_interface}"
            )

        row["dG_separated_reu"] = d_g
        row["rosetta_affinity_score"] = affinity
        row["dSASA_int_A2"] = d_sasa
        row["dG_separated_per_dSASA_x100"] = density
        row["rosetta_interface_density_score"] = -density
        row["nres_int"] = nres_int
        row["score_function"] = score_function
        row["interface_definition"] = interface
        lookup[code] = row
        d_g_values.append(d_g)
        d_sasa_values.append(d_sasa)
        density_values.append(density)
        interface_counts[interface] += 1

    d_g_q1, d_g_q3, d_g_lower, d_g_upper = three_iqr_bounds(d_g_values)
    den_q1, den_q3, den_lower, den_upper = three_iqr_bounds(density_values)
    for row in rows:
        d_g = float(row["dG_separated_reu"])
        density = float(row["dG_separated_per_dSASA_x100"])
        row["dG_extreme_3iqr_global"] = d_g < d_g_lower or d_g > d_g_upper
        row["density_extreme_3iqr_global"] = density < den_lower or density > den_upper

    added_fields = [
        "rosetta_interface_density_score",
        "dG_extreme_3iqr_global",
        "density_extreme_3iqr_global",
    ]
    output_fields = original_fields + [field for field in added_fields if field not in original_fields]
    quality_control = {
        "rosetta_rows": len(rows),
        "unique_pdb_codes": len(lookup),
        "score_function": "ref2015",
        "interface_definition_counts": dict(sorted(interface_counts.items())),
        "dG_positive_count": sum(value > 0 for value in d_g_values),
        "dG_min_reu": min(d_g_values),
        "dG_q1_reu": d_g_q1,
        "dG_median_reu": percentile(d_g_values, 0.5),
        "dG_q3_reu": d_g_q3,
        "dG_max_reu": max(d_g_values),
        "dG_3iqr_lower_reu": d_g_lower,
        "dG_3iqr_upper_reu": d_g_upper,
        "dG_extreme_3iqr_global_count": sum(
            bool(row["dG_extreme_3iqr_global"]) for row in rows
        ),
        "dSASA_min_A2": min(d_sasa_values),
        "dSASA_max_A2": max(d_sasa_values),
        "density_min": min(density_values),
        "density_q1": den_q1,
        "density_median": percentile(density_values, 0.5),
        "density_q3": den_q3,
        "density_max": max(density_values),
        "density_3iqr_lower": den_lower,
        "density_3iqr_upper": den_upper,
        "density_extreme_3iqr_global_count": sum(
            bool(row["density_extreme_3iqr_global"]) for row in rows
        ),
    }
    return output_fields, rows, lookup, quality_control


def load_splits(
    split_root: Path,
    lookup: dict[str, dict],
    expected_samples: int,
) -> dict[str, dict[str, list[dict]]]:
    all_codes = set(lookup)
    result: dict[str, dict[str, list[dict]]] = {}
    for seed in SEEDS:
        split_rows: dict[str, list[dict]] = {}
        for split_name, filename in SPLIT_FILES.items():
            path = split_root / seed / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            _, rows = read_csv(path)
            codes = [row["pdb_code"] for row in rows]
            if len(set(codes)) != len(codes):
                raise ValueError(f"Duplicate pdb_code within {path}")
            for row in rows:
                code = row["pdb_code"]
                if code not in lookup:
                    raise ValueError(f"Missing Rosetta score for {code} in {path}")
                row["proaffinity_label"] = number(row, "proaffinity_label", path)
            split_rows[split_name] = rows

        joined = [
            row["pdb_code"]
            for split_name in ("train", "validation", "test")
            for row in split_rows[split_name]
        ]
        if len(joined) != expected_samples:
            raise ValueError(
                f"{seed} has {len(joined)} rows across splits; expected {expected_samples}"
            )
        if len(set(joined)) != len(joined):
            raise ValueError(f"{seed} has overlapping train/validation/test PDB codes")
        if set(joined) != all_codes:
            missing = sorted(all_codes - set(joined))
            extra = sorted(set(joined) - all_codes)
            raise ValueError(
                f"{seed} partition differs from Rosetta scores; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        result[seed] = split_rows
    return result


def evaluate(
    splits: dict[str, dict[str, list[dict]]], lookup: dict[str, dict]
) -> tuple[list[dict], list[dict], list[dict], dict]:
    metrics: list[dict] = []
    predictions: list[dict] = []
    exclusions: list[dict] = []

    for variant, metadata in SCORE_VARIANTS.items():
        score_column = metadata["column"]
        for seed in SEEDS:
            split_rows = splits[seed]
            train = split_rows["train"]
            test = split_rows["test"]
            train_x = [float(lookup[row["pdb_code"]][score_column]) for row in train]
            train_y = [float(row["proaffinity_label"]) for row in train]
            test_x = [float(lookup[row["pdb_code"]][score_column]) for row in test]
            test_y = [float(row["proaffinity_label"]) for row in test]

            slope, intercept = fit_ols(train_x, train_y)
            test_prediction = [slope * value + intercept for value in test_x]
            calibrated_rmse, calibrated_mae = errors(test_y, test_prediction)
            baseline_prediction = [fmean(train_y)] * len(test_y)
            baseline_rmse, baseline_mae = errors(test_y, baseline_prediction)

            train_q1, train_q3, train_lower, train_upper = three_iqr_bounds(train_x)
            train_keep = [train_lower <= value <= train_upper for value in train_x]
            test_keep = [train_lower <= value <= train_upper for value in test_x]
            filtered_train_x = [value for value, keep in zip(train_x, train_keep) if keep]
            filtered_train_y = [value for value, keep in zip(train_y, train_keep) if keep]
            filtered_test_x = [value for value, keep in zip(test_x, test_keep) if keep]
            filtered_test_y = [value for value, keep in zip(test_y, test_keep) if keep]
            if len(filtered_train_x) < 2 or len(filtered_test_x) < 2:
                raise ValueError(
                    f"Too few rows after train-only 3-IQR filtering: {seed}/{variant}"
                )
            qc_slope, qc_intercept = fit_ols(filtered_train_x, filtered_train_y)
            qc_prediction = [qc_slope * value + qc_intercept for value in filtered_test_x]
            qc_rmse, qc_mae = errors(filtered_test_y, qc_prediction)

            metric = {
                "score_variant": variant,
                "score_role": metadata["role"],
                "score_definition": metadata["definition"],
                "split_seed": seed,
                "n_train": len(train),
                "n_validation": len(split_rows["validation"]),
                "n_test": len(test),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "raw_score_rp": correlation(test_x, test_y),
                "raw_score_rs": correlation(test_x, test_y, rank=True),
                "calibrated_rp": correlation(test_prediction, test_y),
                "calibrated_rs": correlation(test_prediction, test_y, rank=True),
                "calibrated_rmse": calibrated_rmse,
                "calibrated_mae": calibrated_mae,
                "mean_baseline_rmse": baseline_rmse,
                "mean_baseline_mae": baseline_mae,
                "train_score_q1": train_q1,
                "train_score_q3": train_q3,
                "train_score_3iqr_lower": train_lower,
                "train_score_3iqr_upper": train_upper,
                "sensitivity_n_train": len(filtered_train_x),
                "sensitivity_n_test": len(filtered_test_x),
                "sensitivity_test_excluded": len(test_x) - len(filtered_test_x),
                "sensitivity_calibration_slope": qc_slope,
                "sensitivity_calibration_intercept": qc_intercept,
                "sensitivity_raw_score_rp": correlation(filtered_test_x, filtered_test_y),
                "sensitivity_raw_score_rs": correlation(
                    filtered_test_x, filtered_test_y, rank=True
                ),
                "sensitivity_calibrated_rmse": qc_rmse,
                "sensitivity_calibrated_mae": qc_mae,
            }
            metrics.append(metric)

            for row, raw_score, prediction, keep in zip(
                test, test_x, test_prediction, test_keep
            ):
                score_row = lookup[row["pdb_code"]]
                predictions.append(
                    {
                        "score_variant": variant,
                        "score_role": metadata["role"],
                        "split_seed": seed,
                        "pdb_code": row["pdb_code"],
                        "proaffinity_label": row["proaffinity_label"],
                        "raw_score": raw_score,
                        "calibrated_predicted_pkd": prediction,
                        "calibration_slope": slope,
                        "calibration_intercept": intercept,
                        "inside_train_3iqr_bounds": keep,
                        "dG_separated_reu": score_row["dG_separated_reu"],
                        "dSASA_int_A2": score_row["dSASA_int_A2"],
                        "dG_separated_per_dSASA_x100": score_row[
                            "dG_separated_per_dSASA_x100"
                        ],
                        "dG_extreme_3iqr_global": score_row[
                            "dG_extreme_3iqr_global"
                        ],
                        "density_extreme_3iqr_global": score_row[
                            "density_extreme_3iqr_global"
                        ],
                    }
                )

            for split_name, rows, scores, keep_flags in (
                ("train", train, train_x, train_keep),
                ("test", test, test_x, test_keep),
            ):
                for row, raw_score, keep in zip(rows, scores, keep_flags):
                    if not keep:
                        exclusions.append(
                            {
                                "score_variant": variant,
                                "split_seed": seed,
                                "split": split_name,
                                "pdb_code": row["pdb_code"],
                                "raw_score": raw_score,
                                "train_3iqr_lower": train_lower,
                                "train_3iqr_upper": train_upper,
                            }
                        )

    aggregate_names = (
        "raw_score_rp",
        "raw_score_rs",
        "calibrated_rp",
        "calibrated_rs",
        "calibrated_rmse",
        "calibrated_mae",
        "mean_baseline_rmse",
        "mean_baseline_mae",
        "sensitivity_raw_score_rp",
        "sensitivity_raw_score_rs",
        "sensitivity_calibrated_rmse",
        "sensitivity_calibrated_mae",
        "sensitivity_test_excluded",
    )
    aggregate_result: dict[str, dict] = {}
    for variant, metadata in SCORE_VARIANTS.items():
        variant_rows = [row for row in metrics if row["score_variant"] == variant]
        aggregate_result[variant] = {
            "score_role": metadata["role"],
            "score_definition": metadata["definition"],
            "metrics": {
                name: aggregate([float(row[name]) for row in variant_rows])
                for name in aggregate_names
            },
        }
    return metrics, predictions, exclusions, aggregate_result


def main() -> int:
    arguments = parse_args()
    root = arguments.project_root.resolve()
    scores_path = (
        arguments.scores
        or root / "results/runtime/rosetta/rosetta_scores.csv"
    ).resolve()

    split_root = (
        arguments.split_root
        or root / "data/mmseqs_seeds_splits"
    ).resolve()

    output_dir = (
        arguments.output_dir
        or root / "results/summaries/rosetta"
    ).resolve()
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)

    qc_fields, score_rows, lookup, quality_control = validate_score_table(
        scores_path, arguments.expected_samples
    )
    splits = load_splits(split_root, lookup, arguments.expected_samples)
    metrics, predictions, exclusions, aggregates = evaluate(splits, lookup)

    atomic_write_csv(output_dir / "rosetta_scores_qc.csv", qc_fields, score_rows)
    atomic_write_csv(
        output_dir / "rosetta_mmseqs_split_metrics.csv",
        list(metrics[0]),
        metrics,
    )
    atomic_write_csv(
        output_dir / "rosetta_mmseqs_test_predictions.csv",
        list(predictions[0]),
        predictions,
    )
    exclusion_fields = [
        "score_variant",
        "split_seed",
        "split",
        "pdb_code",
        "raw_score",
        "train_3iqr_lower",
        "train_3iqr_upper",
    ]
    atomic_write_csv(
        output_dir / "rosetta_train_3iqr_exclusions.csv",
        exclusion_fields,
        exclusions,
    )

    global_outliers = sorted(
        (
            {
                "pdb_code": row["pdb_code"],
                "dG_separated_reu": row["dG_separated_reu"],
                "dG_separated_per_dSASA_x100": row[
                    "dG_separated_per_dSASA_x100"
                ],
                "dG_extreme_3iqr_global": row["dG_extreme_3iqr_global"],
                "density_extreme_3iqr_global": row[
                    "density_extreme_3iqr_global"
                ],
            }
            for row in score_rows
            if row["dG_extreme_3iqr_global"] or row["density_extreme_3iqr_global"]
        ),
        key=lambda row: row["pdb_code"],
    )
    summary = {
        "method": {
            "primary_score": SCORE_VARIANTS["dG_separated"]["definition"],
            "secondary_score": SCORE_VARIANTS["interface_density"]["definition"],
            "score_units": "Rosetta energy units (REU); not kcal/mol",
            "calibration": "ordinary least squares fitted on each training split only",
            "official_result_population": "complete test split; no score outliers removed",
            "sensitivity_analysis": (
                "3-IQR score bounds derived from training scores only; the same bounds "
                "are then applied to that split's test scores"
            ),
            "test_labels_or_scores_used_to_fit_or_choose_cutoffs": False,
        },
        "quality_control": quality_control,
        "global_score_outliers_diagnostic_only": global_outliers,
        "five_split_aggregate": aggregates,
    }
    atomic_write_json(output_dir / "rosetta_mmseqs_summary.json", summary)

    print("Rosetta quality control: PASS")
    print(
        f"Rows / unique PDBs: {quality_control['rosetta_rows']} / "
        f"{quality_control['unique_pdb_codes']}"
    )
    print(
        "Interface definitions: "
        + ", ".join(
            f"{name}={count}"
            for name, count in quality_control["interface_definition_counts"].items()
        )
    )
    print(
        f"dG_separated range: {quality_control['dG_min_reu']:.3f} to "
        f"{quality_control['dG_max_reu']:.3f} REU; "
        f"positive={quality_control['dG_positive_count']}; "
        f"global 3-IQR extremes={quality_control['dG_extreme_3iqr_global_count']}"
    )
    print(
        f"Interface-density range: {quality_control['density_min']:.3f} to "
        f"{quality_control['density_max']:.3f}; "
        f"global 3-IQR extremes={quality_control['density_extreme_3iqr_global_count']}"
    )

    for variant, metadata in SCORE_VARIANTS.items():
        print(f"\n{metadata['role'].upper()}: {variant} ({metadata['definition']})")
        for row in (item for item in metrics if item["score_variant"] == variant):
            print(
                f"{row['split_seed']}: Rp={row['raw_score_rp']:.4f} "
                f"Rs={row['raw_score_rs']:.4f} "
                f"calibrated_RMSE={row['calibrated_rmse']:.4f} "
                f"MAE={row['calibrated_mae']:.4f} "
                f"baseline_RMSE={row['mean_baseline_rmse']:.4f} "
                f"MAE={row['mean_baseline_mae']:.4f} | "
                f"sensitivity_excluded={row['sensitivity_test_excluded']} "
                f"Rp={row['sensitivity_raw_score_rp']:.4f} "
                f"RMSE={row['sensitivity_calibrated_rmse']:.4f}"
            )
        print("Five-split mean +/- population SD:")
        for name, values in aggregates[variant]["metrics"].items():
            print(f"  {name}: {values['mean']:.4f} +/- {values['population_std']:.4f}")

    print("\nOfficial metrics use every test sample.")
    print("The train-only 3-IQR result is sensitivity analysis, not the headline result.")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
