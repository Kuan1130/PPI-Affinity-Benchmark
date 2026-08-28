#!/usr/bin/env python3
"""Evaluate FoldX on five MMseqs splits using only Python's standard library."""

import argparse, csv, json, math, os
from pathlib import Path
from statistics import fmean

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = ["seed_0", "seed_1", "seed_42", "seed_142", "seed_4242"]

def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
    )
    p.add_argument("--scores", type=Path)
    p.add_argument("--split-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--expected-samples", type=int, default=1243)
    return p.parse_args()

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames: raise ValueError(f"Missing header: {path}")
        fields, rows = list(r.fieldnames), [dict(x) for x in r]
    if "pdb_code" not in fields: raise ValueError(f"Missing pdb_code: {path}")
    for row in rows:
        row["pdb_code"] = str(row["pdb_code"]).strip().upper()
        if not row["pdb_code"]: raise ValueError(f"Blank pdb_code: {path}")
    return fields, rows

def number(row, column, path):
    try: value = float(row[column])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Bad {column} for {row.get('pdb_code','?')} in {path}") from e
    if not math.isfinite(value): raise ValueError(f"Non-finite {column} in {path}")
    return value

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def percentile(values, q):
    values = sorted(values); pos = (len(values)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo]*(hi-pos)+values[hi]*(pos-lo)

def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__); out = [0.0]*len(values); i = 0
    while i < len(order):
        j = i+1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        rank = ((i+1)+j)/2
        for k in range(i,j): out[order[k]] = rank
        i = j
    return out

def corr(x, y, rank=False):
    if rank: x, y = ranks(x), ranks(y)
    mx, my = fmean(x), fmean(y)
    xc, yc = [v-mx for v in x], [v-my for v in y]
    den = math.sqrt(sum(v*v for v in xc)*sum(v*v for v in yc))
    return float("nan") if den == 0 else sum(a*b for a,b in zip(xc,yc))/den

def fit(x, y):
    mx, my = fmean(x), fmean(y); den = sum((v-mx)**2 for v in x)
    if den == 0: raise ValueError("Training FoldX scores have zero variance")
    slope = sum((a-mx)*(b-my) for a,b in zip(x,y))/den
    return slope, my-slope*mx

def errors(y, pred):
    e = [p-o for p,o in zip(pred,y)]
    return math.sqrt(fmean([v*v for v in e])), fmean([abs(v) for v in e])

def aggregate(values):
    values = [v for v in values if math.isfinite(v)]; mean = fmean(values)
    return {"mean": mean, "population_std": math.sqrt(fmean([(v-mean)**2 for v in values]))}

def main():
    a = args(); root = a.project_root.resolve()
    scores_path = (
        a.scores
        or root / "results/runtime/foldx/foldx_scores.csv"
    ).resolve()

    split_root = (
        a.split_root
        or root / "data/mmseqs_seeds_splits"
    ).resolve()

    out = (
        a.output_dir
        or root / "results/summaries/foldx"
    ).resolve()
    if not scores_path.is_file(): raise FileNotFoundError(scores_path)
    if not split_root.is_dir(): raise FileNotFoundError(split_root)

    fields, score_rows = read_csv(scores_path)
    codes = [r["pdb_code"] for r in score_rows]
    if len(score_rows) != a.expected_samples: raise ValueError(f"Expected {a.expected_samples} scores, found {len(score_rows)}")
    if len(set(codes)) != len(codes): raise ValueError("Duplicate FoldX pdb_code")
    energy_col = "interaction_energy_kcal_mol"; lookup = {}; energies = []
    for row in score_rows:
        energy = number(row, energy_col, scores_path); affinity = -energy
        if "foldx_affinity_score" in fields and not math.isclose(number(row,"foldx_affinity_score",scores_path), affinity, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f"Affinity sign mismatch: {row['pdb_code']}")
        row[energy_col], row["foldx_affinity_score"] = energy, affinity
        lookup[row["pdb_code"]] = (energy, affinity); energies.append(energy)
    q1,q3 = percentile(energies,.25),percentile(energies,.75); lo,hi = q1-3*(q3-q1),q3+3*(q3-q1)
    for row in score_rows: row["energy_extreme_3iqr"] = float(row[energy_col]) < lo or float(row[energy_col]) > hi
    qc_fields = fields + ([] if "foldx_affinity_score" in fields else ["foldx_affinity_score"]) + ["energy_extreme_3iqr"]
    write_csv(out/"foldx_scores_qc.csv", qc_fields, score_rows)

    metrics, predictions = [], []; all_codes = set(lookup)
    for seed in SEEDS:
        splits = {}
        for name, filename in (("train","train_split.csv"),("validation","val_split.csv"),("test","test_split.csv")):
            path = split_root/seed/filename
            _, rows = read_csv(path); seen = [r["pdb_code"] for r in rows]
            if len(set(seen)) != len(seen): raise ValueError(f"Duplicate code in {path}")
            for row in rows:
                row["proaffinity_label"] = number(row,"proaffinity_label",path)
                if row["pdb_code"] not in lookup: raise ValueError(f"Missing FoldX score: {row['pdb_code']}")
                row[energy_col],row["foldx_affinity_score"] = lookup[row["pdb_code"]]
            splits[name] = rows
        joined = [r["pdb_code"] for name in ("train","validation","test") for r in splits[name]]
        if len(joined) != a.expected_samples or len(set(joined)) != len(joined) or set(joined) != all_codes:
            raise ValueError(f"Invalid or mismatched partition: {seed}")
        train,test = splits["train"],splits["test"]
        tx,ty = [float(r["foldx_affinity_score"]) for r in train],[float(r["proaffinity_label"]) for r in train]
        x,y = [float(r["foldx_affinity_score"]) for r in test],[float(r["proaffinity_label"]) for r in test]
        slope,intercept = fit(tx,ty); pred = [slope*v+intercept for v in x]
        rmse,mae = errors(y,pred); brmse,bmae = errors(y,[fmean(ty)]*len(y))
        row = {"split_seed":seed,"n_train":len(train),"n_validation":len(splits["validation"]),"n_test":len(test),
               "calibration_slope":slope,"calibration_intercept":intercept,"raw_score_rp":corr(x,y),"raw_score_rs":corr(x,y,True),
               "calibrated_rp":corr(pred,y),"calibrated_rs":corr(pred,y,True),"calibrated_rmse":rmse,"calibrated_mae":mae,
               "mean_baseline_rmse":brmse,"mean_baseline_mae":bmae}
        metrics.append(row)
        for r,p in zip(test,pred):
            predictions.append({"split_seed":seed,"pdb_code":r["pdb_code"],"proaffinity_label":r["proaffinity_label"],energy_col:r[energy_col],
                                "foldx_affinity_score":r["foldx_affinity_score"],"calibrated_predicted_pkd":p,
                                "calibration_slope":slope,"calibration_intercept":intercept})

    write_csv(out/"foldx_mmseqs_split_metrics.csv", list(metrics[0]), metrics)
    write_csv(out/"foldx_mmseqs_test_predictions.csv", list(predictions[0]), predictions)
    names = ["raw_score_rp","raw_score_rs","calibrated_rp","calibrated_rs","calibrated_rmse","calibrated_mae","mean_baseline_rmse","mean_baseline_mae"]
    aggs = {name:aggregate([float(r[name]) for r in metrics]) for name in names}
    summary = {"method":{"raw_score":"-interaction_energy_kcal_mol","calibration":"OLS fitted on training split only","test_data_used_for_fitting":False},
               "quality_control":{"foldx_rows":len(score_rows),"unique_pdb_codes":len(lookup),"energy_min":min(energies),"energy_q1":q1,
                                  "energy_median":percentile(energies,.5),"energy_q3":q3,"energy_max":max(energies),
                                  "energy_extreme_3iqr_count":sum(bool(r["energy_extreme_3iqr"]) for r in score_rows)},
               "five_split_aggregate":aggs}
    write_json(out/"foldx_mmseqs_summary.json",summary)
    print("FoldX quality control: PASS"); print(f"Rows / unique PDBs: {len(score_rows)} / {len(lookup)}")
    print(f"Interaction-energy range: {min(energies):.4f} to {max(energies):.4f} kcal/mol")
    print("\nPer-split test metrics:")
    for r in metrics:
        print(f"{r['split_seed']}: Rp={r['raw_score_rp']:.4f} Rs={r['raw_score_rs']:.4f} calibrated_RMSE={r['calibrated_rmse']:.4f} MAE={r['calibrated_mae']:.4f} baseline_RMSE={r['mean_baseline_rmse']:.4f} MAE={r['mean_baseline_mae']:.4f}")
    print("\nFive-split mean +/- population SD:")
    for name in names: print(f"{name}: {aggs[name]['mean']:.4f} +/- {aggs[name]['population_std']:.4f}")
    print(f"\nOutputs: {out}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
