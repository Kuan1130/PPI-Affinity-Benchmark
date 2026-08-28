# Standardized seven-method benchmark summary

All values are five-MMseqs-split test results reported as mean ± population SD (ddof=0).

| Method | Pearson (Rp) | Spearman (Rs) | RMSE | MAE |
|---|---:|---:|---:|---:|
| ProAffinity | 0.2462 ± 0.1207 | 0.2189 ± 0.1235 | 1.9585 ± 0.4425 | 1.5025 ± 0.3453 |
| Graphomer | 0.3874 ± 0.0686 | 0.3638 ± 0.0695 | 1.5649 ± 0.2396 | 1.2005 ± 0.1510 |
| GearNet-Res | 0.2173 ± 0.0627 | 0.2111 ± 0.0460 | 1.8592 ± 0.2378 | 1.4120 ± 0.1273 |
| GearNet-Atom | 0.1387 ± 0.1138 | 0.1431 ± 0.0837 | 1.7122 ± 0.2303 | 1.3255 ± 0.1469 |
| Frozen ESM2-3B + MLP Head | 0.3391 ± 0.1184 | 0.3271 ± 0.1290 | 1.6495 ± 0.1786 | 1.2721 ± 0.1393 |
| FoldX | 0.2218 ± 0.1250 | 0.2468 ± 0.1116 | 1.6562 ± 0.2570 | 1.2840 ± 0.1553 |
| Rosetta InterfaceAnalyzer | 0.1683 ± 0.0600 | 0.2099 ± 0.1382 | 1.6631 ± 0.2454 | 1.2944 ± 0.1503 |

## Metric interpretation

- Neural baselines: all four metrics are computed directly from test predictions.
- FoldX and Rosetta: Pearson and Spearman use the raw physical affinity score; RMSE and MAE use an affine calibration fitted on the corresponding training split only.
- Rosetta headline values use the pre-specified primary score `-dG_separated`; interface density is retained only as a secondary diagnostic.
- The identifiers 0, 1, 42, 142, and 4242 denote five predefined data-split seeds, not five repetitions on one fixed split.
