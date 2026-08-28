from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from esmppi.io import (
    metric_dict,
    read_single_fasta,
    read_split_csv,
    validate_disjoint_splits,
)


class IoTests(unittest.TestCase):
    def test_read_single_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.fasta"
            path.write_text(">x\nACD\nEFG\n", encoding="utf-8")
            self.assertEqual(read_single_fasta(path), "ACDEFG")

    def test_split_csv_and_disjointness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, code in (("train", "a"), ("val", "b"), ("test", "c")):
                path = root / f"{name}.csv"
                pd.DataFrame(
                    {"pdb_code": [code], "proaffinity_label": [7.0]}
                ).to_csv(path, index=False)
                paths.append(path)
            frames = {
                "train": read_split_csv(paths[0]),
                "val": read_split_csv(paths[1]),
                "test": read_split_csv(paths[2]),
            }
            validate_disjoint_splits(frames)

    def test_metrics(self) -> None:
        target = np.array([1.0, 2.0, 3.0])
        prediction = np.array([1.0, 2.0, 3.0])
        metrics = metric_dict(target, prediction)
        self.assertAlmostEqual(metrics["pearsonr"], 1.0)
        self.assertAlmostEqual(metrics["spearmanr"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()

