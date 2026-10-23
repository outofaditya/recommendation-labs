import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.task22.quantify_baselines import run


class BaselineComparisonTests(unittest.TestCase):
    def write_inputs(self, root):
        summary_path = root / "summary.csv"
        pd.DataFrame(
            [
                {"model": "Random", "category": "baseline", "ndcg@10": 0.1, "coverage@10": 0.2, "ild@10": 0.8},
                {"model": "Pop", "category": "baseline", "ndcg@10": 0.2, "coverage@10": 0.2, "ild@10": 0.9},
                {"model": "ModelA", "category": "individual", "ndcg@10": 0.3, "coverage@10": 0.4, "ild@10": 0.7},
                {"model": "HybridA", "category": "hybrid", "ndcg@10": 0.4, "coverage@10": 0.5, "ild@10": 0.9},
            ]
        ).to_csv(summary_path, index=False)
        protocol_path = root / "source_protocol.json"
        protocol_path.write_text(json.dumps({"experiment": "source", "k": 10, "warnings": ["source warning"]}))
        return summary_path, protocol_path

    def test_absolute_and_relative_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path, protocol_path = self.write_inputs(root)
            comparisons, primary, protocol = run(summary_path, protocol_path, root / "out")
            self.assertEqual(len(comparisons), 2 * 2 * 3)
            row = comparisons[(comparisons.model == "ModelA") & (comparisons.baseline == "Pop") & (comparisons.metric == "ndcg@10")].iloc[0]
            self.assertAlmostEqual(row.absolute_difference, 0.1)
            self.assertAlmostEqual(row.relative_change_percent, 50.0)
            ild = comparisons[(comparisons.model == "ModelA") & (comparisons.baseline == "Pop") & (comparisons.metric == "ild@10")].iloc[0]
            self.assertLess(ild.relative_change_percent, 0)
            self.assertEqual(len(primary), 2)
            self.assertEqual(protocol["comparison_count"], 12)
            self.assertTrue((root / "out/baseline_comparison.csv").exists())

    def test_rejects_missing_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path, protocol_path = self.write_inputs(root)
            frame = pd.read_csv(summary_path)
            frame[frame.model != "Random"].to_csv(summary_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing baselines"):
                run(summary_path, protocol_path, root / "out")


if __name__ == "__main__":
    unittest.main()
