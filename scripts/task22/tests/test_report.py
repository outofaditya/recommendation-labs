"""Check paired reporting direction and executed-source audit behavior."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.task2 import analyze as rerun
from scripts.task2.report import paired, select


class ReportTests(unittest.TestCase):
    def test_reverse_contrast_negates_and_swaps_interval_endpoints(self):
        frame = pd.DataFrame(
            [
                {
                    "experiment": "experiment2",
                    "k": 10,
                    "metric": "ndcg",
                    "model": "A",
                    "reference": "B",
                    "mean": 0.2,
                    "ci95_low": 0.1,
                    "ci95_high": 0.4,
                }
            ]
        )
        result = paired(frame, "B", "A")
        self.assertAlmostEqual(result["mean"], -0.2)
        self.assertAlmostEqual(result.ci95_low, -0.4)
        self.assertAlmostEqual(result.ci95_high, -0.1)

    def test_ambiguous_rows_cannot_silently_enter_report(self):
        frame = pd.DataFrame([{"model": "A"}, {"model": "A"}])
        with self.assertRaisesRegex(ValueError, "Expected one row"):
            select(frame, model="A")

    def test_executed_snapshot_is_checked_independently_of_live_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "analysis_snapshot/source/example.py"
            path.parent.mkdir(parents=True)
            path.write_text("executed code")
            protocol = {"analysis_inputs": {"source/example.py": rerun.digest(path)}}
            rerun.verify_analysis_inputs(root, protocol)
            path.write_text("modified snapshot")
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                rerun.verify_analysis_inputs(root, protocol)

    def test_finalization_requires_verified_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reanalysis_protocol.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "bootstrap"):
                rerun.aggregate(root, {"seeds": []})
            self.assertFalse((root / "aggregate_complete.json").exists())


if __name__ == "__main__":
    unittest.main()
