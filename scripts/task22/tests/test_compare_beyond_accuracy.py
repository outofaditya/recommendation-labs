import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.task22.compare_beyond_accuracy import METRIC_NAMES as METRICS, pareto_mask, run


class BeyondAccuracyComparisonTests(unittest.TestCase):
    def write_inputs(self, root):
        summary_path = root / "summary.csv"
        pd.DataFrame(
            [
                {"model": "Random", "category": "baseline", "coverage@10": 1, "novelty@10": 4, "ild@10": 1, "serendipity@10": 1},
                {"model": "Pop", "category": "baseline", "coverage@10": 1, "novelty@10": 1, "ild@10": 4, "serendipity@10": 1},
                {"model": "Individual", "category": "individual", "coverage@10": 4, "novelty@10": 3, "ild@10": 3, "serendipity@10": 3},
                {"model": "Hybrid", "category": "hybrid", "coverage@10": 3, "novelty@10": 4, "ild@10": 3, "serendipity@10": 4},
            ]
        ).to_csv(summary_path, index=False)
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps({"experiment": "source", "status": "complete", "warnings": []}))
        return summary_path, protocol_path

    def test_rankings_leaders_and_frontier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path, protocol_path = self.write_inputs(root)
            summary, rankings, leaders, frontier, protocol = run(summary_path, protocol_path, root / "out")
            self.assertEqual(len(summary), 4)
            self.assertEqual(len(rankings), 4 * len(METRICS))
            self.assertEqual(set(frontier.model), {"Pop", "Individual", "Hybrid"})
            novelty = rankings[rankings.metric == "novelty@10"]
            self.assertEqual(set(novelty[novelty.global_rank == 1].model), {"Random", "Hybrid"})
            self.assertTrue(((leaders.model == "Individual") & (leaders.metric == "coverage@10")).any())
            self.assertIn("no_composite_score", protocol)
            self.assertTrue((root / "out/metric_rankings.csv").exists())

    def test_pareto_mask_and_missing_metric(self):
        self.assertEqual(pareto_mask([[1, 1], [2, 2], [3, 1]]).tolist(), [False, True, True])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path, protocol_path = self.write_inputs(root)
            frame = pd.read_csv(summary_path).drop(columns="ild@10")
            frame.to_csv(summary_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                run(summary_path, protocol_path, root / "out")


if __name__ == "__main__":
    unittest.main()
