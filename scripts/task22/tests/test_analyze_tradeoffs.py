import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.task22.analyze_tradeoffs import BEYOND_NAMES, display_name, run

BEYOND = tuple(f"{metric}@10" for metric in BEYOND_NAMES)


class TradeoffAnalysisTests(unittest.TestCase):
    def write_inputs(self, root):
        rows = [
            ("Random", "baseline", 0.05, 0.1, 5.0, 0.3, 0.01),
            ("Pop", "baseline", 0.10, 0.1, 1.0, 0.9, 0.05),
            ("IndividualA", "individual", 0.30, 0.5, 2.0, 0.7, 0.20),
            ("IndividualB", "individual", 0.35, 0.4, 2.5, 0.8, 0.25),
            ("TunedCombination", "hybrid", 0.40, 0.6, 2.2, 0.6, 0.30),
            ("TunedMixed", "hybrid", 0.38, 0.45, 2.1, 0.75, 0.28),
        ]
        summary = pd.DataFrame(
            rows,
            columns=["model", "category", "ndcg@10", *BEYOND],
        )
        summary_path = root / "summary.csv"
        summary.to_csv(summary_path, index=False)
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps({"experiment": "source", "status": "complete", "warnings": []}))
        return summary_path, protocol_path

    def test_correlations_frontiers_deltas_and_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path, protocol_path = self.write_inputs(root)
            frame, correlations, frontiers, deltas, protocol = run(summary_path, protocol_path, root / "out")
            self.assertEqual(len(frame), 6)
            self.assertEqual(len(correlations), 2 * len(BEYOND))
            self.assertEqual(set(correlations.scope), {"all_models", "non_baselines"})
            self.assertEqual(set(frontiers.beyond_accuracy_metric), set(BEYOND))
            self.assertEqual(protocol["reference_model"], "TunedCombination")
            reference = deltas[deltas.model == "TunedCombination"].iloc[0]
            self.assertEqual(reference["ndcg@10_delta_vs_TunedCombination"], 0)
            self.assertTrue((root / "out/tradeoffs.png").stat().st_size > 0)
            self.assertEqual(display_name("TunedCombination"), "Feature Combination")

    def test_rejects_missing_metric(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path, protocol_path = self.write_inputs(root)
            frame = pd.read_csv(summary_path).drop(columns="ild@10")
            frame.to_csv(summary_path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                run(summary_path, protocol_path, root / "out")


if __name__ == "__main__":
    unittest.main()
