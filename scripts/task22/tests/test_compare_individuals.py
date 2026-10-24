import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task22.compare_individuals import MODELS, run
from source.metrics.ranking_metrics import RankingMetrics


class IndividualComparisonTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_masking(self):
        # Hide the highest score; test hits occur at ranks 1 and 3.
        scores = np.array([[100, 4, 3, 2, 1], [5, 4, 3, 2, 1]], dtype=float)
        hidden = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=bool)
        relevant = np.array([[0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)
        actual = RankingMetrics.calculate_all_metrics(scores, hidden, relevant, 3)
        expected = {"hit@3": 1, "precision@3": 2 / 3, "recall@3": 1, "mrr@3": 1, "ndcg@3": 1.5 / (1 + 1 / np.log2(3)), "f1@3": 0.8, "map@3": 5 / 6}
        for metric, value in expected.items():
            self.assertAlmostEqual(actual[metric], value)

    def test_complete_pipeline_and_reject_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tuned/scores").mkdir(parents=True)
            (root / "split").mkdir()
            for name in MODELS:
                np.savez(
                    root / "tuned/scores" / f"{name}.npz", users=np.array(["u"]), items=np.array(["a", "b", "c", "d"]), scores=[[9.0, 8.0, 7.0, 6.0]]
                )
            for name, item in (("train", "a"), ("valid", "b"), ("test", "c")):
                pd.DataFrame({"user": ["u"], "item": [item]}).to_csv(root / "split" / f"{name}.tsv", sep="\t", index=False)
            items_path = root / "items.tsv"
            items_path.write_text("item\ttitle\tyear\tgenres\na\tA\t2000\tHistory\nb\tB\t2000\tHistory\nc\tC\t2000\tComedy\nd\tD\t2000\tComedy\n")
            summary, protocol = run(root, root / "out", 2, items_path)
            self.assertEqual(len(summary), 11)
            self.assertTrue((summary["ndcg@2"] == 1).all())
            self.assertTrue((summary["coverage@2"] == 0.5).all())
            self.assertTrue((summary["ild@2"] == 0).all())
            self.assertTrue((summary["serendipity@2"] == 0.5).all())
            self.assertTrue({"mae", "mse", "rmse"}.isdisjoint(summary.columns))
            self.assertEqual(protocol["users_evaluated"], 1)
            self.assertTrue((root / "out/comparison.md").exists())
            (root / "split/test.tsv").write_text("user\titem\nu\ta\n")
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                run(root, root / "out", 2, items_path)


if __name__ == "__main__":
    unittest.main()
