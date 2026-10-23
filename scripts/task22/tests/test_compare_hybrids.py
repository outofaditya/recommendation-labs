import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task22.compare_hybrids import HYBRIDS, MODELS, run


class HybridComparisonTests(unittest.TestCase):
    def prepare(self, root):
        (root / "split").mkdir()
        for split, item in (("train", "a"), ("valid", "b"), ("test", "c")):
            pd.DataFrame({"user": ["u"], "item": [item]}).to_csv(root / "split" / f"{split}.tsv", sep="\t", index=False)
        for name in MODELS + HYBRIDS:
            folder = root / ("hybrid/scores" if name in HYBRIDS else "tuned/scores")
            folder.mkdir(parents=True, exist_ok=True)
            np.savez(
                folder / f"{name}.npz",
                users=["u"],
                items=["a", "b", "c", "d", "e"],
                scores=[[100.0, 99.0, 3.0, 2.0, -np.inf if name in HYBRIDS else 1.0]],
            )
        items_path = root / "items.tsv"
        items_path.write_text(
            "item\ttitle\tyear\tgenres\na\tA\t2000\tHistory\nb\tB\t2000\tHistory\nc\tC\t2000\tComedy\nd\tD\t2000\tComedy\ne\tE\t2000\tDrama\n"
        )
        return items_path

    def test_excluded_candidates_and_complete_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items_path = self.prepare(root)
            summary, protocol = run(root, root / "out", 2, items_path)
            self.assertEqual(len(summary), 18)
            self.assertTrue((summary["ndcg@2"] == 1).all())
            self.assertTrue((summary["precision@2"] == 0.5).all())
            self.assertTrue((summary["coverage@2"] == 0.4).all())
            self.assertTrue((summary["serendipity@2"] == 0.5).all())
            self.assertEqual(protocol["diagnostics"][HYBRIDS[0]]["minimum_candidates"], 2)
            comparisons = pd.read_csv(root / "out/hybrid_vs_individual.csv")
            self.assertEqual(len(comparisons), 7 * 9 * 11)
            self.assertTrue((comparisons.difference == 0).all())
            with self.assertRaisesRegex(ValueError, "finite eligible candidates"):
                run(root, root / "out", 3, items_path)

    def test_reject_misaligned_hybrid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            items_path = self.prepare(root)
            path = root / "hybrid/scores" / f"{HYBRIDS[0]}.npz"
            np.savez(path, users=["different-user"], items=["a", "b", "c", "d", "e"], scores=[[1.0, 2.0, 3.0, 4.0, 5.0]])
            with self.assertRaisesRegex(ValueError, "ID order differs"):
                run(root, root / "out", 2, items_path)


if __name__ == "__main__":
    unittest.main()
