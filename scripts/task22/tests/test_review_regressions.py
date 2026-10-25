"""Review regressions using tiny fixtures and mocked training; no experiments."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import hybrids
from scripts.task22 import task22_bootstrap as bootstrap
from scripts.task22 import task22_seeds as seeds
from scripts.task23 import analyze_contributions as contributions
from scripts.task24 import task24_pipeline as behaviour
from source.data import split
from source.metrics.accuracy import ndcg_per_user, quality, top
from source.metrics.ranking_metrics import RankingMetrics


class ReviewRegressionTests(unittest.TestCase):
    def test_top_and_quality_keep_legacy_interfaces(self):
        scores = np.tile(np.arange(12, dtype=float), (2, 1))
        hidden = np.zeros_like(scores, dtype=bool)
        relevant = np.zeros_like(hidden)
        relevant[0, 11] = True
        ranked = top(scores, hidden)
        self.assertEqual(ranked.shape, (2, 10))
        np.testing.assert_array_equal(top(scores, hidden, 3), ranked[:, :3])
        keep, values = quality(ranked, relevant)
        np.testing.assert_array_equal(keep, [True, False])
        np.testing.assert_allclose(values["ndcg@10"], ndcg_per_user(ranked, relevant)[keep])

    def test_pair_blocks_keep_activity_with_users(self):
        x = np.arange(12).reshape(2, 3, 2)
        train = np.array([[1, 0, 0], [1, 1, 0]], dtype=bool)
        person = np.array([[20, 0, 1], [40, 1, 2]])
        movie = np.array([[1, 1990], [0, 2000], [1, 2010]])
        blocks = hybrids.pair_feature_blocks(2, (person, movie))
        rows = hybrids.pairs(x, train, (person, movie))(np.ones((2, 3), dtype=bool))
        np.testing.assert_array_equal(blocks["user"], [2, 3, 4, 5])
        np.testing.assert_array_equal(blocks["movie"], [6, 7, 8])
        np.testing.assert_array_equal(rows[:, blocks["user"][-1]], [1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(rows[:, blocks["movie"][-1]], [2, 1, 0, 2, 1, 0])

    def test_combination_analysis_uses_selected_tree_and_correct_blocks(self):
        x = np.ones((2, 3, 2))
        masks = {name: np.zeros((2, 3), dtype=bool) for name in ("train", "valid")}
        side = (np.ones((2, 3)), np.ones((3, 2)))
        with (
            patch.object(contributions, "candidates", return_value=np.ones((2, 3), dtype=bool)),
            patch.object(contributions, "folds", return_value=[np.array([0]), np.array([1])]),
            patch.object(contributions, "ndcg", return_value=0.0),
            patch.object(contributions, "HistGradientBoostingClassifier") as classifier,
        ):
            classifier.return_value.decision_function.return_value = np.zeros(6)
            contributions.combination_analysis(x, masks, {"depth": None, "rate": 0.03}, side)
        self.assertEqual(classifier.call_count, 8)
        for call in classifier.call_args_list:
            self.assertIsNone(call.kwargs["max_depth"])
            self.assertEqual(call.kwargs["learning_rate"], 0.03)
        widths = [call.args[0].shape[1] for call in classifier.return_value.fit.call_args_list]
        self.assertEqual(widths, [9, 9, 2, 2, 6, 6, 5, 5])

    def test_weighted_cv_uses_selected_depth_and_c(self):
        x = np.ones((2, 4, 2))
        masks = {name: np.zeros((2, 4), dtype=bool) for name in ("train", "valid")}
        with (
            patch.object(contributions, "candidates", return_value=np.ones((2, 4), dtype=bool)) as candidates,
            patch.object(contributions, "fit_weights", return_value=np.ones(2)) as fit,
            patch.object(contributions, "weighted_rank", return_value=0.0),
        ):
            contributions.weighted_cv(x, masks, [0, 1], 0.7, [np.array([0])], 2)
        self.assertEqual(candidates.call_args.args[2], 2)
        self.assertEqual(fit.call_args.args[2], 0.7)

    def test_seed_evaluation_is_inside_seed_workdir(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp).resolve() / "seed_7"
            with patch.object(seeds, "_run") as run, patch.object(seeds, "run_pipeline") as evaluate:
                seeds.run_seed(7, work, [10], 1, Path(temp) / "reference")
            self.assertEqual(run.call_count, 3)
            self.assertEqual(evaluate.call_args.kwargs["results"], work / "results")
            self.assertEqual(evaluate.call_args.kwargs["output_root"], work / "results/task2/repeated_task22")
            self.assertEqual(evaluate.call_args.kwargs["items"], work / "data/ml-100k/ml-100k.item")

    def test_read_seed_tables_once_in_both_layouts(self):
        for ks in ([10], [3, 10]):
            with self.subTest(ks=ks), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                work = root / "runs/seed_7"
                work.mkdir(parents=True)
                (work / "seed_complete.json").write_text(json.dumps({"protocol_version": 2, "seed": 7, "ks": ks, "complete": True}))
                for k in ks:
                    frames = {
                        "experiment1/summary.csv": pd.DataFrame([{"model": "A", f"ndcg@{k}": 0.2}]),
                        "experiment2/summary.csv": pd.DataFrame([{"model": "A", "category": "individual", f"ndcg@{k}": 0.2}]),
                        "experiment3/baseline_comparison.csv": pd.DataFrame(
                            [
                                {
                                    "model": "A",
                                    "category": "individual",
                                    "baseline": "Pop",
                                    "metric": f"ndcg@{k}",
                                    "absolute_difference": 0.1,
                                    "relative_change_percent": 100,
                                }
                            ]
                        ),
                        "experiment4/metric_rankings.csv": pd.DataFrame(
                            [
                                {
                                    "model": "A",
                                    "category": "individual",
                                    "metric": f"coverage@{k}",
                                    "global_rank": 1,
                                    "category_leader": True,
                                }
                            ]
                        ),
                        "experiment4/summary.csv": pd.DataFrame([{"model": "A", "pareto_frontier": True}]),
                        "experiment5/correlations.csv": pd.DataFrame(
                            [
                                {
                                    "scope": "all_models",
                                    "beyond_accuracy_metric": f"coverage@{k}",
                                    "pearson": 0.2,
                                    "spearman": 0.3,
                                }
                            ]
                        ),
                        "experiment5/pairwise_frontiers.csv": pd.DataFrame(
                            [
                                {
                                    "scope": "all_models",
                                    "beyond_accuracy_metric": f"coverage@{k}",
                                    "model": "A",
                                }
                            ]
                        ),
                    }
                    for relative, frame in frames.items():
                        experiment, name = relative.split("/")
                        folder = work / "results/task2/repeated_task22" / experiment
                        if len(ks) > 1:
                            folder /= f"k_{k}"
                        folder.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(folder / name, index=False)
                metrics, *derived = seeds._read_per_seed(root, [7], ks)
                self.assertEqual(len(metrics), 2 * len(ks))
                self.assertTrue(all(len(frame) == len(ks) for frame in derived))

    def test_shared_bootstrap_and_coverage_with_missing_test_user(self):
        users, items = np.array(["u0", "u1", "u2"]), np.array(["a", "b", "c", "d", "e"])
        scores = np.array([[0, 1, 5, 4, 2], [5, 4, 3, 2, 1], [0, 1, 2, 5, 4]], dtype=float)
        masks = {name: np.zeros((3, 5), dtype=bool) for name in ("train", "valid", "test")}
        masks["train"][:, 0] = True
        masks["test"][0, 2] = True
        masks["test"][2, 1] = True
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(bootstrap, "MODELS", ["A", "B"]),
            patch.object(bootstrap, "HYBRIDS", []),
            patch.object(bootstrap, "load_predictions", return_value=(users, items, scores)),
            patch.object(bootstrap, "load_masks", return_value=masks),
            patch.object(bootstrap, "load_item_features", return_value=[["genre"]] * 5),
            patch.object(bootstrap, "_make_plots"),
        ):
            summary, _ = bootstrap.run(output=Path(temp), reps=100, ks=[2])
            coverage = summary[summary.metric == "coverage"]
            np.testing.assert_allclose(coverage.value, 3 / 5)
            self.assertTrue(coverage.ci95_low.isna().all())
            self.assertTrue({"mae", "mse", "rmse"}.isdisjoint(summary.metric))
            pairs = pd.read_csv(Path(temp) / "paired_differences.csv")
            self.assertTrue((pairs[["difference", "ci95_low", "ci95_high"]] == 0).all().all())

    def test_routing_keeps_original_user_ids_with_dense_positions(self):
        scores = np.arange(45).reshape(3, 15).astype(float)
        masks = {name: np.zeros((3, 15), dtype=bool) for name in ("train", "valid", "test")}
        masks["test"][[0, 2], [14, 13]] = True
        features = pd.DataFrame({"user_index": [0, 1, 2], "activity_group": [1, 1, 2], "entropy_group": [1, 2, 2]})
        with tempfile.TemporaryDirectory() as temp:
            result = behaviour.experiment_routing({name: scores for name in behaviour.ANALYSIS_MODELS}, masks, features, Path(temp))
            rows = pd.read_csv(Path(temp) / "routing_per_user.csv")
        self.assertEqual(result["eligible_users"], 2)
        self.assertEqual(rows.user_index.tolist(), [0, 2])

    def test_ndcg_and_bucket_contributions_share_normalization(self):
        scores = np.array([[100, 4, 3, 2, 1], [5, 4, 3, 2, 1]], dtype=float)
        hidden = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=bool)
        relevant = np.array([[0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)
        actual = behaviour.per_user_ndcg(scores, hidden, relevant, 3)
        expected = RankingMetrics.calculate_all_metrics(scores, hidden, relevant, 3)["ndcg@3"]
        self.assertAlmostEqual(actual[0], expected)
        self.assertEqual(actual[1], 0)
        ranked = behaviour.unseen_top(scores, hidden, 3)
        left = relevant & np.array([0, 1, 0, 0, 0], dtype=bool)
        right = relevant & ~np.array([0, 1, 0, 0, 0], dtype=bool)
        np.testing.assert_allclose(
            ndcg_per_user(ranked, left, counts=relevant.sum(axis=1)) + ndcg_per_user(ranked, right, counts=relevant.sum(axis=1)), actual
        )

    def test_shared_split_rejects_unknown_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "split").mkdir()
            (root / "split/train.tsv").write_text("user\titem\nunknown\ta\n")
            with self.assertRaisesRegex(ValueError, "Unmapped"):
                split(["u"], ["a"], root)

    def test_reranking_ignores_short_pool_for_users_without_test_items(self):
        scores = np.array([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0]])
        pool = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        hidden = np.zeros_like(pool)
        ranked = behaviour.rerank(scores, pool, hidden, 2, eligible=np.array([True, False]))
        np.testing.assert_array_equal(ranked[0], [0, 1])
        with self.assertRaisesRegex(ValueError, "finite unseen"):
            behaviour.rerank(scores, pool, hidden, 2, eligible=np.array([True, True]))

    def test_across_seed_intervals_use_paired_differences(self):
        frame = pd.DataFrame(
            [
                {"seed": seed, "experiment": "experiment2", "k": 10, "metric": "ndcg", "model": model, "value": value}
                for seed, values in [(7, (0.1, 0.2)), (8, (0.3, 0.4))]
                for model, value in zip(("A", "B"), values)
            ]
        )
        differences = seeds.paired_model_differences(frame)
        np.testing.assert_allclose(differences.difference, -0.1)
        summary = seeds._ci_summary(differences, ["experiment", "k", "metric", "model", "reference"], ["difference"])
        self.assertAlmostEqual(summary.iloc[0].ci95_low, -0.1)
        self.assertAlmostEqual(summary.iloc[0].ci95_high, -0.1)
        self.assertEqual(summary.iloc[0].n_seeds, 2)


if __name__ == "__main__":
    unittest.main()
