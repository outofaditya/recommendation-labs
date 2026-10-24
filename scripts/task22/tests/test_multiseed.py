"""Seed resume/integrity and aggregation checks; no model training or GPU required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.task2 import multiseed as multi
from scripts.task22 import build_task22_repeat_hybrids as builder


class MultiseedTests(unittest.TestCase):
    def test_frozen_run_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            reference = Path(temp) / "reference"
            reference.mkdir()
            for name in multi.HYBRIDS:
                (reference / f"Tuned{name}.json").write_text('{"fixed": true}')
            with patch.object(multi, "input_hashes", return_value={"config": "original"}):
                multi.prepare(root, reference, [2020, 2021], [1, 3, 5, 10, 20])
                self.assertEqual(multi.protocol_for(root)["seeds"], [2020, 2021])
            with patch.object(multi, "input_hashes", return_value={"config": "changed"}):
                with self.assertRaisesRegex(ValueError, "changed since prepare"):
                    multi.protocol_for(root)
                with self.assertRaisesRegex(ValueError, "Run inputs changed"):
                    multi.prepare(root, reference, [2020, 2021], [1, 3, 5, 10, 20])

    def test_train_passes_seed_and_gpu_config_and_reuses_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_protocol.json").write_text("{}")

            def execute(module, args, workdir):
                self.assertEqual(module, "scripts.export_scores")
                self.assertIn("--require-gpu", args)
                self.assertEqual(args[args.index("--seed") + 1], 2021)
                self.assertEqual(args[args.index("--config") + 1], root / "training_config.yaml")
                model = args[0]
                for kind, extension in [("scores", "npz"), ("metrics", "json")]:
                    path = workdir / "results/tuned" / kind / f"{model}.{extension}"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("complete")
                for split in ("train", "valid", "test"):
                    path = workdir / "results/split" / f"{split}.tsv"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("same split")
                if model in ("NGCF", "LightGCN"):
                    path = workdir / "results/checkpoints" / f"ml-100k-{model}-test-embeddings.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("embeddings")

            with patch.object(multi, "execute", side_effect=execute) as run:
                multi.train(root, 2021, {"training_config": "use_gpu: true", "require_gpu": True})
                self.assertEqual(run.call_count, 11)
                self.assertEqual(run.call_args_list[0].args[1][0], "Random")
                self.assertTrue(multi.training_complete(root, 2021))
                run.reset_mock()
                multi.train(root, 2021, {"training_config": "use_gpu: true", "require_gpu": True})
                run.assert_not_called()
                (root / "runs/seed_2021/results/tuned/scores/NeuMF.npz").write_text("interrupted")
                multi.train(root, 2021, {"training_config": "use_gpu: true", "require_gpu": True})
                self.assertEqual(run.call_count, 1)

    def test_analysis_resume_and_upstream_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "run_protocol.json").write_text("{}")
            folders = {
                "scripts.task22.build_task22_repeat_hybrids": "hybrid",
                "scripts.task22.task22_pipeline": "task2/repeated_task22",
                "scripts.task23.analyze_contributions": "task23",
                "scripts.task24.task24_pipeline": "task24",
            }

            def execute(module, args, workdir):
                output = workdir / "results" / folders[module] / "output.csv"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("value\n1\n")

            with patch.object(multi, "training_complete", return_value=True), patch.object(multi, "execute", side_effect=execute) as run:
                multi.analyze(root, 2021, {"ks": [1, 3, 5, 10, 20]})
                self.assertEqual(run.call_count, 4)
                run.reset_mock()
                multi.analyze(root, 2021, {"ks": [1, 3, 5, 10, 20]})
                run.assert_not_called()
                (root / "runs/seed_2021/results/hybrid/output.csv").write_text("changed")
                multi.analyze(root, 2021, {"ks": [1, 3, 5, 10, 20]})
                self.assertEqual(run.call_count, 4)

    def test_resume_rejects_changed_outputs_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "scores.npz"
            output.write_bytes(b"original")
            marker = root / "marker.json"
            multi.record(marker, root, "seed-2021", [output])
            self.assertTrue(multi.valid_marker(marker, root, "seed-2021"))
            self.assertFalse(multi.valid_marker(marker, root, "seed-2022"))
            output.write_bytes(b"interrupted replacement")
            self.assertFalse(multi.valid_marker(marker, root, "seed-2021"))
            output.unlink()
            self.assertFalse(multi.valid_marker(marker, root, "seed-2021"))

    def test_refuse_partial_aggregate(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(multi, "seed_complete", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "incomplete seeds"):
                multi.aggregate(Path(temp), {"seeds": [2020, 2021], "ks": [10]})
            self.assertFalse((Path(temp) / "aggregate_complete.json").exists())

    def test_reference_combination_parameters_are_used(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp)
            settings = {
                "Weighted": {"depth": 100, "C": 0.0001},
                "Mixed": {"k": 10, "members": builder.MEMBERS},
                "Cascade": {"stages": ["BPR", "SLIMElastic"], "shortlist": 20},
                "Switching": {"groups": [{}, {}, {}, {}]},
                "Combination": {"depth": None, "rate": 0.03},
                "Augmentation": {"pseudo": 5, "k": 25},
                "Metalevel": {"embeddings": "NGCF", "k": 50},
            }
            for name, value in settings.items():
                (reference / f"Tuned{name}.json").write_text(json.dumps(value))
            x = np.ones((2, 3, len(builder.MEMBERS)))
            masks = {s: np.zeros((2, 3), dtype=bool) for s in ("train", "valid")}
            table = np.zeros((2, 3))
            with (
                patch.object(builder, "features", return_value=(x, masks, np.array([1, 2]), np.array([1, 2, 3]))),
                patch.object(builder, "candidates", return_value=np.ones((2, 3), dtype=bool)),
                patch.object(builder, "fit", return_value=np.ones(len(builder.MEMBERS))),
                patch.object(builder, "save"),
                patch.object(builder, "ranks", return_value=x),
                patch.object(builder, "shortlisted", return_value=table),
                patch.object(builder, "switching", return_value=(table, {})),
                patch.object(builder, "augmentation", return_value=(table, {})),
                patch.object(builder, "metalevel", return_value=(table, {})),
                patch.object(builder, "combination", return_value=(table, {})) as combination,
            ):
                builder.build(reference)
                combination.assert_called_once_with(x, masks, depth=None, rate=0.03)

    def test_task23_task24_seed_intervals_and_undefined_rates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed, value in [(2020, 0.2), (2021, 0.4)]:
                folder = root / "runs" / f"seed_{seed}" / "results/task24"
                folder.mkdir(parents=True)
                pd.DataFrame([{"strategy": "base", "ndcg@10": value, "tail_hit_rate": np.nan, "tail_ci95_low": 0.1}]).to_csv(
                    folder / "specialized_retrieval.csv", index=False
                )
                (folder / "routing_summary.json").write_text(
                    json.dumps(
                        {"oracle_ndcg@10": value, "combination_ndcg@10": value, "oracle_minus_combination": 0, "model_win_rates": {"EASE": 0.5}}
                    )
                )
            with patch.object(multi, "TABLES", {"task24": {"specialized_retrieval": ["strategy"]}}):
                multi.aggregate_tables(root, [2020, 2021])
            frame = pd.read_csv(root / "aggregate/task24/specialized_retrieval_summary.csv")
            score = frame[frame.measure == "ndcg@10"].iloc[0]
            self.assertEqual(score.n_seeds, 2)
            self.assertAlmostEqual(score["mean"], 0.3)
            self.assertAlmostEqual(score["std"], np.sqrt(0.02))
            self.assertFalse(frame.measure.str.contains("ci95").any())
            self.assertEqual(frame[frame.measure == "tail_hit_rate"].iloc[0].n_seeds, 0)


if __name__ == "__main__":
    unittest.main()
