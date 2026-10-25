"""Provenance and CPU-only execution tests for saved-export reanalysis."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.task2 import analyze as rerun


class ReanalysisTests(unittest.TestCase):
    def test_original_stage_hashes_are_required_and_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            source = snapshot / "results/task2/multiseed"
            work = source / "runs/seed_7"
            work.mkdir(parents=True)
            original_input = snapshot / "code.py"
            original_input.write_text("original")
            protocol = {"seeds": [7], "inputs": {"code.py": rerun.digest(original_input)}}
            (source / "run_protocol.json").write_text(json.dumps(protocol))
            artifact = work / "saved.npz"
            artifact.write_bytes(b"verified original scores")
            for name in ["train_A", "hybrids"]:
                rerun.record(work / "stages" / f"{name}.json", work, rerun.digest(source / "run_protocol.json") + ":7", [artifact])
            (work / "results/split").mkdir(parents=True)
            for name in ("train", "valid", "test"):
                (work / "results/split" / f"{name}.tsv").write_text("user\titem\n")
            with patch.object(rerun, "SOURCE_MODELS", ["A"]):
                saved, evidence = rerun.verify_source(source)
                self.assertEqual(saved, protocol)
                self.assertEqual(len(evidence), 2)
                artifact.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "failed validation"):
                    rerun.verify_source(source)

    def test_subprocess_is_cpu_only_and_uses_analysis_module(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            with patch.object(rerun.subprocess, "run") as execute:
                rerun.execute("scripts.task23.analyze_contributions", [], work, work / "analysis.log", 2)
            args, kwargs = execute.call_args
            self.assertEqual(args[0][1:], ["-m", "scripts.task23.analyze_contributions"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "2")
            self.assertEqual(kwargs["cwd"], work)
            self.assertTrue(kwargs["check"])


if __name__ == "__main__":
    unittest.main()
