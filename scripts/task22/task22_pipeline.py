"""Run the complete deterministic Task 2.2 evaluation pipeline.

This module is the single orchestration entry point for Experiments 1--5.
It never trains a model: it evaluates saved individual and hybrid exports.
The repeated-seed runner imports :func:`run_pipeline` so that the ordinary
and repeated evaluations cannot silently diverge.
"""

import argparse
from pathlib import Path

from scripts.task22.analyze_tradeoffs import run as run_tradeoffs
from scripts.task22.compare_beyond_accuracy import run as run_beyond_accuracy
from scripts.task22.compare_hybrids import run as run_hybrids
from scripts.task22.compare_individuals import run as run_individuals
from scripts.task22.quantify_baselines import run as run_baselines


DEFAULT_KS = (10,)


def run_pipeline(
    results=Path("results"),
    output_root=Path("results/task2"),
    ks=DEFAULT_KS,
    items=Path("data/ml-100k/ml-100k.item"),
):
    """Run Experiments 1--5 for every requested K and return output paths."""
    results = Path(results)
    output_root = Path(output_root)
    ks = tuple(dict.fromkeys(int(k) for k in ks))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    outputs = []
    for k in ks:
        e1 = output_root / "experiment1" if len(ks) == 1 else output_root / "experiment1" / f"k_{k}"
        e2 = output_root / "experiment2" if len(ks) == 1 else output_root / "experiment2" / f"k_{k}"
        e3 = output_root / "experiment3" if len(ks) == 1 else output_root / "experiment3" / f"k_{k}"
        e4 = output_root / "experiment4" if len(ks) == 1 else output_root / "experiment4" / f"k_{k}"
        e5 = output_root / "experiment5" if len(ks) == 1 else output_root / "experiment5" / f"k_{k}"

        run_individuals(results, e1, k, items)
        run_hybrids(results, e2, k, items)
        run_baselines(e2 / "summary.csv", e2 / "protocol.json", e3)
        run_beyond_accuracy(e2 / "summary.csv", e2 / "protocol.json", e4, k)
        run_tradeoffs(e2 / "summary.csv", e2 / "protocol.json", e5, k)
        print(f"Task 2.2 K={k}: Experiments 1–5 complete", flush=True)
        outputs.append({"k": k, "experiment1": e1, "experiment2": e2, "experiment3": e3, "experiment4": e4, "experiment5": e5})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task2"))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--items", type=Path, default=Path("data/ml-100k/ml-100k.item"))
    args = parser.parse_args()
    outputs = run_pipeline(args.results, args.output, args.ks, args.items)
    print(f"Completed Task 2.2 Experiments 1--5 for K={[entry['k'] for entry in outputs]}")


if __name__ == "__main__":
    main()
