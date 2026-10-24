import os
import argparse
from functools import partial
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

from export_scores import export
from source.data import MODELS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--seed", type=int, help="Override the configuration seed (use the same value for a shared split)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        help="Only run the selected models; Random is added automatically when needed to create the split.",
    )
    args = parser.parse_args()
    selected = list(args.models) if args.models else list(MODELS)
    # Random is the fastest model and creates the shared split.  It is needed
    # even when the caller requests a subset of the learned models.
    if "Random" not in selected:
        selected.append("Random")
    ordered = [name for name in MODELS if name in selected]
    run = partial(export, tuned=args.tuned, seed=args.seed)
    if "Random" in ordered:
        run("Random")
        ordered.remove("Random")
    if ordered:
        with ProcessPoolExecutor(min(args.workers, len(ordered)), mp_context=get_context("spawn")) as pool:
            list(pool.map(run, ordered))
