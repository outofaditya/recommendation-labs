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
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    run = partial(export, tuned=args.tuned)
    # the fastest model writes the split before the pool compares against it
    run(MODELS[-1])
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(run, MODELS[:-1]))
