import os
import argparse
from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

import torch
import pandas as pd
from export_scores import fit, load

SEEDS = [2020, 2021, 2022, 2023, 2024]
MODELS = [
    "Random",
    "Pop",
    "ItemKNN",
    "UserKNN",
    "BPR",
    "NeuMF",
    "LightGCN",
    "NGCF",
    "EASE",
    "SLIMElastic",
    "FISM",
]
FOLDER = Path("results/seeds")


def repeat(name, seed):
    torch.set_num_threads(1)
    overrides = {"seed": seed, "checkpoint_dir": f"results/checkpoints/{seed}"}
    _, loaders, _, trainer, _, epochs = fit(load(name, "parameters/tuned", overrides))
    test = trainer.evaluate(loaders[2], load_best_model=True)
    return {"model": name, "seed": seed, "epochs": epochs} | dict(test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("models", nargs="*", default=MODELS)
    args = parser.parse_args()
    jobs = [(name, seed) for name in args.models for seed in SEEDS]
    jobs.sort(key=lambda job: job[0] not in ("NGCF", "FISM", "LightGCN"))
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        frame = pd.DataFrame(pool.map(repeat, *zip(*jobs)))
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, runs in frame.groupby("model"):
        runs.to_csv(FOLDER / f"{name}.csv", index=False)
    metrics = frame.drop(columns=["seed", "epochs"]).groupby("model", sort=False)
    table = metrics.agg(["mean", "std"]).round(4)
    table.to_csv(FOLDER / "summary.csv")
    print(table)
