import argparse
from pathlib import Path

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


def repeat(name):
    rows = []
    for seed in SEEDS:
        config = load(name, "parameters/tuned", {"seed": seed})
        _, loaders, _, trainer, _, epochs = fit(config)
        test = trainer.evaluate(loaders[2], load_best_model=True)
        rows.append({"seed": seed, "epochs": epochs} | dict(test))
    frame = pd.DataFrame(rows)
    frame.to_csv(FOLDER / f"{name}.csv", index=False)
    return frame.drop(columns=["seed", "epochs"]).agg(["mean", "std"]).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=MODELS)
    FOLDER.mkdir(parents=True, exist_ok=True)
    summary = {name: repeat(name) for name in parser.parse_args().models}
    table = pd.concat(summary).unstack().round(4)
    table.to_csv(FOLDER / "summary.csv")
    print(table)
