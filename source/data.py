from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")


def scores(name, tuned=True):
    table = np.load(RESULTS / ("tuned" if tuned else "") / "scores" / f"{name}.npz")
    return table["scores"], table["users"], table["items"]


# boolean user by item masks aligned with the score tables
def split(users, items):
    rows = {user: i for i, user in enumerate(users)}
    cols = {item: i for i, item in enumerate(items)}
    masks = {}
    for name in ("train", "valid", "test"):
        frame = pd.read_csv(RESULTS / "split" / f"{name}.tsv", sep="\t", dtype=str)
        mask = np.zeros((len(users), len(items)), dtype=bool)
        mask[frame.iloc[:, 0].map(rows), frame.iloc[:, 1].map(cols)] = True
        masks[name] = mask
    return masks
