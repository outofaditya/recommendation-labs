from pathlib import Path

import numpy as np
import pandas as pd

# slowest first so pools finish early
MODELS = [
    "FISM",
    "NGCF",
    "LightGCN",
    "BPR",
    "NeuMF",
    "SLIMElastic",
    "ItemKNN",
    "UserKNN",
    "EASE",
    "Pop",
    "Random",
]
RESULTS = Path("results")


def scores(name, tuned=True):
    table = np.load(RESULTS / ("tuned" if tuned else "") / "scores" / f"{name}.npz")
    return table["scores"], table["users"], table["items"]


# boolean user by item masks aligned with the score tables
def split(users, items, results=None):
    """Load disjoint masks in export order, optionally from another results root."""
    folder = (RESULTS if results is None else Path(results)) / "split"
    rows, cols = pd.Index(np.asarray(users).astype(str)), pd.Index(np.asarray(items).astype(str))
    if not rows.is_unique or not cols.is_unique:
        raise ValueError("Duplicate user/item IDs in score exports")
    masks = {}
    for name in ("train", "valid", "test"):
        frame = pd.read_csv(folder / f"{name}.tsv", sep="\t", dtype=str)
        if frame.shape[1] < 2 or frame.empty:
            raise ValueError(f"Empty or malformed {name} split")
        u, i = rows.get_indexer(frame.iloc[:, 0]), cols.get_indexer(frame.iloc[:, 1])
        if (u < 0).any() or (i < 0).any():
            raise ValueError(f"Unmapped user/item IDs in {name}")
        mask = np.zeros((len(users), len(items)), dtype=bool)
        mask[u, i] = True
        if any(np.any(mask & previous) for previous in masks.values()):
            raise ValueError(f"Overlapping interactions in {name} split")
        masks[name] = mask
    return masks


# movie by genre indicators aligned with the score tables
def genres(items):
    frame = pd.read_csv("data/ml-100k/ml-100k.item", sep="\t", dtype=str)
    labels = frame.set_index("item_id:token").loc[items, "class:token_seq"]
    return labels.str.get_dummies(sep=" ").to_numpy(bool)
