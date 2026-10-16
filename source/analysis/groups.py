import numpy as np
import pandas as pd
from source.metrics.societal import head, tastes, thirds, tiers

TASTES = ["niche", "diverse", "blockbuster"]
TIERS = ["head", "mid", "tail"]
ACTIVITY = ["low", "mid-low", "mid-high", "high"]
POPULARITY = ["long-tail", "low-mid", "high-mid", "head"]


# equal sized quartiles with ties ranked in id order as the qcut of 2.5 does
def quartiles(values, ids):
    rank = np.empty(len(values))
    rank[np.lexsort((ids.astype(int), values))] = np.arange(1, len(values) + 1)
    return pd.qcut(rank, 4, labels=False)


# taste thirds drawn inside each activity quartile so every taste group holds the same mix of activity
def stratified(hidden, ids):
    share, level = head(hidden), quartiles(hidden.sum(axis=1), ids)
    taste = np.empty(len(share), dtype=int)
    for q in range(4):
        taste[level == q] = thirds(share[level == q])
    return taste


# user groupings as labels and a group per user where activity is shared with 2.5
def people(hidden, ids):
    groups = {"activity": (ACTIVITY, quartiles(hidden.sum(axis=1), ids)), "taste": (TASTES, tastes(hidden))}
    return groups | {"taste within activity": (TASTES, stratified(hidden, ids))}


# movie groupings as labels and a group per movie with our tiers beside the quartiles of 2.5
def movies(hidden, ids):
    return {"tier": (TIERS, tiers(hidden)), "popularity": (POPULARITY, quartiles(hidden.sum(axis=0), ids))}
