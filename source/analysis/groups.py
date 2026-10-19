import numpy as np
import pandas as pd
from source.metrics.societal import head, tastes, thirds, tiers

TASTES = ["niche", "diverse", "blockbuster"]
TIERS = ["head", "mid", "tail"]
ACTIVITY = ["low", "mid-low", "mid-high", "high"]
POPULARITY = ["long-tail", "low-mid", "high-mid", "head"]
STRATA = 20


# equal sized bands with ties ranked in id order as the qcut of 2.5 does
def bands(values, ids, count):
    rank = np.empty(len(values))
    rank[np.lexsort((ids.astype(int), values))] = np.arange(1, len(values) + 1)
    return pd.qcut(rank, count, labels=False)


def quartiles(values, ids):
    return bands(values, ids, 4)


# taste thirds drawn inside twenty activity bands since inside quartiles niche users still held more test items
def stratified(hidden, ids):
    share, level = head(hidden), bands(hidden.sum(axis=1), ids, STRATA)
    taste = np.empty(len(share), dtype=int)
    for band in range(STRATA):
        taste[level == band] = thirds(share[level == band])
    return taste


# user groupings as labels and a group per user where activity is shared with 2.5
def people(hidden, ids):
    groups = {"activity": (ACTIVITY, quartiles(hidden.sum(axis=1), ids)), "taste": (TASTES, tastes(hidden))}
    return groups | {"taste within activity": (TASTES, stratified(hidden, ids))}


# movie groupings as labels and a group per movie with our tiers beside the quartiles of 2.5
def movies(hidden, ids):
    return {"tier": (TIERS, tiers(hidden)), "popularity": (POPULARITY, quartiles(hidden.sum(axis=0), ids))}
