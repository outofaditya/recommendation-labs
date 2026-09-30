import json

import numpy as np
from export_scores import MODELS
from sklearn.linear_model import LogisticRegression
from source.data import RESULTS, scores, split
from source.metrics.accuracy import accuracy, top

MEMBERS = [name for name in MODELS if name != "Random"]
DEPTH = 100
FOLDS = 5
GRID = [10.0**power for power in range(-7, 1)]
FOLDER = RESULTS / "hybrid"


# each member standardised per user over the movies outside train
def features():
    tables = [scores(name) for name in MEMBERS]
    _, users, items = tables[0]
    assert all((u == users).all() and (i == items).all() for _, u, i in tables)
    masks = split(users, items)
    stacked = np.stack([table for table, *_ in tables], axis=-1).astype(np.float64)
    outside = ~masks["train"][..., None]
    mean = stacked.mean(axis=1, keepdims=True, where=outside)
    std = stacked.std(axis=1, keepdims=True, where=outside)
    return (stacked - mean) / std, masks, users, items


# union of every member's top movies outside the hidden ones
def candidates(x, hidden):
    pool = np.zeros(hidden.shape, dtype=bool)
    for member in np.moveaxis(x, -1, 0):
        masked = np.where(hidden, -np.inf, member)
        best = np.argpartition(-masked, DEPTH, axis=1)[:, :DEPTH]
        np.put_along_axis(pool, best, True, axis=1)
    return pool


def fit(x, labels, strength):
    return LogisticRegression(C=strength, max_iter=1000).fit(x, labels).coef_[0]


def rank(x, weights, pool, hidden, relevant):
    return accuracy(top(np.where(pool, x @ weights, -np.inf), hidden), relevant)


# each strength is fitted on four folds of users and scored on the fifth
def validate(x, pool, masks):
    users = np.random.RandomState(2020).permutation(len(x))
    folds, runs = np.array_split(users, FOLDS), {}
    for strength in GRID:
        ndcg, weights = [], []
        for held in folds:
            rows = pool.copy()
            rows[held] = False
            weights.append(fit(x[rows], masks["valid"][rows], strength))
            hidden, relevant = masks["train"][held], masks["valid"][held]
            ndcg.append(rank(x[held], weights[-1], pool[held], hidden, relevant))
        runs[strength] = [run["ndcg@10"] for run in ndcg], np.array(weights)
    return runs


if __name__ == "__main__":
    x, masks, users, items = features()
    pool = candidates(x, masks["train"])
    runs = validate(x, pool, masks)
    strength = max(GRID, key=lambda strength: np.mean(runs[strength][0]))
    weights = fit(x[pool], masks["valid"][pool], strength)
    seen = masks["train"] | masks["valid"]
    hybrid = np.where(candidates(x, seen), x @ weights, -np.inf)
    test = accuracy(top(hybrid, seen), masks["test"])
    spread = runs[strength][1].std(axis=0)

    (FOLDER / "scores").mkdir(parents=True, exist_ok=True)
    (FOLDER / "metrics").mkdir(parents=True, exist_ok=True)
    table = hybrid.astype(np.float32)
    np.savez_compressed(
        FOLDER / "scores" / "Weighted.npz", scores=table, users=users, items=items
    )
    coverage = round(float(masks["valid"][pool].sum() / masks["valid"].sum()), 4)
    cv = {f"{c:g}": round(float(np.mean(runs[c][0])), 4) for c in GRID}
    metrics = {"rows": int(pool.sum()), "coverage": coverage, "cv": cv, "C": strength}
    metrics["weights"] = dict(zip(MEMBERS, weights.round(4).tolist()))
    metrics["spread"] = dict(zip(MEMBERS, spread.round(4).tolist()))
    metrics["test"] = test
    (FOLDER / "metrics" / "Weighted.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
