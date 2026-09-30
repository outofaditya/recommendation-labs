import json

import numpy as np
from export_scores import MODELS
from sklearn.linear_model import LogisticRegression
from source.data import RESULTS, scores, split
from source.metrics.accuracy import accuracy, top

MEMBERS = [name for name in MODELS if name != "Random"]
DEPTH = 100
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


if __name__ == "__main__":
    x, masks, users, items = features()
    pool = candidates(x, masks["train"])
    fit = LogisticRegression(max_iter=1000).fit(x[pool], masks["valid"][pool])
    seen = masks["train"] | masks["valid"]
    hybrid = np.where(candidates(x, seen), x @ fit.coef_[0], -np.inf)
    test = accuracy(top(hybrid, seen), masks["test"])
    weights = dict(zip(MEMBERS, fit.coef_[0].round(4).tolist()))

    (FOLDER / "scores").mkdir(parents=True, exist_ok=True)
    (FOLDER / "metrics").mkdir(parents=True, exist_ok=True)
    table = hybrid.astype(np.float32)
    np.savez_compressed(
        FOLDER / "scores" / "Weighted.npz", scores=table, users=users, items=items
    )
    coverage = round(float(masks["valid"][pool].sum() / masks["valid"].sum()), 4)
    metrics = {"rows": int(pool.sum()), "coverage": coverage, "weights": weights}
    metrics["test"] = test
    (FOLDER / "metrics" / "Weighted.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
