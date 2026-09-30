import json

import numpy as np
from source.data import MODELS, RESULTS, scores, split

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


def save(name, table, users, items, metrics):
    for kind in ("scores", "metrics"):
        (FOLDER / kind).mkdir(parents=True, exist_ok=True)
    scores = table.astype(np.float32)
    path = FOLDER / "scores" / f"{name}.npz"
    np.savez_compressed(path, scores=scores, users=users, items=items)
    (FOLDER / "metrics" / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    print(name, json.dumps(metrics, indent=2))
