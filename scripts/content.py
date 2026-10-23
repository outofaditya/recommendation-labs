import ast
import json
from pathlib import Path
from itertools import product

import yaml
import numpy as np
import pandas as pd
from source.content import features, predict
from source.data import RESULTS, scores, split
from source.metrics.accuracy import accuracy, top

NAME = "ContentBased"
BASE = Path("parameters") / f"{NAME}.yaml"
SEARCH = Path("parameters/search") / f"{NAME}.hyper"
TUNED = Path("parameters/tuned") / f"{NAME}.yaml"


# every point of the search file where k only applies to the neighbourhood method
def grid():
    lines = [line.split(maxsplit=2) for line in SEARCH.read_text().splitlines() if line.strip()]
    keys = [key for key, *_ in lines]
    points = [dict(zip(keys, values)) for values in product(*(ast.literal_eval(v) for *_, v in lines))]
    return [p for p in points if p["method"] == "knn" or p["k"] is None]


def table(params, masks, items):
    matrix = features(items, params["decade"], params["idf"])
    return predict(masks["train"], matrix, params["method"], params["k"])


# profiles come from train only as every other model is trained on train only
def export(params, masks, users, items, tuned):
    scored = table(params, masks, items)
    valid = accuracy(top(scored, masks["train"]), masks["valid"])
    test = accuracy(top(scored, masks["train"] | masks["valid"]), masks["test"])
    out = RESULTS / ("tuned" if tuned else "")
    for kind in ("scores", "metrics"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores" / f"{NAME}.npz", scores=scored.astype(np.float32), users=users, items=items)
    metrics = {"params": params, "valid": valid, "test": test}
    (out / "metrics" / f"{NAME}.json").write_text(json.dumps(metrics, indent=2))
    print(NAME, "tuned" if tuned else "base", metrics["test"])


def tune(masks, items):
    rows = []
    for params in grid():
        rows.append(params | accuracy(top(table(params, masks, items), masks["train"]), masks["valid"]))
        print(NAME, len(rows), params, rows[-1]["ndcg@10"], flush=True)
    trials = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False, kind="stable")
    (RESULTS / "tuning").mkdir(parents=True, exist_ok=True)
    trials.to_csv(RESULTS / "tuning" / f"{NAME}.csv", index=False)
    best = rows[trials.index[0]]
    return {key: best[key] for key in grid()[0]}


if __name__ == "__main__":
    _, users, items = scores("Pop", tuned=False)
    masks = split(users, items)
    base = yaml.safe_load(BASE.read_text())
    export({key: base[key] for key in grid()[0]}, masks, users, items, tuned=False)
    best = tune(masks, items)
    TUNED.parent.mkdir(parents=True, exist_ok=True)
    TUNED.write_text(yaml.safe_dump(base | best, sort_keys=False))
    export(best, masks, users, items, tuned=True)
