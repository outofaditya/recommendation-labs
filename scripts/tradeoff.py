import json

import numpy as np
import pandas as pd
from rerank import BASES, FOLDER, RERANKERS
from source.data import genres, scores, split
from source.metrics.accuracy import accuracy, top
from source.metrics.societal import deviation, exposure, gini, ild, miscalibration, tail, tastes

TASTES = ["niche", "diverse", "blockbuster"]


# every metric of a set of lists where hidden is what the lists could not show
def measure(ranked, hidden, relevant, flags):
    shown, drift, taste = exposure(ranked, hidden.shape[1]), deviation(ranked, hidden), tastes(hidden)
    groups = {name: drift[taste == k].mean() for k, name in enumerate(TASTES)}
    row = {"ndcg": accuracy(ranked, relevant)["ndcg@10"], "ild": ild(ranked, flags).mean()}
    row |= {"miscalibration": miscalibration(ranked, hidden, flags).mean(), "tail": tail(ranked, hidden).mean()}
    row |= {"coverage": (shown > 0).mean(), "gini": gini(shown), "deviation": drift.mean()} | groups
    row["gap"] = max(groups.values()) - min(groups.values())
    return {name: round(float(value), 4) for name, value in row.items()}


# validation curves over every lambda and test before and after the chosen one
def tables():
    curves, tests = [], []
    for base in BASES:
        values, users, items = scores(base)
        masks, flags = split(users, items), genres(items)
        seen = masks["train"] | masks["valid"]
        tests.append({"model": base, "reranker": "Original", "lambda": 0.0} | measure(top(values, seen), seen, masks["test"], flags))
        for name in RERANKERS:
            saved = np.load(FOLDER / "ranked" / f"{base}-{name}.npz")
            chosen = json.loads((FOLDER / "metrics" / f"{base}-{name}.json").read_text())["lambda"]
            for lam, ranked in zip(saved["lambdas"], saved["valid"]):
                row = {"model": base, "reranker": name, "lambda": float(lam), "chosen": bool(lam == chosen)}
                curves.append(row | measure(ranked, masks["train"], masks["valid"], flags))
            tests.append({"model": base, "reranker": name, "lambda": chosen} | measure(saved["test"], seen, masks["test"], flags))
    return pd.DataFrame(curves), pd.DataFrame(tests)


if __name__ == "__main__":
    curves, tests = tables()
    curves.to_csv(FOLDER / "curves.csv", index=False)
    tests.to_csv(FOLDER / "test.csv", index=False)
    print(tests.to_string(index=False))
