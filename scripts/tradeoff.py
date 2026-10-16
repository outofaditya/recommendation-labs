import json

import numpy as np
import pandas as pd
from source.analysis.groups import TASTES
from rerank import BASES, FOLDER, RERANKERS
from source.data import genres, scores, split
from source.metrics.accuracy import DISCOUNT, quality, top
from source.metrics.societal import exposure, gini, gru, ild, miscalibration, parity, tail, tastes, tiers, treatment, upd


# every metric of a set of lists where hidden is what the lists could not show and the tail is the protected item group
def measure(ranked, hidden, relevant, flags):
    keep, users = quality(ranked, relevant)
    ndcg, taste, protected = users["ndcg@10"], tastes(hidden), tiers(hidden) == 2
    niche, blockbuster = taste == 0, taste == 2
    shown, exposed, deviation = exposure(ranked, len(protected)), exposure(ranked, len(protected), DISCOUNT), upd(ranked, hidden)
    row = {"ndcg": ndcg.mean(), "ild": ild(ranked, flags).mean()}
    row |= {"miscalibration": miscalibration(ranked, hidden, flags).mean(), "tail": tail(ranked, hidden).mean()}
    row |= {"coverage": (shown > 0).mean(), "gini": gini(shown), "demographic_parity": parity(exposed, protected)}
    row["disparate_treatment"] = treatment(exposed, relevant.sum(axis=0), protected)
    row |= {"upd": deviation.mean()} | {f"upd_{name}": deviation[taste == k].mean() for k, name in enumerate(TASTES)}
    row |= {"gru_upd": gru(deviation, niche, blockbuster), "gru_ndcg": gru(ndcg, niche[keep], blockbuster[keep])}
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
