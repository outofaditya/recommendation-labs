import json

import numpy as np
import pandas as pd
from rerank import BASES, FOLDER, RERANKERS
from source.data import genres, scores, split
from source.analysis.groups import movies, people
from source.metrics.accuracy import DISCOUNT, quality, top
from scipy.stats import false_discovery_control, wilcoxon
from source.metrics.societal import exposure, ild, miscalibration, tail, upd


# per user quality of a set of lists with nan where a metric skips the user
def card(ranked, hidden, relevant, flags):
    ndcg = np.full(len(ranked), np.nan)
    keep, values = quality(ranked, relevant)
    ndcg[keep] = values["ndcg@10"]
    row = {"ndcg": ndcg, "ild": ild(ranked, flags), "miscalibration": miscalibration(ranked, hidden, flags)}
    return row | {"tail": tail(ranked, hidden), "upd": upd(ranked, hidden)}


# per user recall of one movie group's test movies as in 2.5 with nan for users who hold none
def recall(ranked, relevant, group):
    values = np.full(len(ranked), np.nan)
    keep, metrics = quality(ranked, relevant & group)
    values[keep] = metrics["recall@10"]
    return values


# the same users before and after where wilcoxon drops the users re-ranking leaves unchanged
def paired(before, after):
    both = np.isfinite(before) & np.isfinite(after)
    change = after[both] - before[both]
    p = wilcoxon(change).pvalue if change.any() else 1.0
    return {"before": before[both].mean(), "after": after[both].mean(), "change": change.mean(), "p": p}


# share of position weighted exposure a movie group gets and the share of it shown at all
def shares(ranked, count, group):
    exposed, shown = exposure(ranked, count, DISCOUNT), exposure(ranked, count)
    return {"exposure": exposed[group].sum() / exposed.sum(), "coverage": (shown[group] > 0).mean()}


# group means of history and test size show whether a grouping still mixes in activity
def userwise(groups, sizes, before, after):
    for dimension, (labels, group) in groups.items():
        for k, label in enumerate(labels):
            rows = group == k
            info = {"side": "user", "dimension": dimension, "group": label, "size": int(rows.sum())}
            info |= {name: values[rows].mean() for name, values in sizes.items()}
            for metric in before:
                yield info | {"metric": metric} | paired(before[metric][rows], after[metric][rows])


def itemwise(groups, original, ranked, relevant):
    for dimension, (labels, group) in groups.items():
        for k, label in enumerate(labels):
            mask, count = group == k, relevant.shape[1]
            info = {"side": "item", "dimension": dimension, "group": label, "size": int(mask.sum())}
            yield info | {"metric": "recall"} | paired(recall(original, relevant, mask), recall(ranked, relevant, mask))
            old, new = shares(original, count, mask), shares(ranked, count, mask)
            for metric in old:
                yield info | {"metric": metric, "before": old[metric], "after": new[metric], "change": new[metric] - old[metric], "p": np.nan}


# every chosen re-ranking on test against its model's original lists per user and movie group
def tables():
    rows = []
    for base in BASES:
        values, users, items = scores(base)
        masks, flags = split(users, items), genres(items)
        seen, relevant = masks["train"] | masks["valid"], masks["test"]
        crowds, shelves = people(seen, users), movies(seen, items)
        sizes = {"history": seen.sum(axis=1), "tests": relevant.sum(axis=1)}
        original = top(values, seen)
        before = card(original, seen, relevant, flags)
        for name in RERANKERS:
            ranked = np.load(FOLDER / "ranked" / f"{base}-{name}.npz")["test"]
            lam = json.loads((FOLDER / "metrics" / f"{base}-{name}.json").read_text())["lambda"]
            found = [*userwise(crowds, sizes, before, card(ranked, seen, relevant, flags)), *itemwise(shelves, original, ranked, relevant)]
            rows += [{"model": base, "reranker": name, "lambda": lam} | row for row in found]
    table = pd.DataFrame(rows)
    tested = table.p.notna()
    table.loc[tested, "q"] = false_discovery_control(table.p[tested])
    return table.round({"before": 4, "after": 4, "change": 4, "history": 1, "tests": 1})


if __name__ == "__main__":
    table = tables()
    table.to_csv(FOLDER / "groups.csv", index=False)
    ndcg = table[table.metric == "ndcg"].assign(percent=lambda frame: 100 * frame.change / frame.before)
    print("test ndcg change in percent per user group")
    print(ndcg.pivot_table(index=["dimension", "group"], columns=["model", "reranker"], values="percent", sort=False).round(1).to_string())
