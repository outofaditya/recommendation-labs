import json

import numpy as np
import pandas as pd
from hybrids import ranks
from tradeoff import measure
from source.data import RESULTS, genres
from rerank import FOLDER, LAMBDAS, RERANKERS, choose
from source.hybrids.pool import MEMBERS, features
from source.metrics.accuracy import K, top

TUNED = json.loads((RESULTS / "hybrid" / "metrics" / "TunedMixed.json").read_text())
VOTERS = [MEMBERS.index(name) for name in TUNED["members"]]
OUTPUT = FOLDER / "hybrid"


# reciprocal rank fusion with the members and constant that won 1.5
def fuse(rank):
    return (1 / (TUNED["k"] + rank)).sum(axis=-1)


# a member's re-ranked list takes the first places and every other movie keeps its order behind it
def promote(rank, chosen):
    key = rank.astype(np.float64)
    np.put_along_axis(key, chosen, np.arange(-K, 0), axis=1)
    return np.argsort(np.argsort(key, axis=1), axis=1) + 1


# individual recommenders each with a re-ranker and then fused
def members(x, rank, hidden, flags, rerank, lam):
    lists = [promote(rank[..., j], rerank(x[..., j], hidden, flags, lam)) for j in range(x.shape[-1])]
    return top(fuse(np.stack(lists, axis=-1)), hidden)


# individual recommenders fused and the hybrid's list re-ranked afterwards
def hybrid(x, rank, hidden, flags, rerank, lam):
    return rerank(fuse(rank), hidden, flags, lam)


PIPELINES = {"Members": members, "Hybrid": hybrid}


def mark(ranked, count):
    mask = np.zeros((len(ranked), count), dtype=bool)
    np.put_along_axis(mask, ranked, True, axis=1)
    return mask


# what a pipeline swaps into the original hybrid's lists and at which slots since lower slots cost less ndcg
def swaps(ranked, base, relevant):
    final, swapped = mark(ranked, base.shape[1]), ~np.take_along_axis(base, ranked, axis=1)
    slots = np.broadcast_to(np.arange(1, K + 1), ranked.shape)[swapped]
    row = {"swapped": swapped.sum(axis=1).mean(), "hits_in": relevant[final & ~base].mean(), "hits_out": relevant[base & ~final].mean()}
    return final, row | {"slot": slots.mean(), "top3": (slots <= 3).mean()}


# how many members promote each movie and whether fusion keeps it since fusion rewards agreement
def agreement(x, hidden, flags, rerank, lam, final):
    count = hidden.shape[1]
    promoted = [mark(rerank(x[..., j], hidden, flags, lam), count) & ~mark(top(x[..., j], hidden), count) for j in range(x.shape[-1])]
    votes = np.sum(promoted, axis=0)
    shared = (votes > 1).sum() / (votes > 0).sum()
    return {"promoters": votes[votes > 0].mean(), "shared": shared, "kept_single": final[votes == 1].mean(), "kept_shared": final[votes > 1].mean()}


# every lambda of both pipelines is ranked on validation and only the chosen one is scored on test
def tables():
    x, masks, users, items = features()
    x, flags, seen = x[..., VOTERS], genres(items), masks["train"] | masks["valid"]
    valid, test = ranks(x, masks["train"]), ranks(x, seen)
    base = mark(top(fuse(valid), masks["train"]), len(items))
    original = {"pipeline": "Original", "reranker": "Mixed", "lambda": 0.0}
    curves, tests, evidence = [], [original | measure(top(fuse(test), seen), seen, masks["test"], flags)], []
    (OUTPUT / "ranked").mkdir(parents=True, exist_ok=True)
    for pipeline, build in PIPELINES.items():
        for name, rerank in RERANKERS.items():
            lists = np.stack([build(x, valid, masks["train"], flags, rerank, lam) for lam in LAMBDAS])
            rows = [measure(ranked, masks["train"], masks["valid"], flags) for ranked in lists]
            lam = choose([row["ndcg"] for row in rows])
            final = build(x, test, seen, flags, rerank, lam)
            path = OUTPUT / "ranked" / f"{pipeline}-{name}.npz"
            np.savez_compressed(path, valid=lists, test=final, lambdas=LAMBDAS, users=users, items=items)
            for value, row in zip(LAMBDAS, rows):
                curves.append({"pipeline": pipeline, "reranker": name, "lambda": float(value), "chosen": bool(value == lam)} | row)
            tests.append({"pipeline": pipeline, "reranker": name, "lambda": float(lam)} | measure(final, seen, masks["test"], flags))
            kept, row = swaps(lists[np.flatnonzero(LAMBDAS == lam)[0]], base, masks["valid"])
            if build is members:
                row |= agreement(x, masks["train"], flags, rerank, lam, kept)
            evidence.append({"pipeline": pipeline, "reranker": name, "lambda": float(lam)} | row)
            print(f"{pipeline:8} {name:11} lambda {lam:.2f}  test ndcg {tests[0]['ndcg']:.4f} -> {tests[-1]['ndcg']:.4f}")
    return pd.DataFrame(curves), pd.DataFrame(tests), pd.DataFrame(evidence).round(4)


if __name__ == "__main__":
    curves, tests, evidence = tables()
    curves.to_csv(OUTPUT / "curves.csv", index=False)
    tests.to_csv(OUTPUT / "test.csv", index=False)
    evidence.to_csv(OUTPUT / "evidence.csv", index=False)
    print(tests.to_string(index=False))
    print(evidence.to_string(index=False))
