import numpy as np
from itertools import permutations, product
from hybrids import best, groups, ndcg, ranks, shortlisted, switching
from source.hybrids.pool import MEMBERS, candidates, features, folds, save
from source.metrics.accuracy import accuracy, top
from weighted import fit, validate

VOTERS = [2, 3, 5, 10]
COUNTS = [1, 2, 3, 4, 5]
DEPTHS = [25, 50, 100, 200]
SHORTLISTS = [20, 50, 100, 200]
FUSIONS = [1, 10, 30, 60, 100, 300]


# depth and strength are chosen together by user cross-validation
def weighted(x, masks):
    runs = {}
    for depth in DEPTHS:
        pool = candidates(x, masks["train"], depth)
        for strength, (scores, _) in validate(x, pool, masks).items():
            runs[depth, strength] = float(np.mean(scores))
    depth, strength = max(runs, key=runs.get)
    pool = candidates(x, masks["train"], depth)
    weights = fit(x[pool], masks["valid"][pool], strength)
    seen = masks["train"] | masks["valid"]
    table = np.where(candidates(x, seen, depth), x @ weights, -np.inf)
    cv = {f"{d} {c:g}": round(score, 4) for (d, c), score in runs.items()}
    chosen = {"depth": depth, "C": strength, "cv": cv}
    return table, chosen | {"weights": dict(zip(MEMBERS, weights.round(4).tolist()))}


# held users take the member their group picked on the other folds
def switched(x, masks, count):
    scores = []
    for held in folds(len(x)):
        table = np.empty(x.shape[:2])
        for group in groups(masks, count):
            rest = np.setdiff1d(group, held)
            table[group] = x[group, :, best(x, masks, rest)]
        hidden, relevant = masks["train"][held], masks["valid"][held]
        scores.append(ndcg(table[held], hidden, relevant))
    return float(np.mean(scores))


def switch(x, masks):
    runs = {count: switched(x, masks, count) for count in COUNTS}
    count = max(runs, key=runs.get)
    table, chosen = switching(x, masks, count)
    return table, chosen | {"cv": {k: round(v, 4) for k, v in runs.items()}}


# the best members by their own validation score fuse with each constant
def mix(x, masks):
    hidden, relevant = masks["train"], masks["valid"]
    alone = [ndcg(x[..., j], hidden, relevant) for j in range(len(MEMBERS))]
    order, valid = np.argsort(alone)[::-1], ranks(x, hidden)
    runs = {}
    for voters in VOTERS:
        for fusion in FUSIONS:
            fused = (1 / (fusion + valid[..., order[:voters]])).sum(axis=-1)
            runs[voters, fusion] = ndcg(fused, hidden, relevant)
    voters, fusion = max(runs, key=runs.get)
    seen = masks["train"] | masks["valid"]
    table = (1 / (fusion + ranks(x, seen)[..., order[:voters]])).sum(axis=-1)
    members = [MEMBERS[j] for j in order[:voters]]
    grid = {f"{v} {k}": round(score, 4) for (v, k), score in runs.items()}
    return table, {"members": members, "k": fusion, "valid": grid}


# every ordered pair of members with each shortlist size
def chain(x, masks):
    hidden, relevant, runs = masks["train"], masks["valid"], {}
    pairs = permutations(range(len(MEMBERS)), 2)
    for (first, second), size in product(pairs, SHORTLISTS):
        table = shortlisted(x[..., first], x[..., second], hidden, size)
        runs[first, second, size] = ndcg(table, hidden, relevant)
    first, second, size = max(runs, key=runs.get)
    seen = masks["train"] | masks["valid"]
    table = shortlisted(x[..., first], x[..., second], seen, size)
    leaders = sorted(runs, key=runs.get, reverse=True)[:10]
    grid = {f"{MEMBERS[a]} {MEMBERS[b]} {n}": runs[a, b, n] for a, b, n in leaders}
    stages = MEMBERS[first], MEMBERS[second]
    return table, {"stages": stages, "shortlist": size, "valid": grid}


if __name__ == "__main__":
    x, masks, users, items = features()
    seen = masks["train"] | masks["valid"]
    builds = {"Mixed": mix, "Cascade": chain, "Weighted": weighted, "Switching": switch}
    for name, build in builds.items():
        table, details = build(x, masks)
        test = accuracy(top(table, seen), masks["test"])
        save(f"Tuned{name}", table, users, items, details | {"test": test})
