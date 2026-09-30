import numpy as np
from source.hybrids.pool import MEMBERS, features, save
from source.metrics.accuracy import accuracy, top

GROUPS = 3
FUSION = 60
SHORTLIST = 50
STAGES = "EASE", "NeuMF"


def ndcg(scores, hidden, relevant):
    return accuracy(top(scores, hidden), relevant)["ndcg@10"]


# each activity group takes its best member on validation
def switching(x, masks):
    order = np.argsort(masks["train"].sum(axis=1), kind="stable")
    table, chosen = np.empty(x.shape[:2]), []
    for group in np.array_split(order, GROUPS):
        hidden, relevant = masks["train"][group], masks["valid"][group]
        runs = [ndcg(x[group, :, j], hidden, relevant) for j in range(len(MEMBERS))]
        best = int(np.argmax(runs))
        table[group] = x[group, :, best]
        size = masks["train"][group].sum(axis=1)
        chosen.append(
            {"train": [int(size.min()), int(size.max())], "member": MEMBERS[best]}
        )
    return table, {"groups": chosen}


# reciprocal rank fusion of every member over the unseen movies
def mixed(x, masks):
    seen = masks["train"] | masks["valid"]
    masked = np.where(seen[..., None], -np.inf, x)
    ranks = np.argsort(np.argsort(-masked, axis=1), axis=1) + 1
    return (1 / (FUSION + ranks)).sum(axis=-1), {"k": FUSION}


# the first stage shortlists unseen movies and the second orders them
def cascade(x, masks):
    seen = masks["train"] | masks["valid"]
    first, second = (x[..., MEMBERS.index(name)] for name in STAGES)
    shortlist = np.argsort(np.where(seen, np.inf, -first), axis=1)[:, :SHORTLIST]
    table = np.full(first.shape, -np.inf)
    np.put_along_axis(table, shortlist, np.take_along_axis(second, shortlist, 1), 1)
    return table, {"stages": STAGES, "shortlist": SHORTLIST}


if __name__ == "__main__":
    x, masks, users, items = features()
    seen = masks["train"] | masks["valid"]
    for build in (switching, mixed, cascade):
        table, details = build(x, masks)
        test = accuracy(top(table, seen), masks["test"])
        save(build.__name__.title(), table, users, items, details | {"test": test})
