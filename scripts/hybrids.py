import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from source.hybrids.pool import MEMBERS, candidates, features, save
from source.hybrids.side import embeddings, side, userknn
from source.metrics.accuracy import accuracy, top

GROUPS = 3
PSEUDO = 10
FUSION = 60
SHORTLIST = 50
DENSIFIER = "EASE"
EMBEDDED = "LightGCN"
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


# member scores beside side data for each candidate pair
def pairs(x, train):
    person, movie = side()
    movies = np.column_stack([movie, train.sum(axis=0)])
    people = np.column_stack([person, train.sum(axis=1)])

    def rows(pool):
        users, items = np.nonzero(pool)
        return np.column_stack([x[pool], people[users], movies[items]])

    return rows


# one boosted learner over every source learns where each member is right
def combination(x, masks):
    train, seen = masks["train"], masks["train"] | masks["valid"]
    rows, occupation = pairs(x, train), [len(MEMBERS) + 2]
    pool, test = candidates(x, train), candidates(x, seen)
    model = HistGradientBoostingClassifier(
        categorical_features=occupation, random_state=2020
    )
    model.fit(rows(pool), masks["valid"][pool])
    table = np.full(train.shape, -np.inf)
    table[test] = model.decision_function(rows(test))
    return table, {"iterations": model.n_iter_}


# the densifier's top unseen picks join train before userknn is refit
def augmentation(x, masks, pseudo=PSEUDO):
    train = masks["train"]
    first = np.where(train, -np.inf, x[..., MEMBERS.index(DENSIFIER)])
    picks = np.argpartition(-first, pseudo, axis=1)[:, :pseudo]
    dense = train.copy()
    np.put_along_axis(dense, picks, True, axis=1)
    return userknn(dense, dense), {"densifier": DENSIFIER, "pseudo": pseudo}


# userknn takes its peers from learned embeddings instead of raw ratings
def metalevel(x, masks):
    table = userknn(embeddings(EMBEDDED), masks["train"])
    return table, {"embeddings": EMBEDDED}


if __name__ == "__main__":
    x, masks, users, items = features()
    seen = masks["train"] | masks["valid"]
    for build in (switching, mixed, cascade, combination, augmentation, metalevel):
        table, details = build(x, masks)
        test = accuracy(top(table, seen), masks["test"])
        save(build.__name__.title(), table, users, items, details | {"test": test})
