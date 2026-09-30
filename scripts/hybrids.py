import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from source.hybrids.pool import MEMBERS, candidates, features, save
from source.hybrids.side import K, embeddings, side, userknn
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


def best(x, masks, users):
    hidden, relevant = masks["train"][users], masks["valid"][users]
    runs = [ndcg(x[users, :, j], hidden, relevant) for j in range(len(MEMBERS))]
    return int(np.argmax(runs))


# users split into equal groups by train size
def groups(masks, count):
    order = np.argsort(masks["train"].sum(axis=1), kind="stable")
    return np.array_split(order, count)


# each activity group takes its best member on validation
def switching(x, masks, count=GROUPS):
    table, chosen = np.empty(x.shape[:2]), []
    for group in groups(masks, count):
        member = best(x, masks, group)
        table[group] = x[group, :, member]
        size = masks["train"][group].sum(axis=1)
        span = [int(size.min()), int(size.max())]
        chosen.append({"train": span, "member": MEMBERS[member]})
    return table, {"groups": chosen}


# every member's rank of each movie with hidden movies last
def ranks(x, hidden):
    masked = np.where(hidden[..., None], -np.inf, x)
    return np.argsort(np.argsort(-masked, axis=1), axis=1) + 1


# reciprocal rank fusion of every member over the unseen movies
def mixed(x, masks):
    seen = masks["train"] | masks["valid"]
    return (1 / (FUSION + ranks(x, seen))).sum(axis=-1), {"k": FUSION}


# the first stage shortlists unhidden movies and the second orders them
def shortlisted(first, second, hidden, size):
    shortlist = np.argsort(np.where(hidden, np.inf, -first), axis=1)[:, :size]
    table = np.full(first.shape, -np.inf)
    np.put_along_axis(table, shortlist, np.take_along_axis(second, shortlist, 1), 1)
    return table


def cascade(x, masks):
    seen = masks["train"] | masks["valid"]
    first, second = (x[..., MEMBERS.index(name)] for name in STAGES)
    details = {"stages": STAGES, "shortlist": SHORTLIST}
    return shortlisted(first, second, seen, SHORTLIST), details


# member scores beside side data for each candidate pair
def pairs(x, train):
    person, movie = side()
    movies = np.column_stack([movie, train.sum(axis=0)])
    people = np.column_stack([person, train.sum(axis=1)])

    def rows(pool):
        users, items = np.nonzero(pool)
        return np.column_stack([x[pool], people[users], movies[items]])

    return rows


def booster(depth=None, rate=0.1):
    occupation = [len(MEMBERS) + 2]
    return HistGradientBoostingClassifier(
        max_depth=depth,
        learning_rate=rate,
        categorical_features=occupation,
        random_state=2020,
    )


# one boosted learner over every source learns where each member is right
def combination(x, masks, depth=None, rate=0.1):
    train, seen = masks["train"], masks["train"] | masks["valid"]
    pool, test = candidates(x, train), candidates(x, seen)
    rows, model = pairs(x, train), booster(depth, rate)
    model.fit(rows(pool), masks["valid"][pool])
    table = np.full(train.shape, -np.inf)
    table[test] = model.decision_function(rows(test))
    return table, {"iterations": model.n_iter_}


# the densifier's top unseen picks join train before userknn is refit
def augmentation(x, masks, pseudo=PSEUDO, k=K):
    train = masks["train"]
    first = np.where(train, -np.inf, x[..., MEMBERS.index(DENSIFIER)])
    picks = np.argpartition(-first, pseudo, axis=1)[:, :pseudo]
    dense = train.copy()
    np.put_along_axis(dense, picks, True, axis=1)
    return userknn(dense, dense, k), {"densifier": DENSIFIER, "pseudo": pseudo, "k": k}


# userknn takes its peers from learned embeddings instead of raw ratings
def metalevel(x, masks, source=EMBEDDED, k=K):
    table = userknn(embeddings(source), masks["train"], k)
    return table, {"embeddings": source, "k": k}


if __name__ == "__main__":
    x, masks, users, items = features()
    seen = masks["train"] | masks["valid"]
    for build in (switching, mixed, cascade, combination, augmentation, metalevel):
        table, details = build(x, masks)
        test = accuracy(top(table, seen), masks["test"])
        save(build.__name__.title(), table, users, items, details | {"test": test})
