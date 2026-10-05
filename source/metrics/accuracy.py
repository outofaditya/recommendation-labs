import numpy as np

K = 10
DISCOUNT = 1 / np.log2(np.arange(2, K + 2))


def top(scores, hidden):
    masked = np.where(hidden, -np.inf, scores)
    return np.argsort(-masked, axis=1, kind="stable")[:, :K]


# users without relevant items are skipped as in recbole
def accuracy(ranked, relevant):
    hits = np.take_along_axis(relevant, ranked, axis=1)
    count = relevant.sum(axis=1)
    hits, count = hits[count > 0], count[count > 0]
    found = hits.any(axis=1)
    ideal = np.cumsum(DISCOUNT)[np.minimum(count, K) - 1]
    users = {
        "recall@10": hits.sum(axis=1) / count,
        "mrr@10": np.where(found, 1 / (hits.argmax(axis=1) + 1), 0),
        "ndcg@10": (hits * DISCOUNT).sum(axis=1) / ideal,
        "hit@10": found,
        "precision@10": hits.sum(axis=1) / K,
    }
    return {name: round(float(values.mean()), 4) for name, values in users.items()}
