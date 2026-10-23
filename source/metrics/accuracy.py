import numpy as np

K = 10
DISCOUNT = 1 / np.log2(np.arange(2, K + 2))


def top(scores, hidden, k=K):
    if k < 1:
        raise ValueError("k must be positive")
    masked = np.where(hidden, -np.inf, scores)
    return np.argsort(-masked, axis=1, kind="stable")[:, :k]


def ndcg_per_user(ranked, relevant, *, counts=None):
    """Binary NDCG in original user order; zero for users without positives.

    Pass overall relevance counts when computing a bucket's contribution so
    bucket DCGs share the overall IDCG and sum to overall NDCG.
    """
    hits = np.take_along_axis(relevant, ranked, axis=1)
    count = relevant.sum(axis=1) if counts is None else np.asarray(counts)
    k = ranked.shape[1]
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.r_[0.0, np.cumsum(discount)][np.minimum(count, k)]
    return np.divide((hits * discount).sum(axis=1), ideal, out=np.zeros(len(ranked)), where=ideal > 0)


# per user quality where users without relevant items are skipped as in recbole
def quality(ranked, relevant):
    hits = np.take_along_axis(relevant, ranked, axis=1)
    count = relevant.sum(axis=1)
    keep = count > 0
    hits, count = hits[keep], count[keep]
    found = hits.any(axis=1)
    users = {
        "recall@10": hits.sum(axis=1) / count,
        "mrr@10": np.where(found, 1 / (hits.argmax(axis=1) + 1), 0),
        "ndcg@10": ndcg_per_user(ranked, relevant)[keep],
        "hit@10": found,
        "precision@10": hits.sum(axis=1) / K,
    }
    return keep, users


def accuracy(ranked, relevant):
    users = quality(ranked, relevant)[1]
    return {name: round(float(values.mean()), 4) for name, values in users.items()}
