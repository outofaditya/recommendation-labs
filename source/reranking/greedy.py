import numpy as np
from source.metrics.accuracy import K

DEPTH = 100


# each user's best unseen movies in the order top() ranks them
def shortlist(scores, hidden, depth=DEPTH):
    masked = np.where(hidden, -np.inf, scores)
    pool = np.argsort(-masked, axis=1, kind="stable")[:, :depth]
    return pool, np.isfinite(np.take_along_axis(masked, pool, axis=1))


# per user min max of the shortlist so lambda means the same for every model
def scale(values, free):
    low = np.min(values, axis=1, where=free, initial=np.inf, keepdims=True)
    high = np.max(values, axis=1, where=free, initial=-np.inf, keepdims=True)
    span = np.where(high > low, high - low, 1)
    return np.where(free, (values - low) / span, 0)


# fills the list one slot at a time trading relevance against each candidate's gain in its paper's units
def greedy(scores, hidden, gain, lam, k=K):
    pool, free = shortlist(scores, hidden)
    relevance = scale(np.take_along_axis(scores, pool, axis=1).astype(np.float64), free)
    rows = np.arange(len(pool))
    chosen = np.zeros((len(pool), 0), dtype=int)
    for _ in range(k):
        value = (1 - lam) * relevance
        if lam > 0:
            value += lam * gain(pool, chosen)
        best = np.where(free, value, -np.inf).argmax(axis=1)
        free[rows, best] = False
        chosen = np.column_stack([chosen, pool[rows, best]])
    return chosen
