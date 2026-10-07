import numpy as np
from source.reranking.greedy import greedy
from source.reranking.calibration import calibrated, js

HEAD, TAIL = 0.2, 0.8


# head movies take the first fifth of all interactions and tail movies the last fifth
def tiers(hidden):
    counts = hidden.sum(axis=0)
    order = np.argsort(-counts, kind="stable")
    tier = np.empty(len(counts), dtype=int)
    tier[order] = np.searchsorted([HEAD, TAIL], np.cumsum(counts[order]) / counts.sum())
    return tier


# item side xquad where a movie gains by covering the tail or the rest as much as the user's history does
def xquad(scores, hidden, genres, lam):
    tail = tiers(hidden) == 2
    taste = hidden @ tail.astype(float) / hidden.sum(axis=1)

    # smooth coverage from the paper with the share of the list already in each category
    def gain(pool, chosen):
        size = chosen.shape[1]
        covered = tail[chosen].sum(axis=1, keepdims=True) / max(size, 1)
        rest = (1 - taste[:, None]) * covered**size
        return np.where(tail[pool], taste[:, None] * (1 - covered) ** size, rest)

    return greedy(scores, hidden, gain, lam)


# user side calibrated popularity so niche users get as much of the tail as their history holds
def cp(scores, hidden, genres, lam):
    shares = np.eye(3)[tiers(hidden)]
    return calibrated(scores, hidden, shares, lam, js)
