import numpy as np
from source.reranking.greedy import greedy
from source.metrics.societal import js, tiers
from source.reranking.calibration import calibrated


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
