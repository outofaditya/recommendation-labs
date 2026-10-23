import numpy as np
from source.reranking.greedy import greedy
from source.metrics.societal import jaccard


# maximal marginal relevance where a movie gains by being unlike its closest pick so far
def mmr(scores, hidden, genres, lam):
    similar = jaccard(genres)

    def gain(pool, chosen):
        if chosen.shape[1] == 0:
            return np.zeros(pool.shape)
        return 1 - similar[pool[:, :, None], chosen[:, None, :]].max(axis=2)

    return greedy(scores, hidden, gain, lam)
