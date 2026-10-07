import numpy as np
from source.reranking.greedy import greedy

ALPHA = 0.01


def kl(p, q):
    p, q = np.broadcast_arrays(p, q)
    ratio = np.divide(p, q, out=np.ones(p.shape), where=p > 0)
    return np.sum(p * np.log(ratio), axis=-1)


def js(p, q):
    m = (p + q) / 2
    return (kl(p, m) + kl(q, m)) / 2


# a movie gains by pulling the list's category mix toward the mix of the user's history
def calibrated(scores, hidden, shares, lam, divergence):
    profile = hidden @ shares / hidden.sum(axis=1, keepdims=True)

    def gain(pool, chosen):
        mix = (shares[chosen].sum(axis=1)[:, None] + shares[pool]) / (chosen.shape[1] + 1)
        return -divergence(profile[:, None], mix)

    return greedy(scores, hidden, gain, lam)


# steck's genre calibration where a movie's weight is split evenly over its genres
def steck(scores, hidden, genres, lam):
    shares = genres / genres.sum(axis=1, keepdims=True)

    # the small share of the profile keeps the divergence finite
    def divergence(p, q):
        return kl(p, (1 - ALPHA) * q + ALPHA * p)

    return calibrated(scores, hidden, shares, lam, divergence)
