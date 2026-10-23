from source.reranking.greedy import greedy
from source.metrics.societal import ALPHA, kl, profile, spread


# a movie gains by pulling the list's category mix toward the mix of the user's history
def calibrated(scores, hidden, shares, lam, divergence):
    history = profile(hidden, shares)

    def gain(pool, chosen):
        mix = (shares[chosen].sum(axis=1)[:, None] + shares[pool]) / (chosen.shape[1] + 1)
        return -divergence(history[:, None], mix)

    return greedy(scores, hidden, gain, lam)


# steck's genre calibration where a movie's weight is split evenly over its genres
def steck(scores, hidden, genres, lam):
    # the small share of the profile keeps the divergence finite
    def divergence(p, q):
        return kl(p, (1 - ALPHA) * q + ALPHA * p)

    return calibrated(scores, hidden, spread(genres), lam, divergence)
