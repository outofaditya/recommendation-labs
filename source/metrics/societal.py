import numpy as np

ALPHA = 0.01
HEAD, TAIL = 0.2, 0.8


def kl(p, q):
    p, q = np.broadcast_arrays(p, q)
    ratio = np.divide(p, q, out=np.ones(p.shape), where=p > 0)
    return np.sum(p * np.log(ratio), axis=-1)


def js(p, q):
    m = (p + q) / 2
    return (kl(p, m) + kl(q, m)) / 2


# share of genres two movies have in common
def jaccard(genres):
    flags = genres.astype(np.float32)
    common = flags @ flags.T
    sizes = flags.sum(axis=1)
    return common / (sizes[:, None] + sizes[None, :] - common)


# each movie's weight split evenly over its genres as in steck
def spread(genres):
    return genres / genres.sum(axis=1, keepdims=True)


# share of each category in every user's history
def profile(hidden, shares):
    return hidden @ shares / hidden.sum(axis=1, keepdims=True)


# head movies take the first fifth of all interactions and tail movies the last fifth
def tiers(hidden):
    counts = hidden.sum(axis=0)
    order = np.argsort(-counts, kind="stable")
    tier = np.empty(len(counts), dtype=int)
    tier[order] = np.searchsorted([HEAD, TAIL], np.cumsum(counts[order]) / counts.sum())
    return tier


# niche diverse and blockbuster users as equal thirds by the head share of their history
def tastes(hidden):
    head = profile(hidden, np.eye(3)[tiers(hidden)])[:, 0]
    taste = np.empty(len(head), dtype=int)
    taste[np.argsort(head, kind="stable")] = np.arange(len(head)) * 3 // len(head)
    return taste


# per user mean genre distance over every pair in the list
def ild(ranked, genres):
    k = ranked.shape[1]
    distance = 1 - jaccard(genres)[ranked[:, :, None], ranked[:, None, :]]
    return distance.sum(axis=(1, 2)) / (k * (k - 1))


# per user kl of the list's genre mix from the history's as in steck
def miscalibration(ranked, hidden, genres):
    shares = spread(genres)
    p, q = profile(hidden, shares), shares[ranked].mean(axis=1)
    return kl(p, (1 - ALPHA) * q + ALPHA * p)


# per user js of the list's head mid tail mix from the history's as in abdollahpouri et al
def deviation(ranked, hidden):
    shares = np.eye(3)[tiers(hidden)]
    return js(profile(hidden, shares), shares[ranked].mean(axis=1))


# per user share of the list from the tail
def tail(ranked, hidden):
    return (tiers(hidden)[ranked] == 2).mean(axis=1)


# per movie number of lists that show it
def exposure(ranked, count):
    return np.bincount(ranked.ravel(), minlength=count)


# inequality of exposure from 0 when every movie is shown equally to 1 when one movie takes all
def gini(exposure):
    total = np.cumsum(np.sort(exposure))
    n = len(exposure)
    return (n + 1 - 2 * total.sum() / total[-1]) / n
