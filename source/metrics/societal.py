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


# shannon entropy of each distribution so a wider mix scores higher
def entropy(p):
    return -np.sum(p * np.log(np.where(p > 0, p, 1)), axis=-1)


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


# equal thirds by value with ties kept in their order
def thirds(values):
    third = np.empty(len(values), dtype=int)
    third[np.argsort(values, kind="stable")] = np.arange(len(values)) * 3 // len(values)
    return third


# share of each user's history from head movies
def head(hidden):
    return profile(hidden, np.eye(3)[tiers(hidden)])[:, 0]


# niche diverse and blockbuster users as equal thirds by the head share of their history
def tastes(hidden):
    return thirds(head(hidden))


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


# user popularity deviation of abdollahpouri et al as the js of the list's head mid tail mix from the history's
def upd(ranked, hidden):
    shares = np.eye(3)[tiers(hidden)]
    return js(profile(hidden, shares), shares[ranked].mean(axis=1))


# per user share of the list from the tail
def tail(ranked, hidden):
    return (tiers(hidden)[ranked] == 2).mean(axis=1)


# per movie exposure over all lists where each slot counts as its position discount
def exposure(ranked, count, discount=1):
    return np.bincount(ranked.ravel(), np.broadcast_to(discount, ranked.shape).ravel(), count)


# inequality of exposure from 0 when every movie is shown equally to 1 when one movie takes all
def gini(exposure):
    total = np.cumsum(np.sort(exposure))
    n = len(exposure)
    return (n + 1 - 2 * total.sum() / total[-1]) / n


# group recommendation unfairness of fu et al as the gap in mean quality between two user groups
def gru(quality, first, second):
    return abs(quality[first].mean() - quality[second].mean())


# demographic parity of singh and joachims as the protected group's mean exposure over the rest's so 1 is fair
def parity(exposure, protected):
    return exposure[protected].mean() / exposure[~protected].mean()


# disparate treatment of singh and joachims as parity of exposure per unit of relevance so 1 is fair
def treatment(exposure, utility, protected):
    return parity(exposure, protected) / parity(utility, protected)
