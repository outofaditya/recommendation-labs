import json

import numpy as np
from source.reranking.diversity import mmr
from source.reranking.calibration import steck
from source.reranking.fairness import cp, xquad
from source.metrics.accuracy import accuracy, top
from source.data import RESULTS, genres, scores, split

BASES = ["EASE", "LightGCN", "UserKNN"]
RERANKERS = {"MMR": mmr, "Calibrated": steck, "xQuAD": xquad, "CP": cp}
LAMBDAS = np.round(np.arange(0, 1.001, 0.05), 2)
TOLERANCE = 0.05
FOLDER = RESULTS / "rerank"


# the strongest re-ranking that keeps validation ndcg within the tolerance of the original lists
def choose(curve):
    floor = (1 - TOLERANCE) * curve[0]
    return max(lam for lam, ndcg in zip(LAMBDAS, curve) if ndcg >= floor)


# every lambda is ranked on validation and only the chosen one is scored on test
def run(base, rerank):
    values, users, items = scores(base)
    masks, flags = split(users, items), genres(items)
    seen = masks["train"] | masks["valid"]
    valid = np.stack([rerank(values, masks["train"], flags, lam) for lam in LAMBDAS])
    curve = [accuracy(ranked, masks["valid"])["ndcg@10"] for ranked in valid]
    lam = choose(curve)
    test = rerank(values, seen, flags, lam)
    original = accuracy(top(values, seen), masks["test"])
    metrics = {"lambda": float(lam), "valid": dict(zip(map(str, LAMBDAS), curve))}
    metrics |= {"original": original, "test": accuracy(test, masks["test"])}
    return valid, test, users, items, metrics


def save(label, valid, test, users, items, metrics):
    for kind in ("ranked", "metrics"):
        (FOLDER / kind).mkdir(parents=True, exist_ok=True)
    path = FOLDER / "ranked" / f"{label}.npz"
    np.savez_compressed(path, valid=valid, test=test, lambdas=LAMBDAS, users=users, items=items)
    (FOLDER / "metrics" / f"{label}.json").write_text(json.dumps(metrics, indent=2))
    before, after = metrics["original"]["ndcg@10"], metrics["test"]["ndcg@10"]
    print(f"{label:22} lambda {metrics['lambda']:.2f}  test ndcg {before:.4f} -> {after:.4f}")


if __name__ == "__main__":
    for base in BASES:
        for name, rerank in RERANKERS.items():
            save(f"{base}-{name}", *run(base, rerank))
