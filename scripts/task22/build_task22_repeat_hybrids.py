"""Rebuild the fixed Task 1 tuned hybrids from a single repeated-seed run."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.hybrids import augmentation, combination, metalevel, ranks, shortlisted, switching
from scripts.weighted import fit
from source.hybrids.pool import MEMBERS, candidates, features, save


def build(reference_metrics: Path):
    x, masks, users, items = features()
    seen = masks["train"] | masks["valid"]

    def reference(name):
        path = reference_metrics / f"Tuned{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing tuned-hybrid settings: {path}")
        return json.loads(path.read_text())

    settings = reference("Weighted")
    depth, strength = int(settings["depth"]), float(settings["C"])
    pool = candidates(x, masks["train"], depth)
    weights = fit(x[pool], masks["valid"][pool], strength)
    table = np.where(candidates(x, seen, depth), x @ weights, -np.inf)
    save("TunedWeighted", table, users, items, {"depth": depth, "C": strength, "weights": dict(zip(MEMBERS, weights.tolist(), strict=True))})

    settings = reference("Mixed")
    fusion = int(settings["k"])
    members = [name for name in settings["members"] if name in MEMBERS]
    order = [MEMBERS.index(name) for name in members]
    table = (1 / (fusion + ranks(x, seen)[..., order])).sum(axis=-1)
    save("TunedMixed", table, users, items, {"k": fusion, "members": members})

    settings = reference("Cascade")
    first, second = settings["stages"]
    table = shortlisted(x[..., MEMBERS.index(first)], x[..., MEMBERS.index(second)], seen, int(settings["shortlist"]))
    save("TunedCascade", table, users, items, {"stages": [first, second], "shortlist": int(settings["shortlist"])})

    settings = reference("Switching")
    count = len(settings["groups"])
    table, details = switching(x, masks, count)
    save("TunedSwitching", table, users, items, details | {"group_count": count})

    # Task 1's selected configuration is depth=2, learning_rate=0.1.
    table, details = combination(x, masks, depth=2, rate=0.1)
    save("TunedCombination", table, users, items, details | {"depth": 2, "rate": 0.1})

    settings = reference("Augmentation")
    table, details = augmentation(x, masks, int(settings["pseudo"]), int(settings["k"]))
    save("TunedAugmentation", table, users, items, details)

    settings = reference("Metalevel")
    table, details = metalevel(x, masks, settings["embeddings"], int(settings["k"]))
    save("TunedMetalevel", table, users, items, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    args = parser.parse_args()
    build(args.reference_metrics)


if __name__ == "__main__":
    main()
