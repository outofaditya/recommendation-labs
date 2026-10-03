"""Task 2.2: compare fixed Task 1 exports with Random and Pop; never train/select."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from source.data import MODELS
from source.metrics.evaluation import evaluate, genre_vectors, load_masks, load_model

HYBRIDS = [
    f"Tuned{name}"
    for name in (
        "Weighted",
        "Mixed",
        "Cascade",
        "Switching",
        "Combination",
        "Augmentation",
        "Metalevel",
    )
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS + HYBRIDS,
        help="Fixed Task 1 models; baselines are always included",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--items", type=Path, default=Path("data/ml-100k/ml-100k.item"))
    parser.add_argument(
        "--skip-genres",
        action="store_true",
        help="Omit genre diversity if metadata is unavailable",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write an explicitly partial evaluation",
    )
    parser.add_argument("--output", type=Path, default=Path("results/task2/evaluation"))
    args = parser.parse_args()
    names = list(dict.fromkeys(["Random", "Pop", *args.models]))
    tables, summaries, missing, reference = {}, [], [], None
    for name in names:
        try:
            users, items, scores = load_model(args.results, name)
        except FileNotFoundError:
            if name in ("Random", "Pop") or not args.allow_missing:
                parser.error(
                    f"Missing {name} export. Supply it or use --allow-missing for non-baselines."
                )
            missing.append(name)
            continue
        if reference is None:
            reference = users, items
            masks = load_masks(args.results, users, items)
            genres = None if args.skip_genres else genre_vectors(args.items, items)
        elif not (
            np.array_equal(users, reference[0]) and np.array_equal(items, reference[1])
        ):
            raise ValueError(f"Export ID order differs: {name}; use compatible exports")
        measured, per_user = evaluate(users, items, scores, masks, args.k, genres)
        summaries.append({"model": name, **measured})
        tables[name] = per_user
    summary = pd.DataFrame(summaries)
    comparisons = []
    for row in summaries:
        for baseline in ("Random", "Pop"):
            base = next(r for r in summaries if r["model"] == baseline)
            for metric in (c for c in summary if "@" in c):
                value, reference_value = row[metric], base[metric]
                delta = (
                    None
                    if value is None or reference_value is None
                    else value - reference_value
                )
                comparisons.append(
                    {
                        "model": row["model"],
                        "baseline": baseline,
                        "metric": metric,
                        "difference": delta,
                        "relative_change": delta / reference_value
                        if delta is not None and reference_value
                        else None,
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        args.output / "baseline_comparison.csv", index=False
    )
    pd.concat(
        [table.assign(model=name) for name, table in tables.items()], ignore_index=True
    ).to_csv(args.output / "per_user.csv", index=False)
    protocol = {
        "status": "partial" if missing else "complete_for_requested_models",
        "missing_models": missing,
        "requested_models": names,
        "k": args.k,
        "selection": "Fixed Task 1 configurations; no test-based selection",
        "relevance": "Any held-out interaction; binary",
        "test_hidden": ["train", "valid"],
        "ties": "Stable exported item-column order",
        "users": "Users with test relevance, same for all models",
        "precision_and_ap": "Task 2.1 formulas: actual list length for short lists; empty lists score zero",
        "novelty": "Mean -log2((train item count + 1) / (total train interactions + catalog size))",
        "average_popularity": "Mean training interaction count; lower means less popular recommendations",
        "catalog_coverage": "Distinct recommended items / exported catalog size for evaluated users",
        "genre_diversity": None
        if genres is None
        else "Mean pairwise cosine distance of binary genre vectors; undefined for fewer than 2 items",
        "undefined_metrics": "Null in JSON, blank in CSV; omitted from the corresponding mean",
        "beyond_accuracy_status": "Draft definitions to agree with Task 2.1 author; calibration and fairness are not implemented",
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(summary.to_string(index=False))
    print(f"Saved {protocol['status']} evaluation to {args.output}; missing: {missing}")


if __name__ == "__main__":
    main()
