"""Task 2.2 experiment 2: compare saved tuned hybrids and individuals; no training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_individuals import METRICS, MODELS, load_item_features, load_masks, load_predictions
from source.metrics.beyond_accuracy import BeyondAccuracyMetrics, jaccard_distance
from source.metrics.ranking_metrics import RankingMetrics

HYBRIDS = [f"Tuned{name}" for name in ("Weighted", "Mixed", "Cascade", "Switching", "Combination", "Augmentation", "Metalevel")]


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def run(results, output, k=10, items_path=Path("data/ml-100k/ml-100k.item")):
    if k < 1:
        raise ValueError("k must be positive")
    columns = [f"{metric}@{k}" for metric in METRICS]
    rows, diagnostics, hashes, warnings = [], {}, {}, []
    reference = None
    for name in MODELS + HYBRIDS:
        hybrid = name in HYBRIDS
        path = results / ("hybrid/scores" if hybrid else "tuned/scores") / f"{name}.npz"
        users, items, scores = load_predictions(path, allow_excluded=hybrid)
        if reference is None:
            reference = users, items
            masks = load_masks(results / "split", users, items)
            hidden = masks["train"] | masks["valid"]
            eligible = masks["test"].any(axis=1)
            if not eligible.any():
                raise ValueError("No users with test interactions")
            item_features = load_item_features(items_path, items)
            distance_matrix = BeyondAccuracyMetrics.build_genre_distance_matrix(item_features, len(items), jaccard_distance)
        elif not (np.array_equal(users, reference[0]) and np.array_equal(items, reference[1])):
            raise ValueError(f"User/item ID order differs for {name}")
        available = np.isfinite(scores) & ~hidden
        counts = available[eligible].sum(axis=1)
        if np.any(counts < k):
            raise ValueError(f"{name}: fewer than k finite eligible candidates for a test user (minimum {counts.min()}, k={k})")
        values = RankingMetrics.calculate_all_metrics(scores, hidden, masks["test"], k)
        values.update(BeyondAccuracyMetrics.calculate_all_metrics(scores, hidden, masks["test"], distance_matrix, k))
        group = "hybrid" if hybrid else ("baseline" if name in ("Random", "Pop") else "individual")
        rows.append({"model": name, "category": group, **{c: values[c] for c in columns}})
        diagnostics[name] = {"minimum_candidates": int(counts.min()), "maximum_candidates": int(counts.max())}
        if name == "Random":
            distinct = len(np.unique(scores, axis=0))
            diagnostics[name]["distinct_score_rows"] = distinct
            if distinct < len(users):
                warnings.append(
                    f"Saved Random has {distinct} distinct score rows for {len(users)} users; it is not independent per-user random scoring."
                )
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in masks:
        path = results / "split" / f"{name}.tsv"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(items_path)] = hashlib.sha256(items_path.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/compare_hybrids.py",
        "scripts/compare_individuals.py",
        "source/metrics/ranking_metrics.py",
        "source/metrics/beyond_accuracy.py",
    ):
        hashes[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    summary = pd.DataFrame(rows)
    individuals = summary[summary.category == "individual"]
    # Descriptive reference only: all fixed configurations are evaluated, none selected/refit on test.
    best = individuals.sort_values(columns[0], ascending=False, kind="stable").iloc[0]
    differences = []
    for _, hybrid_row in summary[summary.category == "hybrid"].iterrows():
        for _, individual in individuals.iterrows():
            for metric in columns:
                delta = float(hybrid_row[metric] - individual[metric])
                differences.append(
                    {
                        "hybrid": hybrid_row.model,
                        "individual": individual.model,
                        "metric": metric,
                        "difference": delta,
                        "relative_change": delta / individual[metric] if individual[metric] else None,
                    }
                )
    comparisons = pd.DataFrame(differences)
    reference_comparison = summary[summary.category == "hybrid"][["model", *columns]].copy()
    reference_comparison[f"ndcg_delta_vs_{best.model}"] = reference_comparison[columns[0]] - best[columns[0]]
    protocol = {
        "experiment": "Task 2.2 experiment 2: tuned hybrids versus individuals",
        "status": "complete_with_baseline_warning" if warnings else "complete",
        "models": MODELS + HYBRIDS,
        "k": k,
        "users_evaluated": int(eligible.sum()),
        "catalog_size": len(items),
        "split_interactions": {name: int(mask.sum()) for name, mask in masks.items()},
        "relevance": "Binary test interaction irrespective of rating; hide train and validation",
        "aggregation": "Macro mean over the same users with test interactions",
        "metrics": {
            "accuracy": "Task 2.1 RankingMetrics.calculate_all_metrics; AP denominator min(test positives, K); per-user F1",
            "beyond_accuracy": "Task 2.1 BeyondAccuracyMetrics: catalog coverage; self-information novelty from hidden interaction frequency; genre-Jaccard ILD; relevance-weighted genre-distance serendipity",
        },
        "ties": "Stable exported item-column order",
        "candidates": "Preserve each hybrid's exported candidate pool; negative infinity excludes an item; require at least K finite unseen items",
        "selection": "All locally available fixed tuned configurations; no training, tuning, or test-based configuration selection",
        "descriptive_reference": {
            "model": best.model,
            "basis": f"Highest observed individual {columns[0]} on this test split; not a validation-selected winner",
        },
        "limitations": [
            "One local saved split and export per model; no statistical significance or multi-seed claims",
            "Current local artifacts may differ from the original Task 1 report; do not mix their numbers",
            "IDs and split disjointness are checked; exports alone cannot prove their original training provenance",
        ],
        "diagnostics": diagnostics,
        "warnings": warnings,
        "sha256": hashes,
    }
    winners = summary[summary.category == "hybrid"].sort_values(columns[0], ascending=False, kind="stable")
    leader = winners.iloc[0]
    beating = winners[winners[columns[0]] > best[columns[0]]].model.tolist()
    report = (
        "# Task 2.2 — Experiment 2: tuned hybrids versus individuals\n\n"
        f"Evaluated all 18 fixed local exports on {int(eligible.sum())} test users at K={k}, using the Task 2.1 accuracy and beyond-accuracy metrics. "
        "Train and validation interactions are hidden; hybrid candidate restrictions are preserved. No training or tuning was performed.\n\n"
        + markdown_table(summary)
        + "\n\n## Hybrid comparison\n\n"
        + f"{best.model} is the strongest observed individual by NDCG on this test split, used only as a descriptive reference.\n\n"
        + markdown_table(reference_comparison)
        + "\n\n## Discussion\n\n"
        + f"{leader.model} has the highest hybrid NDCG@{k} ({leader[columns[0]]:.4f}), compared with {best[columns[0]]:.4f} for {best.model}. "
        + f"Hybrids exceeding that individual on NDCG: {', '.join(beating) or 'none'}. "
        + "Hybridisation therefore does not improve every design equally. Weighted blends member scores, Mixed fuses ranks, and Combination learns from scores and side features; "
        "their aggregate results should be distinguished from Cascade's restricted shortlist and Switching's choice of one member per user group. "
        "These mechanisms motivate explanations but this comparison alone does not establish their causal effects. "
        "Accuracy and beyond-accuracy metrics are reported because ranking quality, catalogue reach, novelty, list diversity and serendipity need not move together.\n\n"
        + "## Limits\n\n"
        + "\n".join(f"- {v}" for v in protocol["limitations"] + warnings)
        + "\n\nDifferences are descriptive, without significance tests. The all-pairs CSV compares every hybrid against every individual, avoiding reliance on a single reference.\n\n"
        + "Reproduce from the repository root: `uv run python -m scripts.compare_hybrids`\n"
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "hybrid_vs_individual.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (output / "comparison.md").write_text(report)
    return summary, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task2/experiment2"))
    parser.add_argument("--items", type=Path, default=Path("data/ml-100k/ml-100k.item"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    summary, protocol = run(args.results, args.output, args.k, args.items)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for warning in protocol["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
