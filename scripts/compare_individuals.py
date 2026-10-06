"""Task 2.2 experiment 1: evaluate fixed Task 1 exports without training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from source.metrics.beyond_accuracy import BeyondAccuracyMetrics, jaccard_distance
from source.metrics.ranking_metrics import RankingMetrics

MODELS = ["Random", "Pop", "ItemKNN", "UserKNN", "BPR", "NeuMF", "FISM", "LightGCN", "NGCF", "EASE", "SLIMElastic"]
ACCURACY_METRICS = ["ndcg", "recall", "precision", "mrr", "hit", "map", "f1"]
BEYOND_ACCURACY_METRICS = ["coverage", "novelty", "ild", "serendipity"]
METRICS = ACCURACY_METRICS + BEYOND_ACCURACY_METRICS


def load_predictions(path, *, allow_excluded=False):
    with np.load(path, allow_pickle=False) as data:
        users, items, scores = data["users"].astype(str), data["items"].astype(str), data["scores"]
    if users.ndim != 1 or items.ndim != 1 or scores.shape != (len(users), len(items)):
        raise ValueError(f"Invalid prediction dimensions: {path}")
    if len(set(users)) != len(users) or len(set(items)) != len(items):
        raise ValueError(f"Duplicate user/item IDs: {path}")
    if np.isnan(scores).any() or np.isposinf(scores).any() or (not allow_excluded and np.isneginf(scores).any()):
        raise ValueError(f"Individual predictions must be finite: {path}")
    return users, items, scores


def load_masks(folder, users, items):
    user_index, item_index = pd.Index(users), pd.Index(items)
    masks = {}
    for name in ("train", "valid", "test"):
        frame = pd.read_csv(folder / f"{name}.tsv", sep="\t", dtype=str)
        if frame.shape[1] < 2 or frame.empty:
            raise ValueError(f"Empty or malformed {name} split")
        u = user_index.get_indexer(frame.iloc[:, 0])
        i = item_index.get_indexer(frame.iloc[:, 1])
        if (u < 0).any() or (i < 0).any():
            raise ValueError(f"Unmapped user/item IDs in {name}")
        mask = np.zeros((len(users), len(items)), dtype=bool)
        mask[u, i] = True
        if any(np.any(mask & previous) for previous in masks.values()):
            raise ValueError(f"Overlapping interactions in {name} split")
        masks[name] = mask
    return masks


def load_item_features(path, items):
    frame = pd.read_csv(path, sep="\t", dtype=str)
    if frame.shape[1] < 4 or frame.empty:
        raise ValueError(f"Empty or malformed item metadata: {path}")
    features_by_id = {
        item_id: genres.split() if isinstance(genres, str) else []
        for item_id, genres in zip(frame.iloc[:, 0], frame.iloc[:, 3], strict=True)
    }
    missing = [item for item in items if item not in features_by_id]
    if missing:
        raise ValueError(f"Item metadata is missing {len(missing)} exported item IDs")
    return [features_by_id[item] for item in items]


def run(results, output, k=10, items_path=Path("data/ml-100k/ml-100k.item")):
    if k < 1:
        raise ValueError("k must be positive")
    rows, diagnostics, hashes = [], {}, {}
    reference = None
    warnings = []
    for name in MODELS:
        path = results / "tuned/scores" / f"{name}.npz"
        users, items, scores = load_predictions(path)
        if reference is None:
            reference = users, items
            masks = load_masks(results / "split", users, items)
            hidden = masks["train"] | masks["valid"]
            eligible = masks["test"].any(axis=1)
            if not eligible.any() or np.any((~hidden[eligible]).sum(axis=1) < k):
                raise ValueError("No test users or fewer than k eligible items for a test user")
            item_features = load_item_features(items_path, items)
            distance_matrix = BeyondAccuracyMetrics.build_genre_distance_matrix(
                item_features, len(items), jaccard_distance
            )
        elif not (np.array_equal(users, reference[0]) and np.array_equal(items, reference[1])):
            raise ValueError(f"User/item ID order differs for {name}")
        values = RankingMetrics.calculate_all_metrics(scores, hidden, masks["test"], k)
        values.update(
            BeyondAccuracyMetrics.calculate_all_metrics(
                scores, hidden, masks["test"], distance_matrix, k
            )
        )
        rows.append({"model": name, **{f"{m}@{k}": values[f"{m}@{k}"] for m in METRICS}})
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in ("Random", "Pop"):
            distinct = len(np.unique(scores, axis=0))
            diagnostics[name] = {"distinct_score_rows": distinct, "total_users": len(users)}
            if name == "Random" and distinct < len(users):
                warnings.append(
                    f"Random has {distinct} distinct score rows for {len(users)} users. "
                    "Shared rankings limit interpretation as independent per-user random recommendations; "
                    "the original Task 1 export is preserved."
                )
    for name in masks:
        path = results / "split" / f"{name}.tsv"
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(items_path)] = hashlib.sha256(items_path.read_bytes()).hexdigest()
    metric_path = Path(__file__).resolve().parents[1] / "source/metrics/ranking_metrics.py"
    hashes["source/metrics/ranking_metrics.py"] = hashlib.sha256(metric_path.read_bytes()).hexdigest()
    beyond_path = Path(__file__).resolve().parents[1] / "source/metrics/beyond_accuracy.py"
    hashes["source/metrics/beyond_accuracy.py"] = hashlib.sha256(beyond_path.read_bytes()).hexdigest()
    hashes["scripts/compare_individuals.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = pd.DataFrame(rows)
    protocol = {
        "experiment": "Task 2.2 experiment 1: tuned individual models",
        "status": "complete_with_baseline_warning" if warnings else "complete",
        "models": MODELS, "k": k, "users_evaluated": int(eligible.sum()), "catalog_size": len(items),
        "split_interactions": {name: int(mask.sum()) for name, mask in masks.items()},
        "selection": "All nine fixed Task 1 tuned individuals plus Random and Pop; no test-based selection or retraining",
        "relevance": "Binary test interaction, irrespective of rating value",
        "hidden": ["train", "valid"], "ties": "Stable exported item-column order",
        "aggregation": "Macro mean over users with at least one test interaction",
        "metrics": {
            "accuracy": "Task 2.1 RankingMetrics.calculate_all_metrics; AP denominator min(test positives, K); per-user F1",
            "beyond_accuracy": "Task 2.1 BeyondAccuracyMetrics: catalog coverage; self-information novelty from hidden interaction frequency; genre-Jaccard ILD; relevance-weighted genre-distance serendipity",
        },
        "uncertainty": "One saved split and one export per model; no significance or multi-seed claims",
        "baseline_diagnostics": diagnostics, "warnings": warnings, "sha256": hashes,
    }
    columns = list(summary.columns)
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        table.append("| " + row["model"] + " | " + " | ".join(f"{row[c]:.4f}" for c in columns[1:]) + " |")
    report = (
        f"# Task 2.2 — Experiment 1\n\n"
        f"Evaluated {len(rows)} fixed individual-model exports on {int(eligible.sum())} users at K={k}. "
        "Train and validation interactions are excluded. Accuracy and beyond-accuracy metrics use the Task 2.1 implementations. "
        "No models were trained or selected using test results.\n\n"
        + "\n".join(table)
        + "\n\nResults describe one saved split; numerical differences do not establish statistical significance.\n"
        + "\n".join(f"\nWarning: {warning}\n" for warning in warnings)
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (output / "comparison.md").write_text(report)
    return summary, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task2/experiment1"))
    parser.add_argument("--items", type=Path, default=Path("data/ml-100k/ml-100k.item"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    summary, protocol = run(args.results, args.output, args.k, args.items)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for warning in protocol["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
