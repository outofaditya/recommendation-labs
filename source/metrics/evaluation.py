"""Task 2.1 - Shared export-based protocol for Task 2.2 and coefficient diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd

from source.metrics.ranking_metrics import RankingMetrics
from source.metrics.util import RecommenderResultLoader

METRICS = ("hit", "precision", "recall", "f1", "map", "mrr", "ndcg")


def load_model(root, model):
    folder = "hybrid" if model.startswith("Tuned") or model == "Weighted" else "tuned"
    with RecommenderResultLoader(Path(root) / folder) as loader:
        return loader.load_predictions(f"{model}.npz")


def load_masks(root, users, items):
    rows, cols = (
        {u: j for j, u in enumerate(users)},
        {i: j for j, i in enumerate(items)},
    )
    masks = {}
    with RecommenderResultLoader(root) as loader:
        for name in ("train", "valid", "test"):
            mask = np.zeros((len(users), len(items)), dtype=bool)
            for user, item in loader.load_split_data_file(f"{name}.tsv"):
                if user not in rows or item not in cols:
                    raise ValueError(
                        f"{name}: split ID absent from export: {user}, {item}"
                    )
                mask[rows[user], cols[item]] = True
            masks[name] = mask
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if np.any(masks[a] & masks[b]):
            raise ValueError(f"Overlapping {a}/{b} interactions")
    return masks


def top_items(scores, hidden, k):
    if k < 1:
        raise ValueError("k must be positive")
    masked = np.where(hidden, -np.inf, scores)
    order = np.argsort(-masked, axis=1, kind="stable")
    # Explicit filtering prevents excluded or seen items filling a short list.
    return [row[np.isfinite(masked[u, row])][:k] for u, row in enumerate(order)]


def ranking_values(true_items, recommendations):
    if not len(recommendations):
        return dict.fromkeys(METRICS, 0.0)
    m = RankingMetrics
    precision = m.calculate_precision(true_items, recommendations)
    recall = m.calculate_recall(true_items, recommendations)
    return {
        "hit": m.calculate_hit(true_items, recommendations),
        "precision": precision,
        "recall": recall,
        "f1": m.calculate_f1(precision, recall),
        "map": m.calculate_ap(true_items, recommendations),
        "mrr": m.calculate_mrr(true_items, recommendations),
        "ndcg": m.calculate_ndcg(true_items, recommendations),
    }


def genre_vectors(path, items):
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    ids = frame.iloc[:, 0]
    column = next((c for c in frame if c.split(":")[0] == "class"), None)
    if column is None:
        raise ValueError(f"No class/genre column: {path}")
    lookup = dict(zip(ids, frame[column].str.split()))
    missing = set(items) - lookup.keys()
    if missing:
        raise ValueError(f"Missing genre metadata for {len(missing)} items")
    genres = sorted({g for item in items for g in lookup[item]})
    vectors = np.array(
        [[g in lookup[item] for g in genres] for item in items], dtype=float
    )
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Items with no genres cannot be assigned a genre distance")
    return vectors / norm


def evaluate(users, items, scores, masks, k=10, genres=None):
    rankings = top_items(scores, masks["train"] | masks["valid"], k)
    counts = masks["train"].sum(axis=0)
    # Laplace smoothing gives unseen training items finite self-information.
    probability = (counts + 1) / (counts.sum() + len(items))
    rows, exposed = [], set()
    for u in np.flatnonzero(masks["test"].any(axis=1)):
        recs = rankings[u]
        true_items = set(np.flatnonzero(masks["test"][u]))
        row = {
            "user_id": users[u],
            "relevant_count": len(true_items),
            "recommended_count": len(recs),
        }
        row.update(
            {
                f"{name}@{k}": value
                for name, value in ranking_values(true_items, recs).items()
            }
        )
        # Keep Task 2.1's actual-list denominators explicit for short candidate lists.
        row[f"novelty@{k}"] = (
            float(-np.log2(probability[recs]).mean()) if len(recs) else None
        )
        row[f"average_popularity@{k}"] = (
            float(counts[recs].mean()) if len(recs) else None
        )
        if genres is not None:
            diversity = None
            if len(recs) >= 2:
                distance = 1 - genres[recs] @ genres[recs].T
                diversity = float(
                    np.clip(distance[np.triu_indices(len(recs), 1)], 0, 1).mean()
                )
            row[f"genre_diversity@{k}"] = diversity
        exposed.update(recs.tolist())
        rows.append(row)
    if not rows:
        raise ValueError("No users with test relevance")
    per_user = pd.DataFrame(rows)
    summary = {
        c: float(per_user[c].mean()) if per_user[c].notna().any() else None
        for c in per_user
        if "@" in c
    }
    summary.update(
        {
            "users": len(rows),
            "short_lists": int((per_user.recommended_count < k).sum()),
            f"catalog_coverage@{k}": len(exposed) / len(items),
        }
    )
    return summary, per_user
