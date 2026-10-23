"""Run Task 2.4 analyses on the saved Task 1 exports.

The pipeline performs three complementary analyses:

1. Per-user model wins and the descriptive value of contextual routing.
2. Candidate recall versus reranking quality at several candidate depths.
3. Popularity and long-tail exposure.
4. Specialized long-tail candidate retrieval.
5. Retrieval and ranking outcomes by users' training-time tail interest.

No model is trained in this module. Any deployable policy must be selected on
validation users in a separate experiment; the oracle routing result below is
an upper bound and is never treated as a deployable policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from source.data import split
from source.metrics.accuracy import ndcg_per_user, top

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INDIVIDUALS = ["EASE", "SLIMElastic", "LightGCN", "FISM", "BPR", "NeuMF"]
HYBRIDS = ["Weighted", "Combination"]
ANALYSIS_MODELS = INDIVIDUALS + HYBRIDS
RERANKERS = INDIVIDUALS
MODEL_PATHS = {
    **{name: f"tuned/scores/{name}.npz" for name in INDIVIDUALS},
    "Weighted": "hybrid/scores/TunedWeighted.npz",
    "Combination": "hybrid/scores/TunedCombination.npz",
}
K = 10
DEPTHS = (20, 50, 100, 200, 0)
BOOTSTRAP_REPS = 2000


def load_exports(results: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load selected exports and verify their common user/item ordering."""
    tables = {}
    users = items = None
    for name in ANALYSIS_MODELS:
        table = np.load(results / MODEL_PATHS[name])
        if users is None:
            users, items = table["users"], table["items"]
        elif not (np.array_equal(users, table["users"]) and np.array_equal(items, table["items"])):
            raise ValueError(f"User or item order differs for {name}")
        tables[name] = table["scores"].astype(np.float64)
    return users, items, tables


def load_masks(results: Path, users: np.ndarray, items: np.ndarray) -> dict[str, np.ndarray]:
    """Build dense split masks aligned with the saved score exports."""
    return split(users, items, results)


def unseen_top(scores: np.ndarray, hidden: np.ndarray, k: int) -> np.ndarray:
    """Return top-k unseen item indices using stable exported-column tie order."""
    return top(scores, hidden, k)


def per_user_ndcg(scores: np.ndarray, hidden: np.ndarray, relevant: np.ndarray, k: int = K) -> np.ndarray:
    """Return NDCG in original user order, including zero for ineligible users."""
    return ndcg_per_user(top(scores, hidden, k), relevant)


def bootstrap_mean_difference(values: np.ndarray, reps: int = BOOTSTRAP_REPS, seed: int = 2020) -> tuple[float, float, float]:
    """Return the mean and percentile bootstrap interval for paired user values."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    samples = values[indices].mean(axis=1)
    return float(values.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def user_features(masks: dict[str, np.ndarray], items_path: Path, items: np.ndarray) -> pd.DataFrame:
    """Create user features that are available before observing test labels."""
    item_frame = pd.read_csv(items_path, sep="\t", dtype=str)
    item_frame.columns = [column.split(":", 1)[0] for column in item_frame.columns]
    genres = item_frame.set_index("item_id").reindex([str(value) for value in items])["class"].fillna("")
    genre_sets = [set(value.split()) for value in genres]
    train = masks["train"]
    counts = train.sum(axis=1)
    entropy = np.zeros(len(train), dtype=float)
    mean_genres = np.zeros(len(train), dtype=float)
    popularity = train.sum(axis=0)
    history_popularity = np.zeros(len(train), dtype=float)
    for user in range(len(train)):
        seen = np.flatnonzero(train[user])
        if len(seen):
            frequencies = {}
            for item in seen:
                for genre in genre_sets[item]:
                    frequencies[genre] = frequencies.get(genre, 0) + 1
            probabilities = np.asarray(list(frequencies.values()), dtype=float)
            probabilities = probabilities / probabilities.sum()
            entropy[user] = float(-(probabilities * np.log2(probabilities)).sum())
            mean_genres[user] = float(np.mean([len(genre_sets[item]) for item in seen]))
            history_popularity[user] = float(np.mean(popularity[seen]))
    return pd.DataFrame(
        {
            "user_index": np.arange(len(train)),
            "train_interactions": counts,
            "genre_entropy": entropy,
            "mean_genres_per_item": mean_genres,
            "mean_history_popularity": history_popularity,
        }
    )


def add_quantile_group(frame: pd.DataFrame, source: str, target: str, groups: int = 4) -> None:
    """Add stable rank-based quantile labels, including ties."""
    frame[target] = pd.qcut(frame[source].rank(method="first"), groups, labels=False) + 1


def experiment_routing(tables, masks, features, output: Path) -> dict:
    """Measure per-user wins and the descriptive oracle-routing upper bound."""
    hidden = masks["train"] | masks["valid"]
    relevant = masks["test"]
    eligible = relevant.any(axis=1)
    features = features.loc[eligible].reset_index(drop=True).copy()
    user_values = {name: per_user_ndcg(tables[name], hidden, relevant)[eligible] for name in ANALYSIS_MODELS}
    matrix = np.column_stack([user_values[name] for name in ANALYSIS_MODELS])
    best_index = matrix.argmax(axis=1)
    best_values = matrix[np.arange(len(matrix)), best_index]
    reference = matrix[:, ANALYSIS_MODELS.index("Combination")]
    features["best_model"] = np.asarray(ANALYSIS_MODELS)[best_index]
    features["oracle_ndcg@10"] = best_values
    features["combination_ndcg@10"] = reference
    for name in ANALYSIS_MODELS:
        features[f"{name}_ndcg@10"] = user_values[name]
    features.to_csv(output / "routing_per_user.csv", index=False)

    rows = []
    for feature_group, label in (("activity_group", "activity"), ("entropy_group", "genre_entropy")):
        for group, group_frame in features.groupby(feature_group, sort=True):
            indices = group_frame.index.to_numpy()
            group_matrix = matrix[indices]
            group_best = group_matrix.argmax(axis=1)
            for model_index, name in enumerate(ANALYSIS_MODELS):
                rows.append(
                    {
                        "segment": label,
                        "segment_value": int(group),
                        "model": name,
                        "users": len(indices),
                        "win_rate": float(np.mean(group_best == model_index)),
                        "mean_ndcg@10": float(group_matrix[:, model_index].mean()),
                    }
                )
    segment = pd.DataFrame(rows)
    segment.to_csv(output / "routing_segments.csv", index=False)

    gap = best_values - reference
    mean_gap, low_gap, high_gap = bootstrap_mean_difference(gap)
    summary = {
        "eligible_users": len(features),
        "oracle_ndcg@10": float(best_values.mean()),
        "combination_ndcg@10": float(reference.mean()),
        "oracle_minus_combination": mean_gap,
        "oracle_minus_combination_ci95_low": low_gap,
        "oracle_minus_combination_ci95_high": high_gap,
        "model_win_rates": {name: float(np.mean(np.asarray(ANALYSIS_MODELS)[best_index] == name)) for name in ANALYSIS_MODELS},
    }
    (output / "routing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    pivot = segment[segment.segment == "activity"].pivot(index="segment_value", columns="model", values="win_rate")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Training activity quartile")
    ax.set_ylabel("Per-user best-model share")
    ax.set_title("Per-user model wins by training activity")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output / "routing_wins_by_activity.png", dpi=180)
    plt.close()
    return summary


def candidate_pool(scores: np.ndarray, hidden: np.ndarray, depth: int) -> np.ndarray:
    """Build a boolean candidate pool; depth zero means the full unseen catalogue."""
    unseen = ~hidden & np.isfinite(scores)
    if depth == 0:
        return unseen.copy()
    ranked = unseen_top(scores, hidden, min(depth, scores.shape[1]))
    pool = np.zeros_like(unseen)
    np.put_along_axis(pool, ranked, True, axis=1)
    return pool & unseen


def rerank(scores: np.ndarray, pool: np.ndarray, hidden: np.ndarray, k: int = K, eligible: np.ndarray | None = None) -> np.ndarray:
    """Rank the top-k items from an externally supplied candidate pool."""
    allowed = pool & ~hidden & np.isfinite(scores)
    # Refuse short lists rather than score excluded items as recommendations.
    counts = allowed.sum(axis=1)
    if np.any((counts if eligible is None else counts[eligible]) < k):
        raise ValueError(f"Fewer than {k} finite unseen reranking candidates")
    return top(scores, ~allowed, k)


def experiment_candidates(tables, masks, output: Path) -> dict:
    """Decompose candidate exclusion and ordering losses across depths."""
    hidden = masks["train"] | masks["valid"]
    relevant = masks["test"]
    eligible = relevant.any(axis=1)
    rows = []
    for generator in ANALYSIS_MODELS:
        for depth in DEPTHS:
            pool = candidate_pool(tables[generator], hidden, depth)
            relevant_total = relevant[eligible].sum(axis=1)
            relevant_in_pool = (relevant & pool)[eligible].sum(axis=1)
            candidate_recall = relevant_in_pool / relevant_total
            for ranker in RERANKERS:
                ranked = rerank(tables[ranker], pool, hidden, K, eligible=eligible)
                available = pool & np.isfinite(tables[ranker]) & ~hidden
                ndcg = ndcg_per_user(ranked, relevant)[eligible]
                conditional = ndcg[candidate_recall > 0]
                rows.append(
                    {
                        "candidate_generator": generator,
                        "reranker": ranker,
                        "depth": "full" if depth == 0 else depth,
                        "eligible_users": len(ndcg),
                        "candidate_recall": float(candidate_recall.mean()),
                        "candidate_catalogue_coverage": float(pool[eligible].any(axis=0).mean()),
                        "ranker_available_fraction": float(available[eligible].sum() / pool[eligible].sum() if pool[eligible].sum() else 0.0),
                        "ndcg@10": float(ndcg.mean()),
                        "conditional_ndcg@10": float(conditional.mean()) if len(conditional) else 0.0,
                        "users_with_relevant_candidate": int((candidate_recall > 0).sum()),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(output / "candidate_pipeline.csv", index=False)
    best = table.sort_values(["ndcg@10", "candidate_recall"], ascending=False, kind="stable").head(20)
    best.to_csv(output / "candidate_pipeline_top20.csv", index=False)
    return {"rows": len(table), "best_rows": best.to_dict(orient="records")}


def load_item_metadata(items_path: Path, items: np.ndarray) -> tuple[np.ndarray, list[set[str]]]:
    """Load item popularity-independent metadata aligned to exported items."""
    frame = pd.read_csv(items_path, sep="\t", dtype=str)
    frame.columns = [column.split(":", 1)[0] for column in frame.columns]
    aligned = frame.set_index("item_id").reindex([str(value) for value in items]).fillna("")
    genres = [set(str(value).split()) for value in aligned["class"]]
    return aligned.index.to_numpy(), genres


def bucket_labels(popularity: np.ndarray) -> np.ndarray:
    """Assign fixed relative popularity buckets before test outcomes are read."""
    order = np.argsort(-popularity, kind="stable")
    labels = np.empty(len(popularity), dtype=object)
    n = len(popularity)
    labels[order[: max(1, int(np.ceil(0.2 * n)))]] = "head_top20pct"
    labels[order[max(1, int(np.ceil(0.2 * n))) : max(2, int(np.ceil(0.5 * n)))]] = "middle_20_50pct"
    labels[order[max(2, int(np.ceil(0.5 * n))) :]] = "tail_bottom50pct"
    return labels


def experiment_popularity(tables, masks, items: np.ndarray, output: Path) -> dict:
    """Measure relevance and exposure separately across popularity buckets."""
    train_popularity = masks["train"].sum(axis=0)
    labels = bucket_labels(train_popularity)
    hidden = masks["train"] | masks["valid"]
    relevant = masks["test"]
    eligible = relevant.any(axis=1)
    rows = []
    top_items = []
    for name in ANALYSIS_MODELS:
        ranked = unseen_top(tables[name], hidden, K)
        top_items.append(ranked[eligible])
        for bucket in ("head_top20pct", "middle_20_50pct", "tail_bottom50pct"):
            item_mask = labels == bucket
            bucket_relevant = relevant & item_mask[None, :]
            bucket_total = bucket_relevant[eligible].sum(axis=1)
            hits = np.take_along_axis(bucket_relevant, ranked, axis=1)[eligible]
            bucket_ndcg = ndcg_per_user(ranked, bucket_relevant, counts=relevant.sum(axis=1))[eligible]
            bucket_recall = np.divide(hits.sum(axis=1), bucket_total, out=np.zeros(len(hits)), where=bucket_total > 0)
            exposure = np.mean(labels[ranked[eligible]] == bucket)
            rows.append(
                {
                    "model": name,
                    "bucket": bucket,
                    "items_in_bucket": int(item_mask.sum()),
                    "train_interaction_share": float(train_popularity[item_mask].sum() / train_popularity.sum()),
                    "recommendation_exposure_share": float(exposure),
                    "relevant_item_count": int(bucket_total.sum()),
                    "users_with_relevant_item": int((bucket_total > 0).sum()),
                    "ndcg_contribution@10": float(bucket_ndcg.mean()),
                    "recall_within_bucket@10": float(bucket_recall[bucket_total > 0].mean()) if np.any(bucket_total > 0) else 0.0,
                    "mean_relevant_items_per_user": float(bucket_total.mean()),
                }
            )
        overall_hits = np.take_along_axis(relevant[eligible], ranked[eligible], axis=1).sum(axis=1)
        overall_recall = np.divide(
            overall_hits,
            relevant[eligible].sum(axis=1),
            out=np.zeros(len(overall_hits)),
            where=relevant[eligible].sum(axis=1) > 0,
        )
        rows.append(
            {
                "model": name,
                "bucket": "all",
                "items_in_bucket": len(items),
                "train_interaction_share": 1.0,
                "recommendation_exposure_share": 1.0,
                "relevant_item_count": int(relevant[eligible].sum()),
                "users_with_relevant_item": len(relevant[eligible]),
                "ndcg_contribution@10": float(per_user_ndcg(tables[name], hidden, relevant)[eligible].mean()),
                "recall_within_bucket@10": float(overall_recall.mean()),
                "mean_relevant_items_per_user": float(relevant[eligible].sum(axis=1).mean()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "popularity_buckets.csv", index=False)
    exposure = pd.DataFrame(
        [
            {"model": name, "bucket": bucket, "exposure_share": float(np.mean(labels[ranked] == bucket))}
            for name, ranked in zip(ANALYSIS_MODELS, top_items)
            for bucket in ("head_top20pct", "middle_20_50pct", "tail_bottom50pct")
        ]
    )
    exposure.to_csv(output / "popularity_exposure.csv", index=False)
    plot = exposure.pivot(index="model", columns="bucket", values="exposure_share")
    plot.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#b2182b", "#ef8a62", "#67a9cf"])
    plt.ylabel("Recommendation exposure share")
    plt.title("Recommendation exposure by training popularity")
    plt.legend(title="Popularity bucket", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output / "popularity_exposure.png", dpi=180)
    plt.close()
    return {"bucket_counts": {bucket: int(np.sum(labels == bucket)) for bucket in np.unique(labels)}, "models": ANALYSIS_MODELS}


def top_finite_indices(
    scores: np.ndarray,
    hidden: np.ndarray,
    count: int,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Return up to count highest-scoring finite unseen items for each user."""
    result = np.full((len(scores), count), -1, dtype=int)
    for user in range(len(scores)):
        allowed = ~hidden[user] & np.isfinite(scores[user])
        if excluded is not None:
            allowed &= ~excluded[user]
        available = np.flatnonzero(allowed)
        if len(available):
            order = np.argsort(-scores[user, available], kind="stable")[:count]
            selected = available[order]
            result[user, : len(selected)] = selected
    return result


def genre_tail_candidates(
    masks: dict[str, np.ndarray],
    genres: list[set[str]],
    tail_items: np.ndarray,
    count: int,
    excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Retrieve tail items by each user's training-history genre preference."""
    train = masks["train"]
    hidden = masks["train"] | masks["valid"]
    popularity = train.sum(axis=0)
    genre_names = sorted(set().union(*genres))
    genre_index = {genre: index for index, genre in enumerate(genre_names)}
    item_genre = np.zeros((len(genres), len(genre_names)), dtype=float)
    for item, item_genres in enumerate(genres):
        for genre in item_genres:
            item_genre[item, genre_index[genre]] = 1.0
    user_genre = train.astype(float) @ item_genre
    denominators = user_genre.sum(axis=1, keepdims=True)
    user_genre = np.divide(user_genre, denominators, out=np.zeros_like(user_genre), where=denominators > 0)
    preference = user_genre @ item_genre.T
    output = np.full((len(train), count), -1, dtype=int)
    available_tail = np.flatnonzero(tail_items)
    for user in range(len(train)):
        candidates = available_tail[~hidden[user, available_tail]]
        if excluded is not None:
            candidates = candidates[~excluded[user, candidates]]
        if not len(candidates):
            continue
        # Preference is primary; train popularity breaks ties within the tail.
        order = np.lexsort((candidates, -popularity[candidates], -preference[user, candidates]))
        selected = candidates[order[:count]]
        output[user, : len(selected)] = selected
    return output


def pool_from_indices(indices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert a padded user-item index matrix to a candidate mask."""
    pool = np.zeros(shape, dtype=bool)
    rows, columns = np.where(indices >= 0)
    pool[rows, indices[rows, columns]] = True
    return pool


def evaluate_candidate_strategy(
    name: str,
    pool: np.ndarray,
    reranker: np.ndarray,
    masks: dict[str, np.ndarray],
    tail_items: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Evaluate candidate recall, reranked quality, and tail recommendation outcomes."""
    hidden = masks["train"] | masks["valid"]
    relevant = masks["test"]
    eligible = relevant.any(axis=1)
    ranked = rerank(reranker, pool, hidden, K, eligible=eligible)
    ranked = ranked[eligible]
    relevant_users = relevant[eligible]
    tail_relevant = relevant_users & tail_items[None, :]
    total_tail = tail_relevant.sum(axis=1)
    candidate_tail = (pool[eligible] & tail_relevant).sum(axis=1)
    candidate_recall = np.divide(candidate_tail, total_tail, out=np.zeros(len(total_tail)), where=total_tail > 0)
    tail_hits = np.take_along_axis(tail_relevant, ranked, axis=1).sum(axis=1)
    tail_recs = tail_items[ranked].sum(axis=1)
    overall_hits = np.take_along_axis(relevant_users, ranked, axis=1).sum(axis=1)
    overall_total = relevant_users.sum(axis=1)
    overall_recall = np.divide(overall_hits, overall_total, out=np.zeros(len(overall_hits)), where=overall_total > 0)
    ndcg = ndcg_per_user(ranked, relevant_users)
    tail_masked = relevant_users & tail_items[None, :]
    tail_candidate_users = total_tail > 0
    output = {
        "strategy": name,
        "candidate_items_mean": float(pool[eligible].sum(axis=1).mean()),
        "candidate_items_median": float(np.median(pool[eligible].sum(axis=1))),
        "candidate_catalogue_coverage": float(pool[eligible].any(axis=0).mean()),
        "tail_candidate_recall": float(candidate_recall[tail_candidate_users].mean()) if tail_candidate_users.any() else 0.0,
        "users_with_tail_candidates": int(((pool[eligible] & tail_items[None, :]).sum(axis=1) > 0).sum()),
        "ndcg@10": float(ndcg.mean()),
        "recall@10": float(overall_recall.mean()),
        "tail_exposure": float(tail_items[ranked].mean()),
        "tail_recall@10": float(np.divide(tail_hits[tail_candidate_users], total_tail[tail_candidate_users]).mean())
        if tail_candidate_users.any()
        else 0.0,
        "tail_hit_rate": float(tail_hits.sum() / tail_recs.sum()) if tail_recs.sum() else 0.0,
        "tail_recommendations": int(tail_recs.sum()),
        "tail_hits": int(tail_hits.sum()),
        "users_with_tail_test_items": int(tail_candidate_users.sum()),
        "tail_test_items": int(tail_masked.sum()),
    }
    return output, ranked


def experiment_specialized_retrieval(
    tables: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    items_path: Path,
    items: np.ndarray,
    output: Path,
) -> tuple[dict, pd.DataFrame]:
    """Compare a genre-aware tail retriever with model-based retrieval sources."""
    _, genres = load_item_metadata(items_path, items)
    train_popularity = masks["train"].sum(axis=0)
    tail_items = bucket_labels(train_popularity) == "tail_bottom50pct"
    hidden = masks["train"] | masks["valid"]
    base_indices = top_finite_indices(tables["Combination"], hidden, 100)
    base_pool = pool_from_indices(base_indices, tables["Combination"].shape)
    add_count = 20
    specialized_indices = genre_tail_candidates(masks, genres, tail_items, add_count, excluded=base_pool)
    source_indices = {
        "LightGCN_top20_unique": top_finite_indices(tables["LightGCN"], hidden, add_count, excluded=base_pool),
        "NeuMF_top20_unique": top_finite_indices(tables["NeuMF"], hidden, add_count, excluded=base_pool),
        "Genre_preference_tail_top20": specialized_indices,
    }
    source_pools = {name: pool_from_indices(indices, base_pool.shape) for name, indices in source_indices.items()}
    strategies = {"Combination_top100_base": base_pool}
    for name, source_pool in source_pools.items():
        strategies[f"Base_plus_{name}"] = base_pool | source_pool
    strategies["Base_plus_LightGCN_and_genre_tail"] = base_pool | source_pools["LightGCN_top20_unique"] | source_pools["Genre_preference_tail_top20"]

    rows = []
    per_user_rows = []
    for strategy, pool in strategies.items():
        summary, ranked_eligible = evaluate_candidate_strategy(strategy, pool, tables["SLIMElastic"], masks, tail_items)
        rows.append(summary)
        user_ids = np.flatnonzero(masks["test"].any(axis=1))
        relevant_users = masks["test"][user_ids]
        tail_relevant_users = relevant_users & tail_items[None, :]
        tail_test_counts = tail_relevant_users.sum(axis=1)
        candidate_tail_counts = (pool[user_ids] & tail_relevant_users).sum(axis=1)
        candidate_tail_recall = np.divide(
            candidate_tail_counts,
            tail_test_counts,
            out=np.zeros(len(user_ids), dtype=float),
            where=tail_test_counts > 0,
        )
        ndcg_user = ndcg_per_user(ranked_eligible, relevant_users)
        for local_index, user_index in enumerate(user_ids):
            per_user_rows.append(
                {
                    "user_index": int(user_index),
                    "strategy": strategy,
                    "tail_exposure": float(tail_items[ranked_eligible[local_index]].mean()),
                    "tail_test_count": int(tail_test_counts[local_index]),
                    "tail_hits": int((relevant_users[local_index, ranked_eligible[local_index]] & tail_items[ranked_eligible[local_index]]).sum()),
                    "tail_candidate_recall": float(candidate_tail_recall[local_index]),
                    "ndcg@10": float(ndcg_user[local_index]),
                }
            )
    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(output / "specialized_retrieval.csv", index=False)
    per_user = pd.DataFrame(per_user_rows)
    per_user.to_csv(output / "specialized_retrieval_per_user.csv", index=False)
    wide = per_user.pivot(
        index="user_index", columns="strategy", values=["ndcg@10", "tail_exposure", "tail_candidate_recall", "tail_hits", "tail_test_count"]
    )
    baseline_name = "Combination_top100_base"
    delta_rows = []
    for strategy in strategies:
        if strategy == baseline_name:
            continue
        ndcg_delta = wide["ndcg@10"][strategy] - wide["ndcg@10"][baseline_name]
        exposure_delta = wide["tail_exposure"][strategy] - wide["tail_exposure"][baseline_name]
        tail_eligible = wide["tail_test_count"][baseline_name] > 0
        candidate_recall_delta = (wide["tail_candidate_recall"][strategy] - wide["tail_candidate_recall"][baseline_name])[tail_eligible]
        tail_recall_delta = (
            wide["tail_hits"][strategy] / wide["tail_test_count"][strategy].replace(0, np.nan)
            - wide["tail_hits"][baseline_name] / wide["tail_test_count"][baseline_name].replace(0, np.nan)
        )[tail_eligible].fillna(0.0)
        ndcg_mean, ndcg_low, ndcg_high = bootstrap_mean_difference(ndcg_delta.to_numpy())
        exposure_mean, exposure_low, exposure_high = bootstrap_mean_difference(exposure_delta.to_numpy())
        candidate_mean, candidate_low, candidate_high = bootstrap_mean_difference(candidate_recall_delta.to_numpy())
        tail_mean, tail_low, tail_high = bootstrap_mean_difference(tail_recall_delta.to_numpy())
        delta_rows.append(
            {
                "strategy": strategy,
                "comparison": f"{strategy} minus {baseline_name}",
                "ndcg_delta": ndcg_mean,
                "ndcg_delta_ci95_low": ndcg_low,
                "ndcg_delta_ci95_high": ndcg_high,
                "tail_exposure_delta": exposure_mean,
                "tail_exposure_delta_ci95_low": exposure_low,
                "tail_exposure_delta_ci95_high": exposure_high,
                "tail_candidate_recall_delta_eligible_users": candidate_mean,
                "tail_candidate_recall_delta_ci95_low": candidate_low,
                "tail_candidate_recall_delta_ci95_high": candidate_high,
                "tail_recall_delta_eligible_users": tail_mean,
                "tail_recall_delta_ci95_low": tail_low,
                "tail_recall_delta_ci95_high": tail_high,
                "tail_recall_users": int(tail_eligible.sum()),
            }
        )
    pd.DataFrame(delta_rows).to_csv(output / "specialized_retrieval_deltas.csv", index=False)
    source_sizes = {
        "combination_base_candidates_mean": float(base_pool[masks["test"].any(axis=1)].sum(axis=1).mean()),
        "addon_unique_candidates_mean": {
            name: float((source_pools[name] & ~base_pool)[masks["test"].any(axis=1)].sum(axis=1).mean()) for name in source_pools
        },
    }
    return {"strategies": rows, "source_sizes": source_sizes}, per_user


def experiment_tail_interest_segments(
    per_user: pd.DataFrame,
    masks: dict[str, np.ndarray],
    output: Path,
) -> dict:
    """Compare retrieval strategies across user groups defined by train tail interest."""
    train_popularity = masks["train"].sum(axis=0)
    tail_items = bucket_labels(train_popularity) == "tail_bottom50pct"
    train = masks["train"]
    counts = train.sum(axis=1)
    tail_counts = (train & tail_items[None, :]).sum(axis=1)
    tail_share = np.divide(tail_counts, counts, out=np.zeros(len(counts), dtype=float), where=counts > 0)
    user_features_frame = pd.DataFrame({"user_index": np.arange(len(train)), "train_tail_interest_share": tail_share})
    add_quantile_group(user_features_frame, "train_tail_interest_share", "tail_interest_quartile")
    joined = per_user.merge(user_features_frame, on="user_index", how="left", validate="many_to_one")
    rows = []
    for group, group_frame in joined.groupby("tail_interest_quartile", sort=True):
        for strategy, strategy_frame in group_frame.groupby("strategy", sort=True):
            eligible_tail = strategy_frame[strategy_frame.tail_test_count > 0]
            tail_recall_values = (eligible_tail.tail_hits / eligible_tail.tail_test_count).to_numpy(dtype=float)
            candidate_values = eligible_tail.tail_candidate_recall.to_numpy(dtype=float)
            if len(eligible_tail):
                candidate_mean, candidate_low, candidate_high = bootstrap_mean_difference(candidate_values)
                tail_mean, tail_low, tail_high = bootstrap_mean_difference(tail_recall_values)
            else:
                candidate_mean = candidate_low = candidate_high = 0.0
                tail_mean = tail_low = tail_high = 0.0
            rows.append(
                {
                    "tail_interest_quartile": group,
                    "strategy": strategy,
                    "users": len(strategy_frame),
                    "mean_train_tail_interest_share": float(strategy_frame.train_tail_interest_share.mean()),
                    "users_with_tail_test_items": len(eligible_tail),
                    "mean_tail_candidate_recall": float(candidate_mean),
                    "tail_candidate_recall_ci95_low": candidate_low,
                    "tail_candidate_recall_ci95_high": candidate_high,
                    "mean_tail_exposure": float(strategy_frame.tail_exposure.mean()),
                    "tail_recall@10": tail_mean,
                    "tail_recall_ci95_low": tail_low,
                    "tail_recall_ci95_high": tail_high,
                    "ndcg@10": float(strategy_frame["ndcg@10"].mean()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "tail_interest_segments.csv", index=False)
    joined.to_csv(output / "tail_interest_per_user.csv", index=False)
    return {"groups": summary.to_dict(orient="records")}


def write_protocol(results: Path, output: Path, items_path: Path, summaries: dict) -> None:
    """Write reproducibility metadata and input hashes."""
    inputs = [results / "split" / f"{split}.tsv" for split in ("train", "valid", "test")]
    inputs += [results / MODEL_PATHS[name] for name in ANALYSIS_MODELS]
    inputs.append(items_path)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    protocol = {
        "experiment": "Task 2.4 insightful hybrid analysis",
        "models": ANALYSIS_MODELS,
        "primary_k": K,
        "candidate_depths": ["full" if depth == 0 else depth for depth in DEPTHS],
        "split": "Train and validation are hidden; test interactions are binary relevant labels.",
        "selection": "No policy is selected from test labels. The routing oracle is descriptive only.",
        "popularity_buckets": "Top 20 percent by train popularity, next 30 percent, and bottom 50 percent.",
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed": 2020, "unit": "test user"},
        "specialized_retrieval": {
            "base_pool": "Top 100 finite-scoring items from the saved Feature Combination export.",
            "add_on_sources": [
                "20 unique LightGCN items beyond the base pool",
                "20 unique NeuMF items beyond the base pool",
                "20 unique tail items selected by training-history genre preference",
            ],
            "reranker": "SLIMElastic for every strategy to hold ranking scores constant.",
            "tail_interest_groups": "Quartiles of each user's training-interaction share from items in the train-defined tail bucket.",
        },
        "summaries": summaries,
        "sha256": hashes,
        "limitations": [
            "One saved MovieLens split and a small catalogue limit generalisation.",
            "Saved score exports are evaluated; this module does not measure retraining variation.",
            "Serendipity is not recomputed here because it includes relevance; exposure and relevance are reported separately.",
            "Specialized retrieval comparisons use fixed candidate additions on the test split and are exploratory, not validation-selected policies.",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")


def run(results: Path, output: Path, items_path: Path) -> dict:
    """Run all Task 2.4 experiments on the saved exports."""
    output.mkdir(parents=True, exist_ok=True)
    users, items, tables = load_exports(results)
    masks = load_masks(results, users, items)
    features = user_features(masks, items_path, items)
    add_quantile_group(features, "train_interactions", "activity_group")
    add_quantile_group(features, "genre_entropy", "entropy_group")
    routing = experiment_routing(tables, masks, features, output)
    candidates_summary = experiment_candidates(tables, masks, output)
    popularity = experiment_popularity(tables, masks, items, output)
    specialized, per_user = experiment_specialized_retrieval(tables, masks, items_path, items, output)
    tail_interest = experiment_tail_interest_segments(per_user, masks, output)
    summaries = {
        "routing": routing,
        "candidate_pipeline": candidates_summary,
        "popularity": popularity,
        "specialized_retrieval": specialized,
        "tail_interest_segments": tail_interest,
    }
    write_protocol(results, output, items_path, summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task24"))
    parser.add_argument("--items", type=Path, default=Path("data/ml-100k/ml-100k.item"))
    args = parser.parse_args()
    summaries = run(args.results, args.output, args.items)
    print(json.dumps(summaries, indent=2))
    print(f"Saved Task 2.4 outputs to {args.output}")


if __name__ == "__main__":
    main()
