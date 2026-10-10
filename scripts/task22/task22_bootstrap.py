"""Task 2.2 K sweep with user-bootstrap confidence intervals.

This evaluates the currently saved exports and split. Bootstrap intervals
describe uncertainty from sampling test users, not retraining variation.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.task22.compare_hybrids import HYBRIDS
from scripts.task22.compare_individuals import MODELS, load_item_features, load_masks, load_predictions
from source.metrics.beyond_accuracy import BeyondAccuracyMetrics, jaccard_distance
from source.metrics.evaluation_metrics import EvaluationMetrics

KS = (1, 3, 5, 10, 20)
RANKING = ("ndcg", "recall", "precision", "mrr", "hit", "map", "f1")
BEYOND = ("coverage", "novelty", "ild", "serendipity")


def _user_metrics(scores, hidden, relevant, distance, k):
    masked = np.where(hidden, -np.inf, scores)
    top = np.argsort(-masked, axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(relevant, top, axis=1)
    counts = relevant.sum(axis=1)
    eligible = counts > 0
    top, hits, counts = top[eligible], hits[eligible], counts[eligible]
    hit_count = hits.sum(axis=1)
    found = hit_count > 0
    discount = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discount)[np.minimum(counts, k) - 1]
    precision_path = hits.cumsum(axis=1) / np.arange(1, k + 1)
    average_precision = (precision_path * hits).sum(axis=1) / np.minimum(counts, k)

    # Per-user components allow user resampling; coverage is recomputed from
    # the sampled recommendation lists for every bootstrap draw.
    item_counts = hidden.sum(axis=0)
    total = item_counts.sum()
    probabilities = item_counts / total if total else np.zeros(scores.shape[1])
    novelty = -np.log2(np.clip(probabilities[top], 1e-9, 1)).mean(axis=1)
    pairwise = distance[top[:, :, None], top[:, None, :]]
    ild = (pairwise.sum(axis=(1, 2)) - np.trace(pairwise, axis1=1, axis2=2)) / (k * (k - 1)) if k > 1 else np.zeros(len(top))
    user_rows = np.flatnonzero(eligible)
    histories = [np.flatnonzero(hidden[u]) for u in user_rows]
    serendipity = np.zeros(len(top))
    for row, history in enumerate(histories):
        if len(history):
            unexpectedness = distance[np.ix_(top[row], history)].mean(axis=1)
            serendipity[row] = (hits[row] * unexpectedness).mean()

    values = {
        "ndcg": (hits * discount).sum(axis=1) / ideal,
        "recall": hit_count / counts,
        "precision": hit_count / k,
        "mrr": np.where(found, 1 / (hits.argmax(axis=1) + 1), 0),
        "hit": found.astype(float),
        "map": average_precision,
        "f1": 2 * hit_count / (k + counts),
        "novelty": novelty,
        "ild": ild,
        "serendipity": serendipity,
    }
    return top, values, user_rows


def _rating_errors(users, items, scores, test_pairs, ratings):
    u_index = {str(value): i for i, value in enumerate(users)}
    i_index = {str(value): i for i, value in enumerate(items)}
    abs_error = np.zeros(len(users), dtype=float)
    sq_error = np.zeros(len(users), dtype=float)
    pair_count = np.zeros(len(users), dtype=int)
    for row in test_pairs.itertuples(index=False):
        user, item = str(row[0]), str(row[1])
        if user in u_index and item in i_index and (user, item) in ratings:
            prediction = scores[u_index[user], i_index[item]]
            if np.isfinite(prediction):
                delta = prediction - ratings[(user, item)]
                abs_error[u_index[user]] += abs(delta)
                sq_error[u_index[user]] += delta * delta
                pair_count[u_index[user]] += 1
    return abs_error, sq_error, pair_count


def _make_plots(summary, output, ks):
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    models = summary.model.drop_duplicates().tolist()
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(models)))
    colors = dict(zip(models, palette, strict=True))
    for metric in list(RANKING) + list(BEYOND):
        frame = summary[(summary.metric == metric) & (summary.k != "all")].copy()
        frame["k"] = frame["k"].astype(int)
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for name, group in frame.groupby("model", sort=False):
            group = group.sort_values("k")
            x = group.k.to_numpy()
            color = colors[name]
            ax.plot(x, group.value, color=color, marker="o", linewidth=1.5, label=name)
            ax.fill_between(x, group.ci95_low, group.ci95_high, color=color, alpha=0.09)
        ax.set(title=f"{metric.upper()} by recommendation K", xlabel="K", ylabel=metric.upper(), xticks=list(ks))
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_by_k.png", dpi=180)
        plt.close(fig)


def run(
    results=Path("results"),
    output=Path("results/task2/repeated"),
    reps=10000,
    seed=22022,
    ks=KS,
    items_path=Path("data/ml-100k/ml-100k.item"),
    interactions_path=Path("data/ml-100k/ml-100k.inter"),
):
    if reps < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if not ks or any(k < 1 or k > 20 for k in ks):
        raise ValueError("K values must be between 1 and 20 for the prepared top-20 sweep")

    models = MODELS + HYBRIDS
    first_path = results / "tuned/scores" / f"{MODELS[0]}.npz"
    users, items, _ = load_predictions(first_path)
    masks = load_masks(results / "split", users, items)
    hidden, relevant = masks["train"] | masks["valid"], masks["test"]
    eligible = np.flatnonzero(relevant.any(axis=1))
    features = load_item_features(items_path, items)
    distance = BeyondAccuracyMetrics.build_genre_distance_matrix(features, len(items), jaccard_distance)
    test_pairs = pd.read_csv(results / "split/test.tsv", sep="\t", dtype=str)
    ratings = EvaluationMetrics.load_interaction_ratings(interactions_path)

    prepared = {}
    for name in models:
        hybrid = name in HYBRIDS
        path = results / ("hybrid/scores" if hybrid else "tuned/scores") / f"{name}.npz"
        model_users, model_items, scores = load_predictions(path, allow_excluded=hybrid)
        if not np.array_equal(model_users, users) or not np.array_equal(model_items, items):
            raise ValueError(f"User/item order differs for {name}")
        available = np.isfinite(scores) & ~hidden
        if np.any(available[eligible].sum(axis=1) < max(ks)):
            raise ValueError(f"{name} has fewer than K={max(ks)} finite unseen candidates for a test user")
        by_k = {}
        for k in ks:
            top, values, user_rows = _user_metrics(scores, hidden, relevant, distance, k)
            by_k[k] = (top, values, user_rows)
        ae, se, n = _rating_errors(users, items, scores, test_pairs, ratings)
        prepared[name] = (by_k, ae, se, n)
        print(f"Prepared {name}", flush=True)

    rng = np.random.default_rng(seed)
    draws = {(name, k, metric): np.empty(reps, dtype=float) for name in models for k in ks for metric in (list(RANKING) + list(BEYOND))}
    draws.update({(name, 0, metric): np.empty(reps, dtype=float) for name in models for metric in ("mae", "mse")})
    point_rows = []

    for name, (by_k, ae, se, pair_n) in prepared.items():
        for k, (top, values, user_rows) in by_k.items():
            for metric in RANKING:
                point_rows.append({"model": name, "k": k, "metric": metric, "value": float(values[metric].mean())})
            point_rows.append({"model": name, "k": k, "metric": "coverage", "value": float(np.unique(top[user_rows]).size / len(items))})
            for metric in ("novelty", "ild", "serendipity"):
                point_rows.append({"model": name, "k": k, "metric": metric, "value": float(values[metric].mean())})
            positions = {u: pos for pos, u in enumerate(user_rows)}
            for draw in range(reps):
                sampled = rng.choice(user_rows, size=len(user_rows), replace=True)
                local = np.fromiter((positions[u] for u in sampled), dtype=int, count=len(sampled))
                for metric in RANKING + ("novelty", "ild", "serendipity"):
                    draws[name, k, metric][draw] = float(values[metric][local].mean())
                draws[name, k, "coverage"][draw] = np.unique(top[local]).size / len(items)
        point_rows.extend(
            [
                {"model": name, "k": "all", "metric": "mae", "value": float(ae.sum() / pair_n.sum())},
                {"model": name, "k": "all", "metric": "mse", "value": float(se.sum() / pair_n.sum())},
            ]
        )
        # ``pair_n`` is the per-user count of finite test-pair predictions.
        # Bootstrap only users contributing at least one rating error; the
        # previous code referenced an undefined ``valid`` variable here.
        users_with_ratings = np.flatnonzero(pair_n > 0)
        for draw in range(reps):
            sampled = rng.choice(users_with_ratings, size=len(users_with_ratings), replace=True)
            denomin = pair_n[sampled].sum()
            draws[name, 0, "mae"][draw] = ae[sampled].sum() / denomin
            draws[name, 0, "mse"][draw] = se[sampled].sum() / denomin
        print(f"Bootstrapped {name}", flush=True)

    point = pd.DataFrame(point_rows)
    ci_rows = []
    for (name, k, metric), samples in draws.items():
        ci_rows.append(
            {
                "model": name,
                "k": "all" if k == 0 else k,
                "metric": metric,
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "bootstrap_reps": reps,
            }
        )
    intervals = pd.DataFrame(ci_rows)
    summary = point.merge(intervals, on=["model", "k", "metric"], validate="one_to_one")
    output.mkdir(parents=True, exist_ok=True)
    point.to_csv(output / "point_estimates.csv", index=False)
    intervals.to_csv(output / "confidence_intervals.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    _make_plots(summary, output, ks)

    protocol = {
        "experiment": "Task 2.2 K sweep with user-bootstrap confidence intervals",
        "k_values": list(ks),
        "bootstrap_replicates": reps,
        "bootstrap_seed": seed,
        "uncertainty_scope": "Percentile bootstrap over test users with replacement; captures test-user sampling uncertainty, not retraining/random-seed variability.",
        "split_and_exports": "One existing saved split and one fixed export per model.",
        "models": models,
        "users_bootstrapped": len(eligible),
        "ranking_relevance": "Binary test interaction; train and validation items hidden.",
        "rating_metrics": "MAE/MSE are computed once (K-independent), by pooled finite test-pair predictions; bootstrap resamples users and their test pairs.",
        "beyond_accuracy": "Coverage, hidden-interaction self-information novelty, genre-Jaccard ILD, relevance-weighted genre-distance serendipity.",
        "limitations": [
            "Does not include variation from retraining on different seeds or split generation.",
            "Bootstrap intervals are descriptive; paired model comparisons should use shared bootstrap user samples for formal difference intervals.",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Task 2.2 K sweep and user-bootstrap uncertainty\n\n"
        f"Evaluated {len(models)} saved model exports at K={', '.join(map(str, ks))}, with {reps:,} user-bootstrap replicates. "
        "The intervals quantify uncertainty from the sampled test users for this fixed split and fixed exports; they do not quantify variation from retraining. "
        "MAE and MSE are K-independent and summarized once per model.\n\n"
        "See `summary.csv` for estimates and 95% percentile intervals, `plots/` for metric-versus-K figures, and `protocol.json` for details.\n"
    )
    return summary, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task2/repeated"))
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=22022)
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--items", type=Path, default=Path("data/ml-100k/ml-100k.item"))
    parser.add_argument("--interactions", type=Path, default=Path("data/ml-100k/ml-100k.inter"))
    parser.add_argument("--plots-only", action="store_true", help="Regenerate figures from an existing summary.csv")
    args = parser.parse_args()
    if args.plots_only:
        summary = pd.read_csv(args.output / "summary.csv")
        _make_plots(summary, args.output, args.ks)
        print(f"Regenerated plots in {args.output / 'plots'}", flush=True)
        return
    summary, _ = run(args.results, args.output, args.reps, args.seed, args.ks, args.items, args.interactions)
    print(f"Saved {len(summary)} estimates to {args.output}", flush=True)


if __name__ == "__main__":
    main()
