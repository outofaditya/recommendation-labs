"""Run Task 2.3 coefficient, ablation, and hybrid contribution analyses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from scripts.hybrids import groups, pairs, ranks, shortlisted
from source.data import RESULTS
from source.hybrids.pool import MEMBERS, candidates, folds
from source.metrics.accuracy import accuracy, top

GRID = [10.0**power for power in range(-7, 1)]
K = 10
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from source import data


def load_data(results):
    tables = [np.load(results / "tuned" / "scores" / f"{name}.npz") for name in MEMBERS]
    users, items = tables[0]["users"], tables[0]["items"]
    if any(not (np.array_equal(t["users"], users) and np.array_equal(t["items"], items)) for t in tables):
        raise ValueError("Individual score exports do not share identical user and item order")
    user_rows = {value: index for index, value in enumerate(users)}
    item_cols = {value: index for index, value in enumerate(items)}
    masks = {}
    for name in ("train", "valid", "test"):
        frame = pd.read_csv(results / "split" / f"{name}.tsv", sep="\t", dtype=str)
        mask = np.zeros((len(users), len(items)), dtype=bool)
        mask[frame.iloc[:, 0].map(user_rows), frame.iloc[:, 1].map(item_cols)] = True
        masks[name] = mask
    raw = np.stack([t["scores"] for t in tables], axis=-1).astype(np.float64)
    outside_train = ~masks["train"][..., None]
    mean = raw.mean(axis=1, keepdims=True, where=outside_train)
    std = raw.std(axis=1, keepdims=True, where=outside_train)
    x = np.divide(raw - mean, std, out=np.zeros_like(raw), where=std > 0)
    return x, masks, users, items


def ndcg(scores, hidden, relevant):
    return accuracy(top(scores, hidden), relevant)["ndcg@10"]


def fit_weights(x, labels, c):
    model = LogisticRegression(C=c, max_iter=1000).fit(x, labels)
    return model.coef_[0]


def weighted_rank(x, mask, hidden, relevant, weights):
    return ndcg(np.where(mask, x @ weights, -np.inf), hidden, relevant)


def weighted_cv(x, masks, keep, c, user_folds):
    sub = x[..., keep]
    values, fold_weights = [], []
    for held in user_folds:
        pool = candidates(sub, masks["train"])
        train_rows = pool.copy()
        train_rows[held] = False
        weights = fit_weights(sub[train_rows], masks["valid"][train_rows], c)
        values.append(weighted_rank(sub[held], pool[held], masks["train"][held], masks["valid"][held], weights))
        fold_weights.append(weights)
    return float(np.mean(values)), np.asarray(fold_weights)


def coefficient_and_ablation(x, masks, out):
    user_folds = folds(len(x))
    all_members = np.arange(len(MEMBERS))
    reg_rows, coef_rows = [], []
    for c in GRID:
        score, weights = weighted_cv(x, masks, all_members, c, user_folds)
        reg_rows.append({"C": c, "cv_ndcg@10": score})
        for index, name in enumerate(MEMBERS):
            coef_rows.append(
                {
                    "C": c,
                    "member": name,
                    "mean_coefficient": float(weights[:, index].mean()),
                    "sd_across_folds": float(weights[:, index].std()),
                    "mean_absolute_coefficient": float(np.abs(weights[:, index]).mean()),
                }
            )

    selected_c = max(reg_rows, key=lambda row: row["cv_ndcg@10"])["C"]
    baseline, _ = weighted_cv(x, masks, all_members, selected_c, user_folds)
    ablations = []
    fixed_ablations = []
    full_pool = candidates(x, masks["train"])
    for removed, name in enumerate(MEMBERS):
        keep = all_members[all_members != removed]
        score, _ = weighted_cv(x, masks, keep, selected_c, user_folds)
        ablations.append({"removed_member": name, "cv_ndcg@10": score, "full_minus_ablated": baseline - score})

        # Fixed-weight ablation keeps the full candidate pool and the fold's
        # fitted coefficients, setting only the removed member's coefficient
        # to zero. This separates direct score contribution from refitting
        # compensation by correlated members.
        values = []
        for held in user_folds:
            train_rows = full_pool.copy()
            train_rows[held] = False
            weights = fit_weights(x[train_rows], masks["valid"][train_rows], selected_c)
            weights[removed] = 0.0
            values.append(weighted_rank(x[held], full_pool[held], masks["train"][held], masks["valid"][held], weights))
        fixed_score = float(np.mean(values))
        fixed_ablations.append(
            {
                "removed_member": name,
                "cv_ndcg@10": fixed_score,
                "full_minus_ablated": baseline - fixed_score,
            }
        )

    final_pool = candidates(x, masks["train"])
    final_weights = fit_weights(x[final_pool], masks["valid"][final_pool], selected_c)
    magnitude = np.abs(final_weights)
    total = magnitude.sum()
    final = []
    for i, name in enumerate(MEMBERS):
        fold_values = np.asarray([r["mean_coefficient"] for r in coef_rows if r["C"] == selected_c and r["member"] == name])
        final.append(
            {
                "member": name,
                "coefficient": float(final_weights[i]),
                "absolute_magnitude_share": float(magnitude[i] / total) if total else 0.0,
                "cv_mean_coefficient": float(fold_values[0]),
                "cv_sd_across_folds": next(r["sd_across_folds"] for r in coef_rows if r["C"] == selected_c and r["member"] == name),
            }
        )
    pd.DataFrame(reg_rows).to_csv(out / "weighted_regularization.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out / "weighted_coefficients_by_C.csv", index=False)
    pd.DataFrame(final).sort_values("absolute_magnitude_share", ascending=False).to_csv(out / "weighted_coefficients.csv", index=False)
    pd.DataFrame(ablations).sort_values("full_minus_ablated", ascending=False).to_csv(out / "weighted_leave_one_out.csv", index=False)
    pd.DataFrame(fixed_ablations).sort_values("full_minus_ablated", ascending=False).to_csv(out / "weighted_fixed_weight_ablation.csv", index=False)
    return {
        "selected_C": selected_c,
        "cv_ndcg@10": baseline,
        "coefficients": final,
        "ablation": ablations,
        "fixed_weight_ablation": fixed_ablations,
        "regularization": reg_rows,
    }


def correlation_analysis(x, masks, out):
    """Measure redundancy between standardized member scores on candidates."""
    pool = candidates(x, masks["train"])
    matrix = np.corrcoef(x[pool].T)
    rows = []
    for i, left in enumerate(MEMBERS):
        for j, right in enumerate(MEMBERS):
            rows.append({"left_member": left, "right_member": right, "pearson_correlation": float(matrix[i, j])})
    pd.DataFrame(rows).to_csv(out / "member_score_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(MEMBERS)), MEMBERS, rotation=90)
    ax.set_yticks(range(len(MEMBERS)), MEMBERS)
    ax.set_title("Member score correlation on the Weighted candidate pool")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(out / "member_score_correlations.png", dpi=180)
    plt.close(fig)
    return {"candidate_rows": int(pool.sum()), "matrix": matrix.tolist()}


def activity_analysis(x, masks, out):
    """Evaluate Weighted and individual models by user-history group."""
    user_groups = groups(masks, 4)
    user_folds = folds(len(x))
    pool = candidates(x, masks["train"])
    rows = []
    for group_index, user_group in enumerate(user_groups, start=1):
        full_scores, member_scores = [], {name: [] for name in MEMBERS}
        for held in user_folds:
            held_group = np.intersect1d(held, user_group, assume_unique=False)
            if len(held_group) == 0:
                continue
            train_rows = pool.copy()
            train_rows[held] = False
            weights = fit_weights(x[train_rows], masks["valid"][train_rows], 0.0001)
            full_scores.append(weighted_rank(x[held_group], pool[held_group], masks["train"][held_group], masks["valid"][held_group], weights))
            for index, name in enumerate(MEMBERS):
                member_scores[name].append(ndcg(x[held_group, :, index], masks["train"][held_group], masks["valid"][held_group]))
        individual = {name: float(np.mean(values)) for name, values in member_scores.items()}
        best_name = max(individual, key=individual.get)
        best_score = individual[best_name]
        counts = masks["train"][user_group].sum(axis=1)
        rows.append(
            {
                "activity_group": group_index,
                "train_count_min": int(counts.min()),
                "train_count_max": int(counts.max()),
                "weighted_ndcg@10": float(np.mean(full_scores)),
                "best_individual": best_name,
                "best_individual_ndcg@10": best_score,
                "weighted_minus_best": float(np.mean(full_scores) - best_score),
                "member_ndcg@10": json.dumps(individual, sort_keys=True),
            }
        )
    pd.DataFrame(rows).to_csv(out / "weighted_by_activity.csv", index=False)
    return rows


def activity_specific_weight_analysis(x, masks, out):
    """Fit and evaluate a separate Weighted policy for each activity group."""
    user_groups = groups(masks, 4)
    user_folds = folds(len(x))
    pool = candidates(x, masks["train"])
    selected_c = 0.0001
    performance = []
    coefficient_rows = []

    for group_index, user_group in enumerate(user_groups, start=1):
        group_mask = np.zeros(len(x), dtype=bool)
        group_mask[user_group] = True
        group_values, global_values = [], []
        fold_weights = []
        for held in user_folds:
            held_group = np.intersect1d(held, user_group, assume_unique=False)
            if len(held_group) == 0:
                continue
            group_train_users = group_mask.copy()
            group_train_users[held] = False
            group_train_rows = pool & group_train_users[:, None]
            weights = fit_weights(x[group_train_rows], masks["valid"][group_train_rows], selected_c)
            fold_weights.append(weights)
            group_values.append(
                weighted_rank(
                    x[held_group],
                    pool[held_group],
                    masks["train"][held_group],
                    masks["valid"][held_group],
                    weights,
                )
            )

            # This is the same global policy evaluated on the same held-out users.
            global_train_rows = pool.copy()
            global_train_rows[held] = False
            global_weights = fit_weights(x[global_train_rows], masks["valid"][global_train_rows], selected_c)
            global_values.append(
                weighted_rank(
                    x[held_group],
                    pool[held_group],
                    masks["train"][held_group],
                    masks["valid"][held_group],
                    global_weights,
                )
            )

        fold_weights = np.asarray(fold_weights)
        counts = masks["train"][user_group].sum(axis=1)
        group_score = float(np.mean(group_values))
        global_score = float(np.mean(global_values))
        performance.append(
            {
                "activity_group": group_index,
                "train_count_min": int(counts.min()),
                "train_count_max": int(counts.max()),
                "group_specific_ndcg@10": group_score,
                "global_weighted_ndcg@10": global_score,
                "group_specific_minus_global": group_score - global_score,
                "users": len(user_group),
            }
        )
        for member_index, member in enumerate(MEMBERS):
            coefficient_rows.append(
                {
                    "activity_group": group_index,
                    "train_count_min": int(counts.min()),
                    "train_count_max": int(counts.max()),
                    "member": member,
                    "mean_coefficient": float(fold_weights[:, member_index].mean()),
                    "sd_across_folds": float(fold_weights[:, member_index].std()),
                }
            )

    pd.DataFrame(performance).to_csv(out / "weighted_by_activity_policy.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(out / "weighted_activity_coefficients.csv", index=False)
    return {
        "C": selected_c,
        "performance": performance,
        "coefficients": coefficient_rows,
        "fit_scope": "Each activity group's validation labels, excluding the held-out user fold; no test labels.",
    }


def activity_marginal_analysis(x, masks, out):
    """Measure each member's marginal value separately within each activity group."""
    user_groups = groups(masks, 4)
    user_folds = folds(len(x))
    all_members = np.arange(len(MEMBERS))
    selected_c = 0.0001
    rows = []

    for group_index, user_group in enumerate(user_groups, start=1):
        group_mask = np.zeros(len(x), dtype=bool)
        group_mask[user_group] = True
        counts = masks["train"][user_group].sum(axis=1)

        for removed, removed_name in enumerate(MEMBERS):
            keep = all_members[all_members != removed]
            sub = x[..., keep]
            group_pool = candidates(sub, masks["train"])
            full_pool = candidates(x, masks["train"])
            full_values, ablated_values = [], []

            for held in user_folds:
                held_group = np.intersect1d(held, user_group, assume_unique=False)
                if len(held_group) == 0:
                    continue
                group_train_users = group_mask.copy()
                group_train_users[held] = False

                full_train_rows = full_pool & group_train_users[:, None]
                full_weights = fit_weights(x[full_train_rows], masks["valid"][full_train_rows], selected_c)
                full_values.append(
                    weighted_rank(
                        x[held_group],
                        full_pool[held_group],
                        masks["train"][held_group],
                        masks["valid"][held_group],
                        full_weights,
                    )
                )

                ablated_train_rows = group_pool & group_train_users[:, None]
                ablated_weights = fit_weights(sub[ablated_train_rows], masks["valid"][ablated_train_rows], selected_c)
                ablated_values.append(
                    weighted_rank(
                        sub[held_group],
                        group_pool[held_group],
                        masks["train"][held_group],
                        masks["valid"][held_group],
                        ablated_weights,
                    )
                )

            full_score = float(np.mean(full_values))
            ablated_score = float(np.mean(ablated_values))
            rows.append(
                {
                    "activity_group": group_index,
                    "train_count_min": int(counts.min()),
                    "train_count_max": int(counts.max()),
                    "removed_member": removed_name,
                    "full_group_specific_ndcg@10": full_score,
                    "ablated_group_specific_ndcg@10": ablated_score,
                    "marginal_contribution_full_minus_ablated": full_score - ablated_score,
                }
            )

    pd.DataFrame(rows).sort_values(
        ["activity_group", "marginal_contribution_full_minus_ablated"],
        ascending=[True, False],
    ).to_csv(out / "weighted_leave_one_out_by_activity.csv", index=False)
    return rows


def mixed_analysis(x, masks):
    metadata = json.loads((RESULTS / "hybrid" / "metrics" / "TunedMixed.json").read_text())
    members = metadata["members"]
    idx = [MEMBERS.index(name) for name in members]
    hidden, relevant = masks["train"], masks["valid"]
    rank_table = ranks(x, hidden)
    full_score = (1 / (metadata["k"] + rank_table[..., idx])).sum(axis=-1)
    base = ndcg(full_score, hidden, relevant)
    rows = [{"removed_member": "None", "validation_ndcg@10": base, "full_minus_ablated": 0.0}]
    for removed in members:
        keep = [j for j, name in zip(idx, members) if name != removed]
        score = ndcg((1 / (metadata["k"] + rank_table[..., keep])).sum(axis=-1), hidden, relevant)
        rows.append({"removed_member": removed, "validation_ndcg@10": score, "full_minus_ablated": base - score})
    return {"members": members, "k": metadata["k"], "validation_ndcg@10": base, "ablation": rows}


def switching_analysis(x, masks):
    config = json.loads((RESULTS / "hybrid" / "metrics" / "TunedSwitching.json").read_text())
    user_groups = groups(masks, len(config["groups"]))
    result = []
    for users, detail in zip(user_groups, config["groups"]):
        member_index = MEMBERS.index(detail["member"])
        selected = ndcg(x[users, :, member_index], masks["train"][users], masks["valid"][users])
        all_scores = [ndcg(x[users, :, j], masks["train"][users], masks["valid"][users]) for j in range(len(MEMBERS))]
        best_index = int(np.argmax(all_scores))
        result.append(
            {
                "train_count_min": int(masks["train"][users].sum(axis=1).min()),
                "train_count_max": int(masks["train"][users].sum(axis=1).max()),
                "selected_member": detail["member"],
                "selected_validation_ndcg@10": selected,
                "best_member_on_same_group": MEMBERS[best_index],
                "best_member_validation_ndcg@10": all_scores[best_index],
                "selected_minus_best": selected - all_scores[best_index],
            }
        )
    return result


def cascade_analysis(x, masks):
    config = json.loads((RESULTS / "hybrid" / "metrics" / "TunedCascade.json").read_text())
    first, second = (x[..., MEMBERS.index(name)] for name in config["stages"])
    hidden, relevant, size = masks["train"], masks["valid"], config["shortlist"]
    shortlist = np.argsort(np.where(hidden, np.inf, -first), axis=1, kind="stable")[:, :size]
    candidate_mask = np.zeros_like(hidden)
    np.put_along_axis(candidate_mask, shortlist, True, axis=1)
    candidate_recall = float((candidate_mask & relevant).sum() / relevant.sum())
    cascade = shortlisted(first, second, hidden, size)
    rows = [
        {"variant": "First stage full ranking", "validation_ndcg@10": ndcg(first, hidden, relevant)},
        {"variant": f"First stage shortlist ({size}) then second stage rank", "validation_ndcg@10": ndcg(cascade, hidden, relevant)},
        {"variant": "Second stage full ranking", "validation_ndcg@10": ndcg(second, hidden, relevant)},
    ]
    return {"stages": config["stages"], "shortlist": size, "validation_relevant_candidate_recall": candidate_recall, "variants": rows}


def combination_analysis(x, masks):
    train, valid = masks["train"], masks["valid"]
    pool = candidates(x, train)
    owners = np.nonzero(pool)[0]
    member_rows = pairs(x, train)
    full_rows = member_rows(pool)
    score_count = len(MEMBERS)
    # Feature blocks are defined by pairs(): member scores, user attributes/activity, movie genres/year/popularity.
    user_end = score_count + 3
    variants = {
        "All member scores and user/movie side features": np.arange(full_rows.shape[1]),
        "Member scores only": np.arange(score_count),
        "Member scores plus user features": np.arange(user_end),
        "Member scores plus movie features": np.r_[0:score_count, user_end : full_rows.shape[1]],
    }
    rows = []
    for name, columns in variants.items():
        fold_scores = []
        for held in folds(len(x)):
            train_rows = ~np.isin(owners, held)
            model = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.1, random_state=2020)
            # Occupation is categorical only when it remains in the selected user-feature block.
            category = np.flatnonzero(columns == score_count + 2)
            model.set_params(categorical_features=category.tolist() if len(category) else None)
            model.fit(full_rows[train_rows][:, columns], valid[pool][train_rows])
            table = np.full(train.shape, -np.inf)
            table[pool] = model.decision_function(full_rows[:, columns])
            fold_scores.append(ndcg(table[held], train[held], valid[held]))
        rows.append({"variant": name, "cv_ndcg@10": float(np.mean(fold_scores)), "fold_sd": float(np.std(fold_scores))})
    return rows


def run(results=Path("results"), output=Path("results/task23")):
    global RESULTS
    RESULTS = Path(results)
    data.RESULTS = RESULTS
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, masks, _, _ = load_data(RESULTS)
    summary = {
        "protocol": {
            "split": str(RESULTS / "split"),
            "folds": 5,
            "fold_seed": 2020,
            "k": K,
            "primary_metric": "NDCG@10",
            "test_usage": "No Task 2.3 model fitting, member selection, or ablation uses test labels.",
        },
        "weighted": coefficient_and_ablation(x, masks, output),
        "correlations": correlation_analysis(x, masks, output),
        "weighted_by_activity": activity_analysis(x, masks, output),
        "activity_specific_weights": activity_specific_weight_analysis(x, masks, output),
        "activity_marginal_contribution": activity_marginal_analysis(x, masks, output),
        "mixed": mixed_analysis(x, masks),
        "switching": switching_analysis(x, masks),
        "cascade": cascade_analysis(x, masks),
        "feature_combination": combination_analysis(x, masks),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summary["switching"]).to_csv(output / "switching_by_user_group.csv", index=False)
    pd.DataFrame(summary["cascade"]["variants"]).to_csv(output / "cascade_stages.csv", index=False)
    pd.DataFrame(summary["feature_combination"]).to_csv(output / "combination_feature_ablation.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "protocol"}, indent=2))
    print(f"Saved Task 2.3 outputs to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/task23"))
    args = parser.parse_args()
    run(args.results, args.output)


if __name__ == "__main__":
    main()
