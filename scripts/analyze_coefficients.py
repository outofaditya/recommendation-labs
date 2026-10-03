"""Task 2.3: analyse existing Weighted coefficients without fitting any hybrid."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from source.metrics.evaluation import evaluate, load_masks, load_model, top_items
from source.hybrids.pool import candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--weights", type=Path, help="Existing Weighted/TunedWeighted metric JSON")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--depth", type=int, help="Candidate depth; defaults to saved depth or legacy depth 100",)
    parser.add_argument("--allow-missing", action="store_true", help="Partial standalone and correlation diagnostics; no hybrid contributions",)
    parser.add_argument("--output", type=Path, default=Path("results/task2/coefficients"))
    args = parser.parse_args()
    path = args.weights
    if path is None:
        paths = [args.results / "hybrid/metrics" / f"{n}.json" for n in ("TunedWeighted", "Weighted")]
        path = next((p for p in paths if p.exists()), None)
    if path is None or not path.exists():
        parser.error("No coefficient artifact. Supply --weights pointing to an existing Weighted metric JSON; this script does not fit weights.")
    metadata = json.loads(path.read_text())
    weights = metadata.get("weights")
    if not isinstance(weights, dict) or not weights:
        parser.error("Coefficient JSON must contain a non-empty 'weights' mapping")
    members = list(weights)
    beta = np.array([weights[n] for n in members], dtype=float)
    if not np.isfinite(beta).all():
        parser.error("Coefficients must be finite")
    depth = args.depth if args.depth is not None else int(metadata.get("depth", 100))
    if depth < 1 or args.k < 1:
        parser.error("depth and k must be positive")
    if "depth" in metadata and depth != metadata["depth"]:
        parser.error("depth differs from coefficient artifact; use its fitting candidate depth")
    loaded, tables, missing, reference = [], [], [], None
    rows = []
    fold_weights = metadata.get("fold_weights")
    if fold_weights is not None:
        fold_weights = np.asarray(fold_weights, dtype=float)
        if (
            fold_weights.ndim != 2
            or fold_weights.shape[1] != len(members)
            or not np.isfinite(fold_weights).all()
            or len(fold_weights) < 2
        ):
            parser.error("fold_weights must be a finite folds-by-members matrix with at least two folds")
    for j, name in enumerate(members):
        spread = (float(fold_weights[:, j].std()) if fold_weights is not None else metadata.get("spread", {}).get(name))
        row = {
            "member": name,
            "weight": float(beta[j]),
            "absolute_weight": float(abs(beta[j])),
            "fold_std": spread,
            "fold_sign_agreement": float((np.sign(fold_weights[:, j]) == np.sign(beta[j])).mean())
            if fold_weights is not None
            else None,
        }
        try:
            users, items, scores = load_model(args.results, name)
        except FileNotFoundError:
            if not args.allow_missing:
                parser.error(f"Missing member {name}. Use --allow-missing for partial diagnostics.")
            missing.append(name)
            row["scores_available"] = False
            rows.append(row)
            continue
        if reference is None:
            reference = users, items
            masks = load_masks(args.results, users, items)
        elif not (np.array_equal(users, reference[0]) and np.array_equal(items, reference[1])):
            raise ValueError(f"Export ID order differs: {name}")
        measured, _ = evaluate(users, items, scores, masks, args.k)
        row.update({"scores_available": True, **measured})
        outside = ~masks["train"]
        values = scores.astype(float)
        mean = values.mean(axis=1, keepdims=True, where=outside)
        std = values.std(axis=1, keepdims=True, where=outside)
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std == 0)
        ):
            raise ValueError(f"Undefined per-user standardisation: {name}")
        tables.append((values - mean) / std)
        loaded.append(name)
        rows.append(row)
    if not loaded:
        parser.error("No member scores available")
    x = np.stack(tables, axis=-1)
    # The fitting pool is the union of each member's top-depth outside train.
    pool = (
        candidates(x, masks["train"], depth) & ~masks["train"]
        if depth < len(items)
        else ~masks["train"]
    )
    samples = x[pool]
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = (
            np.atleast_2d(np.corrcoef(samples, rowvar=False))
            if len(loaded) > 1
            else np.ones((1, 1))
        )
    ablations, contributions = [], []
    verified = False
    if not missing:
        # Analyse the exported hybrid's fixed candidate pool. No fitting or model construction.
        hybrid_path = args.results / "hybrid/scores" / f"{path.stem}.npz"
        if hybrid_path.exists():
            hu, hi, hybrid = load_model(args.results, path.stem)
            if not (np.array_equal(hu, users) and np.array_equal(hi, items)):
                raise ValueError("Hybrid and member ID orders differ")
            hidden = masks["train"] | masks["valid"]
            available = np.isfinite(hybrid) & ~hidden
            predicted = x @ beta
            # Legacy files round coefficients to 4 decimals; report verification tolerance.
            tolerance = 1e-5 + 5e-5 * np.abs(x).sum(axis=-1)
            if np.any(
                np.abs(predicted[available] - hybrid[available]) > tolerance[available]
            ):
                raise ValueError(
                    "Weighted scores do not match coefficients/member exports"
                )
            verified = True
            full, _ = evaluate(users, items, hybrid, masks, args.k)
            ranking = top_items(hybrid, hidden, args.k)
            eligible = np.flatnonzero(masks["test"].any(axis=1))
            for j, name in enumerate(members):
                without = np.where(available, hybrid - beta[j] * x[..., j], -np.inf)
                measured, _ = evaluate(users, items, without, masks, args.k)
                ablations.append(
                    {
                        "member": name,
                        **{
                            f"full_minus_zero_weight_{metric}": full[metric]
                            - measured[metric]
                            for metric in full
                            if metric in measured
                            and metric.split("@")[0]
                            in (
                                "ndcg",
                                "recall",
                                "hit",
                                "mrr",
                                "precision",
                                "map",
                                "f1",
                            )
                        },
                    }
                )
                per_user = []
                per_hit_user = []
                for u in eligible:
                    recs = ranking[u]
                    if len(recs):
                        per_user.append(float((beta[j] * x[u, recs, j]).mean()))
                        hits = recs[masks["test"][u, recs]]
                        if len(hits):
                            per_hit_user.append(float((beta[j] * x[u, hits, j]).mean()))
                contributions.append(
                    {
                        "member": name,
                        "mean_signed_topk_contribution": float(np.mean(per_user))
                        if per_user
                        else None,
                        "mean_signed_hit_contribution": float(np.mean(per_hit_user))
                        if per_hit_user
                        else None,
                        "users_with_hits": len(per_hit_user),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "coefficients.csv", index=False)
    pd.DataFrame(correlations, index=loaded, columns=loaded).to_csv(
        args.output / "score_correlations.csv"
    )
    # Always overwrite these files to avoid stale diagnostics on partial reruns.
    pd.DataFrame(
        contributions,
        columns=[
            "member",
            "mean_signed_topk_contribution",
            "mean_signed_hit_contribution",
            "users_with_hits",
        ],
    ).to_csv(args.output / "contributions.csv", index=False)
    pd.DataFrame(ablations).reindex(
        columns=[
            "member",
            *[
                f"full_minus_zero_weight_{m}@{args.k}"
                for m in ("hit", "precision", "recall", "f1", "map", "mrr", "ndcg")
            ],
        ]
    ).to_csv(args.output / "fixed_weight_ablation.csv", index=False)
    protocol = {
        "status": "partial" if missing else "coefficient_diagnostics_complete",
        "weights_source": str(path),
        "missing_members": missing,
        "standardisation": "Per user over items outside train, matching pool.features",
        "correlation": "Pearson over standardised scores in union of top-depth outside train; partial union if members missing",
        "depth": depth,
        "depth_source": "CLI"
        if args.depth is not None
        else ("artifact" if "depth" in metadata else "legacy default"),
        "correlation_pairs": int(pool.sum()),
        "fold_count": len(fold_weights) if fold_weights is not None else None,
        "fold_order": "Insertion order of weights, matching pool.MEMBERS when written by prepare_task2",
        "hybrid_scores_verified": verified,
        "contributions_and_ablation": "Available only with all members and a matching Weighted score export; empty otherwise",
        "interpretation": "Coefficients act on standardised scores; not percentages or causal importance. Correlated members can have negative weights.",
        "ablation": "Set one coefficient to zero, keeping remaining weights and original hybrid candidate pool fixed; no retraining or test selection",
        "limitations": "Fold spread unavailable unless saved; rounded legacy coefficients give approximate score contributions; undefined correlations are blank. Other hybrid rules need separate behaviour analysis.",
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"Saved coefficient diagnostics to {args.output}; missing: {missing}; hybrid verified: {verified}"
    )


if __name__ == "__main__":
    main()
