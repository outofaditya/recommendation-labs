"""Run all Task 2.2 experiments on the Task 1 repeated-seed protocol."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from scripts.task22.task22_pipeline import run_pipeline

SEEDS = (2020, 2021, 2022, 2023, 2024)
KS = (1, 3, 5, 10, 20)
LOWER_IS_BETTER = {"mae", "mse"}
METRIC_NAMES = ("ndcg", "recall", "precision", "mrr", "hit", "map", "f1", "coverage", "novelty", "ild", "serendipity")
REPO = Path(__file__).resolve().parents[2]


def _run(command, cwd, env):
    print(f"[{cwd.name}] RUN {' '.join(map(str, command))}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _prepare_workdir(workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "parameters", "scripts", "source"):
        link = workdir / name
        target = REPO / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected symlink target at {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing path: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    (workdir / "results").mkdir(exist_ok=True)


def run_seed(seed, workdir, ks, workers, reference_metrics):
    marker = workdir / "seed_complete.json"
    if marker.exists():
        print(f"Seed {seed} already complete; skipping training and evaluation.", flush=True)
        return
    _prepare_workdir(workdir)
    cache = workdir / "results/task2/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "scripts"), str(REPO), env.get("PYTHONPATH", "")))
    env["MPLCONFIGDIR"] = str(cache)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["LOKY_MAX_CPU_COUNT"] = "1"
    python = sys.executable

    # Match Task 1 inference: hold each tuned individual configuration fixed,
    # but refit on that seed's independently drawn RecBole split.
    # Run the non-FISM models as a parallel batch first.  FISM is deliberately
    # isolated because its long single-model training time otherwise makes the
    # whole seed appear stuck and prevents already-finished models from being
    # inspected or reused.
    non_fism = [name for name in ("NGCF", "LightGCN", "BPR", "NeuMF", "SLIMElastic", "ItemKNN", "UserKNN", "EASE", "Pop", "Random")]
    _run([python, "-m", "scripts.models", "--tuned", "--seed", str(seed), "--workers", str(workers), "--models", *non_fism], workdir, env)
    _run([python, "-m", "scripts.models", "--tuned", "--seed", str(seed), "--workers", "1", "--models", "FISM"], workdir, env)
    _run([python, "-m", "scripts.task22.build_task22_repeat_hybrids", "--reference-metrics", str(reference_metrics)], workdir, env)

    task_root = Path("results/task2/repeated_task22")
    run_pipeline(
        results=Path("results"),
        output_root=task_root,
        ks=ks,
        items=Path("data/ml-100k/ml-100k.item"),
        interactions=Path("data/ml-100k/ml-100k.inter"),
    )
    for k in ks:
        print(f"Seed {seed}: all Task 2.2 experiments complete at K={k}.", flush=True)

    marker.write_text(json.dumps({"seed": seed, "ks": list(ks), "complete": True}, indent=2) + "\n")


def _read_per_seed(root, seeds, ks):
    metric_rows, baseline_rows, rank_rows, pareto_rows, correlation_rows, frontier_rows = [], [], [], [], [], []
    for seed in seeds:
        task_root = root / "runs" / f"seed_{seed}" / "results/task2/repeated_task22"
        if not (root / "runs" / f"seed_{seed}" / "seed_complete.json").exists():
            raise RuntimeError(f"Seed {seed} is incomplete; aggregate only after all requested seeds finish")
        for experiment in ("experiment1", "experiment2"):
            for k in ks:
                frame = pd.read_csv(task_root / experiment / f"k_{k}/summary.csv")
                if "category" not in frame:
                    frame["category"] = np.where(frame.model.isin(["Random", "Pop"]), "baseline", "individual")
                for column in (c for c in frame.columns if c not in ("model", "category")):
                    metric = column.rsplit("@", 1)[0] if "@" in column else column
                    if metric in LOWER_IS_BETTER:
                        if k != 10:
                            continue
                        k_value = "all"
                    else:
                        k_value = k
                    for row in frame[["model", "category", column]].itertuples(index=False, name=None):
                        metric_rows.append(
                            {
                                "seed": seed,
                                "experiment": experiment,
                                "k": k_value,
                                "model": row[0],
                                "category": row[1],
                                "metric": metric,
                                "value": float(row[2]),
                            }
                        )

                e3 = pd.read_csv(task_root / "experiment3" / f"k_{k}/baseline_comparison.csv")
                for row in e3.itertuples(index=False):
                    if row.metric in LOWER_IS_BETTER and k != 10:
                        continue
                    baseline_rows.append(
                        {
                            "seed": seed,
                            "k": "all" if row.metric in LOWER_IS_BETTER else k,
                            "model": row.model,
                            "category": row.category,
                            "baseline": row.baseline,
                            "metric": row.metric,
                            "absolute_difference": row.absolute_difference,
                            "relative_change_percent": row.relative_change_percent,
                        }
                    )

                e4 = pd.read_csv(task_root / "experiment4" / f"k_{k}/metric_rankings.csv")
                for row in e4.itertuples(index=False):
                    rank_rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "metric": row.metric,
                            "model": row.model,
                            "category": row.category,
                            "global_rank": row.global_rank,
                            "category_leader": bool(row.category_leader),
                        }
                    )
                e4_summary = pd.read_csv(task_root / "experiment4" / f"k_{k}/summary.csv")
                for row in e4_summary[["model", "pareto_frontier"]].itertuples(index=False, name=None):
                    pareto_rows.append({"seed": seed, "k": k, "model": row[0], "pareto_frontier": bool(row[1])})

                e5 = pd.read_csv(task_root / "experiment5" / f"k_{k}/correlations.csv")
                for row in e5.itertuples(index=False):
                    correlation_rows.append(
                        {
                            "seed": seed,
                            "k": k,
                            "scope": row.scope,
                            "beyond_accuracy_metric": row.beyond_accuracy_metric,
                            "pearson": row.pearson,
                            "spearman": row.spearman,
                        }
                    )
                e5_frontier = pd.read_csv(task_root / "experiment5" / f"k_{k}/pairwise_frontiers.csv")
                for row in e5_frontier.itertuples(index=False):
                    frontier_rows.append(
                        {"seed": seed, "k": k, "scope": row.scope, "beyond_accuracy_metric": row.beyond_accuracy_metric, "model": row.model}
                    )
    return tuple(pd.DataFrame(rows) for rows in (metric_rows, baseline_rows, rank_rows, pareto_rows, correlation_rows, frontier_rows))


def _ci_summary(frame, groups, values):
    rows = []
    for keys, part in frame.groupby(groups, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys, strict=True))
        for column in values:
            data = pd.to_numeric(part[column], errors="coerce").dropna().to_numpy(dtype=float)
            n = len(data)
            mean = float(np.mean(data)) if n else np.nan
            std = float(np.std(data, ddof=1)) if n > 1 else np.nan
            half = float(student_t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else np.nan
            rows.append(
                base
                | {
                    "measure": column,
                    "n_seeds": n,
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - half if n > 1 else np.nan,
                    "ci95_high": mean + half if n > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def aggregate(root, seeds, ks):
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    metrics, baselines, ranks, pareto, correlations, frontiers = _read_per_seed(root, seeds, ks)
    metrics.to_csv(output / "per_seed_metrics.csv", index=False)
    metric_summary = _ci_summary(metrics, ["experiment", "k", "model", "category", "metric"], ["value"])
    metric_summary.to_csv(output / "metric_summary.csv", index=False)
    baselines.to_csv(output / "per_seed_baseline_differences.csv", index=False)
    _ci_summary(baselines, ["k", "model", "category", "baseline", "metric"], ["absolute_difference", "relative_change_percent"]).to_csv(
        output / "baseline_difference_summary.csv", index=False
    )
    ranks.to_csv(output / "per_seed_beyond_accuracy_ranks.csv", index=False)
    rank_summary = _ci_summary(ranks, ["k", "metric", "model", "category"], ["global_rank"])
    leader_summary = _ci_summary(ranks, ["k", "metric", "model", "category"], ["category_leader"])
    leader_summary.drop(columns="measure", inplace=True)
    leader_summary.rename(
        columns={
            "n_seeds": "leader_n_seeds",
            "mean": "category_leader_frequency",
            "std": "category_leader_std",
            "ci95_low": "category_leader_ci95_low",
            "ci95_high": "category_leader_ci95_high",
        },
        inplace=True,
    )
    rank_summary = rank_summary.merge(leader_summary, on=["k", "metric", "model", "category"], validate="one_to_one")
    rank_summary.to_csv(output / "beyond_accuracy_rank_summary.csv", index=False)
    pareto.to_csv(output / "per_seed_pareto.csv", index=False)
    pareto_summary = _ci_summary(pareto, ["k", "model"], ["pareto_frontier"])
    pareto_summary.drop(columns="measure", inplace=True)
    pareto_summary.rename(columns={"mean": "pareto_frequency", "std": "pareto_std"}, inplace=True)
    pareto_summary.to_csv(output / "pareto_frequency.csv", index=False)
    correlations.to_csv(output / "per_seed_tradeoff_correlations.csv", index=False)
    _ci_summary(correlations, ["k", "scope", "beyond_accuracy_metric"], ["pearson", "spearman"]).to_csv(
        output / "tradeoff_correlation_summary.csv", index=False
    )
    frontiers.to_csv(output / "per_seed_pairwise_frontiers.csv", index=False)
    model_categories = metrics[metrics.experiment == "experiment2"][["model", "category"]].drop_duplicates()
    frontier_set = set(zip(frontiers.seed, frontiers.k, frontiers.scope, frontiers.beyond_accuracy_metric, frontiers.model))
    flag_rows = []
    for seed in seeds:
        for k in ks:
            for scope in ("all_models", "non_baselines"):
                candidates = model_categories if scope == "all_models" else model_categories[model_categories.category != "baseline"]
                for metric in (f"coverage@{k}", f"novelty@{k}", f"ild@{k}", f"serendipity@{k}"):
                    for model in candidates.model:
                        flag_rows.append(
                            {
                                "seed": seed,
                                "k": k,
                                "scope": scope,
                                "beyond_accuracy_metric": metric,
                                "model": model,
                                "on_frontier": (seed, k, scope, metric, model) in frontier_set,
                            }
                        )
    frontier_flags = pd.DataFrame(flag_rows)
    frontier_flags.to_csv(output / "per_seed_pairwise_frontier_flags.csv", index=False)
    frontier_frequency = _ci_summary(frontier_flags, ["k", "scope", "beyond_accuracy_metric", "model"], ["on_frontier"])
    frontier_frequency.drop(columns="measure", inplace=True)
    frontier_frequency.rename(columns={"mean": "frontier_frequency", "std": "frontier_std"}, inplace=True)
    frontier_frequency.to_csv(output / "pairwise_frontier_frequency.csv", index=False)

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    e2 = metric_summary[(metric_summary.experiment == "experiment2") & (metric_summary.k != "all")]
    models = e2.model.drop_duplicates().tolist()
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(models)))
    colors = dict(zip(models, palette, strict=True))
    for metric in METRIC_NAMES:
        frame = e2[e2.metric == metric].copy()
        frame["k"] = frame.k.astype(int)
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for model, group in frame.groupby("model", sort=False):
            group = group.sort_values("k")
            x = group.k.to_numpy()
            ax.plot(x, group["mean"], color=colors[model], marker="o", linewidth=1.5, label=model)
            ax.fill_between(x, group.ci95_low, group.ci95_high, color=colors[model], alpha=0.09)
        ax.set(title=f"{metric.upper()} by K across {len(seeds)} seeds", xlabel="K", ylabel=metric.upper(), xticks=list(ks))
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{metric}_by_k.png", dpi=180)
        plt.close(fig)

    (output / "README.md").write_text(
        f"# Task 2.2 repeated-seed evaluation\n\nSeeds: {', '.join(map(str, seeds))}. "
        f"Each seed refits the fixed tuned Task 1 individual configurations, rebuilds all seven hybrids, and reruns Task 2.2 experiments 1–5 at K={', '.join(map(str, ks))}. "
        "Tables report mean, standard deviation, and 95% t intervals across independent seeds. Experiments 3–5 are derived from each seed's Experiment 2 outputs.\n\n"
        "See `metric_summary.csv` and `baseline_difference_summary.csv` for interval summaries; `beyond_accuracy_rank_summary.csv`, `pareto_frequency.csv`, "
        "and `tradeoff_correlation_summary.csv` aggregate Experiments 4–5. `per_seed_*.csv` retain run-level values.\n"
    )
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "experiment": "Task 2.2 repeated-seed evaluation",
                "seeds": list(seeds),
                "k_values": list(ks),
                "individual_models": "All nine Task 1 tuned individuals plus Random and Pop, retrained with fixed tuned configurations; each seed controls split generation and model initialization, matching scripts/seeds.py.",
                "hybrids": "Seven tuned Task 1 designs rebuilt from each seed's individual scores; selected hyperparameters remain fixed and learned hybrid weights/models are refit using that seed's validation data.",
                "experiments": {
                    "1": "Individual model comparison, ranking and beyond-accuracy metrics for each K; MAE/MSE once per seed.",
                    "2": "All 18 individual, baseline, and hybrid models, same metric/K protocol.",
                    "3": "Per-seed baseline differences derived from Experiment 2.",
                    "4": "Per-seed beyond-accuracy ranks, category leaders, and four-dimensional Pareto frontier at each K.",
                    "5": "Per-seed NDCG/beyond-accuracy correlations and two-dimensional frontiers at each K.",
                },
                "uncertainty": "Mean, sample standard deviation, and two-sided 95% Student t intervals across seeds; each seed is a paired split for model and baseline contrasts.",
                "known_limitations": [
                    "Five-seed intervals are based on a small number of runs.",
                    "Hybrid hyperparameter settings are fixed from the existing Task 1 selection; this measures their repeatability, not variability from retuning.",
                    "The separate user-bootstrap experiment estimates test-user sampling uncertainty on one fixed split.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--workers", type=int, default=1, help="Parallel model workers within a seed; keep at 1 to limit overnight memory use")
    parser.add_argument("--output", type=Path, default=Path("results/task2/repeated_seeds"))
    parser.add_argument("--reference-hybrid-metrics", type=Path, default=Path("results/hybrid/metrics"))
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be non-empty and unique")
    if not args.ks or any(k < 1 or k > 20 for k in args.ks):
        parser.error("K values must be between 1 and 20")
    root = (REPO / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    reference = (
        (REPO / args.reference_hybrid_metrics).resolve()
        if not args.reference_hybrid_metrics.is_absolute()
        else args.reference_hybrid_metrics.resolve()
    )
    if not args.aggregate_only:
        for seed in args.seeds:
            run_seed(seed, root / "runs" / f"seed_{seed}", args.ks, args.workers, reference)
    aggregate_root = aggregate(root, args.seeds, args.ks)
    print(f"Completed repeated Task 2.2 evaluation. Aggregate outputs: {aggregate_root}", flush=True)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main()
