"""Task 2.2 experiment 5: examine accuracy and beyond-accuracy trade-offs."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task22.compare_beyond_accuracy import pareto_mask

BEYOND_NAMES = ("coverage", "novelty", "ild", "serendipity")
COLORS = {"baseline": "#777777", "individual": "#2979b8", "hybrid": "#e07a2d"}
MARKERS = {"baseline": "X", "individual": "o", "hybrid": "s"}


def display_name(name):
    replacements = {
        "TunedCombination": "Feature Combination",
        "TunedAugmentation": "Feature Augmentation",
        "TunedMetalevel": "Meta-Level",
    }
    return replacements.get(name, name.removeprefix("Tuned"))


def correlation_rows(frame, scope, accuracy_metric, beyond_metrics):
    rows = []
    for metric in beyond_metrics:
        accuracy = frame[accuracy_metric]
        beyond = frame[metric]
        rows.append(
            {
                "scope": scope,
                "models": len(frame),
                "accuracy_metric": accuracy_metric,
                "beyond_accuracy_metric": metric,
                "pearson": float(accuracy.corr(beyond)),
                "spearman": float(accuracy.rank().corr(beyond.rank())),
            }
        )
    return rows


def frontier_rows(frame, scope, accuracy_metric, beyond_metrics):
    rows = []
    for metric in beyond_metrics:
        frontier = frame[pareto_mask(frame[[accuracy_metric, metric]].to_numpy())].sort_values(metric, kind="stable")
        for _, row in frontier.iterrows():
            rows.append(
                {
                    "scope": scope,
                    "beyond_accuracy_metric": metric,
                    "model": row["model"],
                    "category": row["category"],
                    accuracy_metric: row[accuracy_metric],
                    metric: row[metric],
                }
            )
    return rows


def save_figure(frame, frontiers, path, accuracy_metric, beyond_metrics):
    cache = Path(tempfile.gettempdir()) / "recommendation-labs-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(top=0.86, hspace=0.28, wspace=0.28)
    non_baseline_frontiers = frontiers[frontiers["scope"] == "non_baselines"]
    labels = {
        beyond_metrics[0]: f"Catalogue coverage@{accuracy_metric.split('@')[1]}",
        beyond_metrics[1]: f"Novelty@{accuracy_metric.split('@')[1]} (bits)",
        beyond_metrics[2]: f"Genre ILD@{accuracy_metric.split('@')[1]}",
        beyond_metrics[3]: f"Serendipity@{accuracy_metric.split('@')[1]}",
    }
    for axis, metric in zip(axes.flat, beyond_metrics, strict=True):
        for category in ("baseline", "individual", "hybrid"):
            group = frame[frame["category"] == category]
            axis.scatter(
                group[accuracy_metric],
                group[metric],
                color=COLORS[category],
                marker=MARKERS[category],
                s=58,
                alpha=0.85,
                label=category.title(),
            )
        frontier = non_baseline_frontiers[non_baseline_frontiers["beyond_accuracy_metric"] == metric].sort_values(accuracy_metric)
        axis.plot(
            frontier[accuracy_metric],
            frontier[metric],
            color="#222222",
            linewidth=1,
            linestyle="--",
            alpha=0.65,
        )
        endpoints = {frontier.iloc[0]["model"], frontier.iloc[-1]["model"]} if len(frontier) else set()
        annotated = endpoints | {"Random", "Pop"}
        for _, row in frame[frame["model"].isin(annotated)].iterrows():
            place_left = row[accuracy_metric] > frame[accuracy_metric].quantile(0.9)
            axis.annotate(
                display_name(row["model"]),
                (row[accuracy_metric], row[metric]),
                xytext=(-4 if place_left else 4, 4),
                textcoords="offset points",
                fontsize=7,
                horizontalalignment="right" if place_left else "left",
                clip_on=True,
            )
        axis.set_xlabel(accuracy_metric.upper())
        axis.set_ylabel(labels[metric])
        axis.margins(x=0.05, y=0.08)
        axis.grid(alpha=0.2)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
    )
    fig.suptitle("Accuracy and beyond-accuracy trade-offs", fontsize=15, y=0.985)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame):
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def run(summary_path, protocol_path, output, k=10):
    accuracy_metric = f"ndcg@{k}"
    beyond_metrics = tuple(f"{metric}@{k}" for metric in BEYOND_NAMES)
    source = pd.read_csv(summary_path)
    required = {"model", "category", accuracy_metric, *beyond_metrics}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if source["model"].duplicated().any():
        raise ValueError("Summary contains duplicate model rows")
    numeric = source[[accuracy_metric, *beyond_metrics]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Trade-off metrics must be finite numbers")
    frame = source[["model", "category"]].copy()
    frame[[accuracy_metric, *beyond_metrics]] = numeric

    scopes = {
        "all_models": frame,
        "non_baselines": frame[frame["category"] != "baseline"],
    }
    correlations = pd.DataFrame([row for scope, subset in scopes.items() for row in correlation_rows(subset, scope, accuracy_metric, beyond_metrics)])
    if np.isinf(correlations[["pearson", "spearman"]].to_numpy()).any():
        raise ValueError("Correlations must be finite or undefined for constant metrics")
    frontiers = pd.DataFrame([row for scope, subset in scopes.items() for row in frontier_rows(subset, scope, accuracy_metric, beyond_metrics)])

    reference = frame.loc[frame[accuracy_metric].idxmax()]
    deltas = frame.copy()
    for metric in (accuracy_metric, *beyond_metrics):
        deltas[f"{metric}_delta_vs_{reference['model']}"] = deltas[metric] - reference[metric]

    source_protocol = json.loads(protocol_path.read_text())
    protocol = {
        "experiment": "Task 2.2 experiment 5: accuracy versus beyond-accuracy trade-offs",
        "k": k,
        "accuracy_metric": accuracy_metric,
        "beyond_accuracy_metrics": list(beyond_metrics),
        "status": source_protocol.get("status", "complete"),
        "source_experiment": source_protocol.get("experiment"),
        "scopes": {name: len(subset) for name, subset in scopes.items()},
        "correlations": "Pearson measures linear association; Spearman measures rank association. Both are descriptive across model configurations; values are undefined when either metric is constant within a scope.",
        "pairwise_frontier": "For NDCG and one beyond-accuracy metric, retain models not dominated on both higher-is-better dimensions.",
        "reference_model": reference["model"],
        "serendipity_dependence": "The implemented serendipity multiplies unexpectedness by test hits, so its association with accuracy is partly structural.",
        "warnings": source_protocol.get("warnings", []),
        "sha256": {
            str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "scripts/task22/analyze_tradeoffs.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    figure_path = output / "tradeoffs.png"
    save_figure(frame, frontiers, figure_path, accuracy_metric, beyond_metrics)
    correlations.to_csv(output / "correlations.csv", index=False)
    frontiers.to_csv(output / "pairwise_frontiers.csv", index=False)
    deltas.to_csv(output / "reference_deltas.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")

    correlation_table = correlations.rename(
        columns={
            "scope": "Scope",
            "beyond_accuracy_metric": "Beyond metric",
            "pearson": "Pearson",
            "spearman": "Spearman",
        }
    )[["Scope", "Beyond metric", "Pearson", "Spearman"]]
    frontier_table = (
        frontiers[frontiers["scope"] == "non_baselines"]
        .groupby("beyond_accuracy_metric")["model"]
        .apply(lambda values: ", ".join(display_name(value) for value in values))
        .rename_axis("Beyond metric")
        .reset_index(name="Non-baseline frontier")
    )
    report = (
        "# Task 2.2 — Experiment 5: accuracy–beyond-accuracy trade-offs\n\n"
        f"{accuracy_metric.upper()} is compared separately with coverage, novelty, genre ILD, and serendipity for all 18 exports and for the 16 non-baseline models. "
        "Correlations are descriptive across configurations, and pairwise Pareto frontiers avoid imposing arbitrary weights.\n\n"
        "## Correlations\n\n"
        + markdown_table(correlation_table)
        + f"\n\n## {accuracy_metric.upper()}/beyond-accuracy frontiers\n\n"
        + markdown_table(frontier_table)
        + "\n\nThe figure `tradeoffs.png` shows every model, with dashed lines joining the non-baseline frontier for each pair.\n"
    )
    (output / "comparison.md").write_text(report)
    return frame, correlations, frontiers, deltas, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/task2/experiment2/summary.csv"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/task2/experiment2/protocol.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/task2/experiment5"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    _, correlations, frontiers, _, _ = run(args.summary, args.protocol, args.output, args.k)
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    counts = frontiers.groupby(["scope", "beyond_accuracy_metric"]).size()
    print("Pairwise frontier sizes:\n" + counts.to_string())
    print(f"Saved trade-off analysis to {args.output}")


if __name__ == "__main__":
    main()
