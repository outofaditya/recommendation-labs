"""Task 2.2 experiment 4: compare beyond-accuracy performance without training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("coverage@10", "novelty@10", "ild@10", "serendipity@10")


def pareto_mask(values):
    """Return rows not dominated when every metric is higher-is-better."""
    frontier = np.ones(len(values), dtype=bool)
    for i, row in enumerate(values):
        others = np.delete(values, i, axis=0)
        frontier[i] = not np.any(np.all(others >= row, axis=1) & np.any(others > row, axis=1))
    return frontier


def markdown_table(frame):
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def run(summary_path, protocol_path, output):
    source = pd.read_csv(summary_path)
    required = {"model", "category", *METRICS}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Summary is missing columns: {sorted(missing)}")
    if source["model"].duplicated().any():
        raise ValueError("Summary contains duplicate model rows")
    values = source[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Beyond-accuracy values must be finite numbers")

    summary = source[["model", "category"]].copy()
    summary[list(METRICS)] = values
    rankings = []
    for metric in METRICS:
        summary[f"{metric}_rank"] = summary[metric].rank(method="min", ascending=False).astype(int)
        category_best = summary.groupby("category")[metric].transform("max")
        summary[f"{metric}_category_leader"] = summary[metric] == category_best
        for _, row in summary.iterrows():
            rankings.append(
                {
                    "metric": metric,
                    "model": row["model"],
                    "category": row["category"],
                    "value": row[metric],
                    "global_rank": row[f"{metric}_rank"],
                    "category_leader": row[f"{metric}_category_leader"],
                }
            )
    rankings = pd.DataFrame(rankings).sort_values(["metric", "global_rank", "model"], kind="stable")
    summary["pareto_frontier"] = pareto_mask(summary[list(METRICS)].to_numpy())
    leaders = rankings[rankings["category_leader"]].copy()
    frontier = summary[summary["pareto_frontier"]][["model", "category", *METRICS]].copy()

    source_protocol = json.loads(protocol_path.read_text())
    protocol = {
        "experiment": "Task 2.2 experiment 4: beyond-accuracy comparison",
        "status": source_protocol.get("status", "complete"),
        "source_experiment": source_protocol.get("experiment"),
        "models": summary["model"].tolist(),
        "metrics": list(METRICS),
        "direction": "Higher is better for all four metrics",
        "ranking": "Descending global rank with minimum rank for ties",
        "category_leaders": "All tied maxima within baseline, individual, and hybrid categories",
        "pareto_frontier": "A model is retained when no other model is at least as high on all four metrics and strictly higher on at least one",
        "no_composite_score": "Metrics remain separate because their scales and meanings differ; no arbitrary weights are imposed",
        "warnings": source_protocol.get("warnings", []),
        "sha256": {
            str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "scripts/compare_beyond_accuracy.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }

    leader_table = leaders.rename(
        columns={
            "metric": "Metric",
            "category": "Category",
            "model": "Leader",
            "value": "Value",
            "global_rank": "Global rank",
        }
    )[["Metric", "Category", "Leader", "Value", "Global rank"]]
    report = (
        "# Task 2.2 — Experiment 4: beyond-accuracy performance\n\n"
        "All 18 fixed local exports are ranked separately on Coverage@10, Novelty@10, genre-Jaccard ILD@10, and relevance-weighted Serendipity@10. "
        "No model is trained or selected, and no composite score is formed.\n\n"
        "## Category leaders\n\n"
        + markdown_table(leader_table)
        + "\n\n## Pareto frontier\n\n"
        + markdown_table(frontier)
        + "\n\nA frontier model is not exceeded by another model on all four beyond-accuracy dimensions. "
        "Membership does not imply a preferred model because the four metrics represent different goals.\n"
    )

    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    rankings.to_csv(output / "metric_rankings.csv", index=False)
    leaders.to_csv(output / "category_leaders.csv", index=False)
    frontier.to_csv(output / "pareto_frontier.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (output / "comparison.md").write_text(report)
    return summary, rankings, leaders, frontier, protocol


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
    parser.add_argument("--output", type=Path, default=Path("results/task2/experiment4"))
    args = parser.parse_args()
    summary, _, leaders, frontier, _ = run(args.summary, args.protocol, args.output)
    print(summary[["model", "category", *METRICS]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Category leaders: {len(leaders)}; Pareto-frontier models: {len(frontier)}")
    print(f"Saved beyond-accuracy comparison to {args.output}")


if __name__ == "__main__":
    main()
