"""Task 2.2 experiment 3: quantify gains over Random and Pop without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINES = ("Random", "Pop")
IDENTIFIERS = ("model", "category")


def markdown_table(frame):
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{value:.2f}" if isinstance(value, float) else str(value) for value in row) + " |")
    return "\n".join(lines)


def run(summary_path, protocol_path, output):
    summary = pd.read_csv(summary_path)
    missing_columns = set(IDENTIFIERS) - set(summary.columns)
    if missing_columns:
        raise ValueError(f"Summary is missing columns: {sorted(missing_columns)}")
    if summary["model"].duplicated().any():
        raise ValueError("Summary contains duplicate model rows")

    metrics = [column for column in summary.columns if column not in IDENTIFIERS]
    if not metrics:
        raise ValueError("Summary contains no metric columns")
    numeric = summary[metrics].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("All metric values must be finite numbers")
    summary[metrics] = numeric

    indexed = summary.set_index("model")
    missing_baselines = [name for name in BASELINES if name not in indexed.index]
    if missing_baselines:
        raise ValueError(f"Summary is missing baselines: {missing_baselines}")
    if any(indexed.loc[name, "category"] != "baseline" for name in BASELINES):
        raise ValueError("Random and Pop must be labelled as baselines")

    targets = summary[~summary["model"].isin(BASELINES)]
    rows = []
    for _, target in targets.iterrows():
        for baseline_name in BASELINES:
            baseline = indexed.loc[baseline_name]
            for metric in metrics:
                baseline_value = float(baseline[metric])
                difference = float(target[metric] - baseline_value)
                rows.append(
                    {
                        "model": target["model"],
                        "category": target["category"],
                        "baseline": baseline_name,
                        "metric": metric,
                        "model_value": float(target[metric]),
                        "baseline_value": baseline_value,
                        "absolute_difference": difference,
                        "relative_change": difference / baseline_value if baseline_value != 0 else None,
                        "relative_change_percent": 100 * difference / baseline_value if baseline_value != 0 else None,
                    }
                )
    comparisons = pd.DataFrame(rows)

    primary = comparisons[comparisons["metric"] == "ndcg@10"].pivot(
        index=["model", "category"],
        columns="baseline",
        values=["absolute_difference", "relative_change_percent"],
    )
    primary.columns = [f"ndcg_{measure}_vs_{baseline.lower()}" for measure, baseline in primary.columns]
    primary = primary.reset_index()

    protocol_source = json.loads(protocol_path.read_text())
    warnings = list(protocol_source.get("warnings", []))
    warnings.append("Relative changes can be very large when a baseline is near zero; interpret them with the absolute difference.")
    protocol = {
        "experiment": "Task 2.2 experiment 3: improvement over naive baselines",
        "status": "complete_with_baseline_warning" if warnings else "complete",
        "source_experiment": protocol_source.get("experiment"),
        "models_compared": targets["model"].tolist(),
        "baselines": list(BASELINES),
        "metrics": metrics,
        "comparison_count": len(comparisons),
        "absolute_difference": "model metric - baseline metric",
        "relative_change": "(model metric - baseline metric) / baseline metric",
        "direction": "All included metrics are higher-is-better; a negative change means the model scores below the baseline on that metric.",
        "interpretation": "Percentage change measures scale relative to a baseline, not statistical significance or causal improvement.",
        "warnings": warnings,
        "sha256": {
            str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "scripts/quantify_baselines.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }

    report_table = primary.rename(
        columns={
            "model": "Model",
            "category": "Category",
            "ndcg_absolute_difference_vs_random": "NDCG Δ vs Random",
            "ndcg_relative_change_percent_vs_random": "NDCG % vs Random",
            "ndcg_absolute_difference_vs_pop": "NDCG Δ vs Pop",
            "ndcg_relative_change_percent_vs_pop": "NDCG % vs Pop",
        }
    )
    report = (
        "# Task 2.2 — Experiment 3: improvement over naïve baselines\n\n"
        "Every tuned individual and hybrid is compared with the saved Random and Pop exports across all 11 Task 2.1 metrics. "
        "Absolute difference is model minus baseline; relative change divides that difference by the baseline. No model is trained or selected here.\n\n"
        + markdown_table(report_table)
        + "\n\nThe long-form CSV contains both comparisons for all accuracy and beyond-accuracy metrics. "
        "Percentage changes against near-zero Random values are mechanically large and must be read with absolute differences. "
        "The saved Random export also contains only four distinct score rows for 943 users.\n"
    )

    output.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output / "baseline_comparison.csv", index=False)
    primary.to_csv(output / "ndcg_summary.csv", index=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    (output / "comparison.md").write_text(report)
    return comparisons, primary, protocol


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
    parser.add_argument("--output", type=Path, default=Path("results/task2/experiment3"))
    args = parser.parse_args()
    comparisons, primary, protocol = run(args.summary, args.protocol, args.output)
    print(primary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Saved {len(comparisons)} baseline comparisons across {len(protocol['metrics'])} metrics to {args.output}")


if __name__ == "__main__":
    main()
