"""Validate saved experiment results and generate Task 2 in report/notes.md."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.task2.multiseed import TABLES, digest, valid_marker, write_json
from scripts.task2.analyze import verify_source
from scripts.task22.task22_seeds import _ci_summary


def select(frame, **filters):
    chosen = frame
    for key, value in filters.items():
        chosen = chosen[chosen[key].isna() if value is None else chosen[key] == value]
    if len(chosen) != 1:
        raise ValueError(f"Expected one row for {filters}, got {len(chosen)}")
    return chosen.iloc[0]


def interval(row, digits=4, scale=1, signed=False):
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return f"{row['mean'] * scale:{spec}} [{row['ci95_low'] * scale:{spec}}, {row['ci95_high'] * scale:{spec}}]"


def paired(frame, left, right, metric="ndcg", k=10):
    subset = frame[(frame.experiment == "experiment2") & (frame.k == k) & (frame.metric == metric)]
    direct = subset[(subset.model == left) & (subset.reference == right)]
    if len(direct) == 1:
        return direct.iloc[0]
    reverse = select(subset, model=right, reference=left).copy()
    mean, low, high = reverse["mean"], reverse.ci95_low, reverse.ci95_high
    reverse["mean"], reverse["ci95_low"], reverse["ci95_high"] = -mean, -high, -low
    return reverse


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def compare_means(old, new, keys, columns):
    merged = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(merged) != len(new):
        raise ValueError("Old/new rows do not align")
    return {column: float((merged[column + "_new"] - merged[column + "_old"]).abs().max()) for column in columns}


def audit(root):
    protocol = json.loads((root / "reanalysis_protocol.json").read_text())
    sig = digest(root / "reanalysis_protocol.json")
    manifests = [(root / "aggregate_complete.json", root, sig), (root / "stages/bootstrap.json", root, sig)]
    for seed in protocol["seeds"]:
        work = root / "runs" / f"seed_{seed}"
        manifests += [(work / "stages" / f"{stage}.json", work, f"{sig}:{seed}") for stage in ("task22", "task23", "task24")]
    for marker, base, signature in manifests:
        if not valid_marker(marker, base, signature):
            raise ValueError(f"Incomplete or changed outputs: {marker}")
    original, _ = verify_source(Path(protocol["source_root"]))
    if digest(Path(protocol["source_root"]) / "run_protocol.json") != protocol["source_protocol_sha256"]:
        raise ValueError("Original protocol identity changed")
    for relative, expected in protocol["analysis_inputs"].items():
        snapshot = root / "analysis_snapshot" / relative
        path = snapshot if snapshot.is_file() else Path(relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Executed analysis input unavailable or changed: {relative}")
    aggregate = root / "aggregate"
    metrics = pd.read_csv(aggregate / "per_seed_metrics.csv")
    if metrics.duplicated(["seed", "experiment", "k", "model", "metric"]).any():
        raise ValueError("Duplicate per-seed metric keys")
    if set(metrics.metric) & {"mae", "mse", "rmse"} or not np.isfinite(metrics.value).all():
        raise ValueError("Invalid ranking metric values or retained rating errors")
    expected_rows = 29 * 11 * len(protocol["ks"]) * len(protocol["seeds"])
    if len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows} per-seed metrics, got {len(metrics)}")
    checked_summary_rows = 0
    sources = [("per_seed_metrics.csv", "metric_summary.csv", ["experiment", "k", "model", "category", "metric"])]
    sources += [
        ("per_seed_paired_model_differences.csv", "paired_model_difference_summary.csv", ["experiment", "k", "metric", "model", "reference"]),
        ("per_seed_baseline_differences.csv", "baseline_difference_summary.csv", ["k", "model", "category", "baseline", "metric"]),
        ("per_seed_tradeoff_correlations.csv", "tradeoff_correlation_summary.csv", ["k", "scope", "beyond_accuracy_metric"]),
    ]
    sources += [(f"{task}/{name}_per_seed.csv", f"{task}/{name}_summary.csv", keys) for task, names in TABLES.items() for name, keys in names.items()]
    for per_seed, summary_name, keys in sources:
        data, summary = pd.read_csv(aggregate / per_seed), pd.read_csv(aggregate / summary_name)
        expected = _ci_summary(data, keys, summary.measure.unique())
        merged = summary.merge(expected, on=[*keys, "measure"], suffixes=("_saved", "_checked"), validate="one_to_one")
        if len(merged) != len(summary):
            raise ValueError(f"Summary rows do not align: {summary_name}")
        for column in ("n_seeds", "mean", "std", "ci95_low", "ci95_high"):
            if not np.allclose(merged[column + "_saved"], merged[column + "_checked"], atol=1e-11, rtol=1e-11, equal_nan=True):
                raise ValueError(f"Seed-level aggregation mismatch: {summary_name}/{column}")
        checked_summary_rows += len(summary)
    boot = root / "bootstrap" / f"seed_{protocol['bootstrap']['seed']}"
    point, pairs = pd.read_csv(boot / "summary.csv"), pd.read_csv(boot / "paired_differences.csv")
    seed_point = metrics[(metrics.seed == protocol["bootstrap"]["seed"]) & (metrics.experiment == "experiment2")]
    aligned = point.merge(seed_point, on=["model", "k", "metric"], suffixes=("_bootstrap", "_pipeline"), validate="one_to_one")
    if len(aligned) != len(point) or not np.allclose(aligned.value_bootstrap, aligned.value_pipeline, atol=1e-8, rtol=1e-8):
        raise ValueError("Bootstrap point estimates disagree with the core evaluation")
    coverage = point.metric == "coverage"
    if len(point) != 990 or len(pairs) != 7650:
        raise ValueError("Incomplete fixed-export bootstrap")
    if not point.loc[coverage, ["ci95_low", "ci95_high"]].isna().all().all():
        raise ValueError("Coverage must not have an ordinary user-bootstrap CI")
    if not np.isfinite(point.loc[~coverage, ["value", "ci95_low", "ci95_high"]]).all().all():
        raise ValueError("Missing marginal bootstrap intervals")
    if not (pairs.bootstrap_reps == protocol["bootstrap"]["reps"]).all():
        raise ValueError("Bootstrap replicate count mismatch")
    if (pairs.ci95_low > pairs.ci95_high).any() or not np.isfinite(pairs[["difference", "ci95_low", "ci95_high"]]).all().all():
        raise ValueError("Invalid paired bootstrap intervals")
    base = point[["model", "k", "metric", "value"]]
    contrasts = pairs.merge(base, on=["model", "k", "metric"], validate="many_to_one")
    contrasts = contrasts.merge(
        base.rename(columns={"model": "reference", "value": "reference_value"}), on=["reference", "k", "metric"], validate="many_to_one"
    )
    if not np.allclose(contrasts.difference, contrasts.value - contrasts.reference_value, atol=1e-12):
        raise ValueError("Bootstrap paired contrast has the wrong model direction")
    old_root = Path(protocol["source_root"]) / "aggregate"
    old_metrics = pd.read_csv(old_root / "per_seed_metrics.csv")
    old_metrics["k"] = old_metrics.k.astype(str)
    new_metrics = metrics.copy()
    new_metrics["k"] = new_metrics.k.astype(str)
    metric_change = compare_means(old_metrics, new_metrics, ["seed", "experiment", "k", "model", "category", "metric"], ["value"])["value"]
    candidate_old = pd.read_csv(old_root / "task24/candidate_pipeline_per_seed.csv")
    candidate_new = pd.read_csv(aggregate / "task24/candidate_pipeline_per_seed.csv")
    candidates = compare_means(
        candidate_old,
        candidate_new,
        ["seed", "candidate_generator", "reranker", "depth"],
        ["candidate_recall", "candidate_catalogue_coverage", "ranker_available_fraction", "ndcg@10"],
    )
    report = {
        "complete": True,
        "seed_stage_manifests": 15,
        "bootstrap_replicates": protocol["bootstrap"]["reps"],
        "source_seeds": original["seeds"],
        "per_seed_metric_rows": len(metrics),
        "checked_summary_rows": checked_summary_rows,
        "bootstrap_point_rows": len(point),
        "bootstrap_paired_rows": len(pairs),
        "max_ranking_beyond_metric_change": metric_change,
        "candidate_pipeline_max_absolute_change": candidates,
        "source_protocol_sha256": protocol["source_protocol_sha256"],
        "analysis_protocol_sha256": sig,
    }
    write_json(root / "report_audit.json", report)
    return protocol, report


def build(root, notes):
    protocol, checks = audit(root)
    aggregate = root / "aggregate"
    read = lambda name: pd.read_csv(aggregate / name)
    metrics, pairs = read("metric_summary.csv"), read("paired_model_difference_summary.csv")
    metric = lambda model, name="ndcg": select(metrics, experiment="experiment2", k=10, model=model, metric=name)
    section = [
        "## Task 2 — Evaluation of Effectiveness",
        "### Experimental design",
        (
            "**Objectives.** Task 2 tests whether combining recommenders improves ranking quality, which members and side features contribute to that performance, "
            "and whether candidate retrieval or final ranking limits recommendations of long-tail movies. "
            "We compare Random and Pop, nine tuned individual models and seven tuned hybrid architectures: Weighted, Combination, Mixed, Cascade, Switching, Augmentation and Metalevel."
        ),
        (
            "**Data and repetitions.** MovieLens 100K contains 943 users and 1,682 movies. Seeds 2020–2024 each define a distinct split and model-training run. "
            "Each split contains 80,808 training, 9,596 validation and 9,596 test interactions, with disjoint partitions. "
            "All models within a seed share the same split and aligned user/item IDs. Any observed rating is treated as binary relevance. "
            "Training interactions fit the individual models; validation selects settings and fits the hybrids. "
            "Task 2 evaluates the resulting 55 individual/baseline and 35 final hybrid score tables on CPU; fitting the contribution ablations is separate from training the final models."
        ),
        (
            "**Ranking and metrics.** Test rankings hide training and validation interactions and are evaluated at K=1,3,5,10,20 for users with test interactions "
            "(all 943 users in these splits). The seven ranking measures are NDCG, Recall, Precision, MRR, Hit, MAP and per-user F1. "
            "Catalogue coverage measures the fraction of movies recommended at least once; novelty is mean popularity self-information; "
            "intra-list diversity (ILD) is mean pairwise genre Jaccard distance; serendipity weights relevant recommendations by genre distance from the user's history. "
            "MAE, MSE and RMSE are outside this implicit-ranking task because the scores are not calibrated 1–5 rating predictions."
        ),
        (
            "**Uncertainty.** Seed means use two-sided 95% Student-t intervals (four degrees of freedom); model differences are paired within seed. "
            "A complementary bootstrap conditions on seed 2020 and uses 10,000 shared user resamples across models and K. "
            "It estimates intervals for additive user-level metrics and paired differences. Catalogue coverage uses seed-level intervals only, "
            "because resampling users changes the unique-item union. These two procedures describe different sources of uncertainty."
        ),
        table(
            ["Study", "Design", "Evaluation scope"],
            [
                [
                    "2.2: Experiments 1–2",
                    "Evaluate 11 individual/baseline and seven hybrid models with 11 metrics at five K values",
                    "Test; matched splits within seed",
                ],
                ["2.2: Experiment 3", "Compare models with Random and Pop using paired differences", "Test; seed-level contrasts"],
                [
                    "2.2: Experiments 4–5",
                    "Compare beyond-accuracy ranks, metric correlations and Pareto frontiers; bootstrap user-level differences",
                    "Test; seed means and conditional user bootstrap",
                ],
                [
                    "2.3: Contributions",
                    "Inspect selected Weighted coefficients, regularization, member removal, activity groups and Combination feature blocks",
                    "Five-fold validation-user CV for Weighted/Combination; validation diagnostics for Mixed/Switching/Cascade",
                ],
                [
                    "2.4: Behaviour and long tail",
                    "Compare user routing, 8 × 6 × 5 candidate pipelines, popularity exposure and five tail-retrieval strategies",
                    "Test; exploratory pipeline sweep and descriptive oracle",
                ],
            ],
        ),
        (
            "**Experiment records.** [Complete results](../results/task2/reanalysis_cpu/aggregate/), "
            "[protocol and provenance](../results/task2/reanalysis_cpu/reanalysis_protocol.json), "
            "[data and aggregation checks](../results/task2/reanalysis_cpu/report_audit.json), "
            "[all seed-level metrics](../results/task2/reanalysis_cpu/aggregate/per_seed_metrics.csv) and "
            "[all metric means, SDs and intervals](../results/task2/reanalysis_cpu/aggregate/metric_summary.csv) accompany the tables below. "
            "Source snapshots and output hashes preserve the executed experiment. Commands are in the [README command table](../README.md#simulate)."
        ),
        "### Experimental results",
        "#### Task 2.2 — Effectiveness evaluation",
        "##### Experiments 1–2: ranking quality",
        "The table reports all 18 models at K=10. NDCG includes its across-seed interval; the other ranking columns show seed means. Full results at every K and intervals for every metric are linked above.",
    ]
    order = [
        "Random",
        "Pop",
        "ItemKNN",
        "UserKNN",
        "BPR",
        "NeuMF",
        "FISM",
        "LightGCN",
        "NGCF",
        "EASE",
        "SLIMElastic",
        "TunedWeighted",
        "TunedCombination",
        "TunedMixed",
        "TunedCascade",
        "TunedSwitching",
        "TunedAugmentation",
        "TunedMetalevel",
    ]
    section.append(
        table(
            ["Model", "Type", "NDCG@10 [95% CI]", "Recall", "Precision", "MRR", "Hit", "MAP", "F1"],
            [
                [
                    name,
                    metric(name).category,
                    interval(metric(name)),
                    *[f"{metric(name, m)['mean']:.4f}" for m in ["recall", "precision", "mrr", "hit", "map", "f1"]],
                ]
                for name in order
            ],
        )
    )
    section += [
        (
            f"Weighted and Combination have the highest mean NDCG among the hybrids. Their paired NDCG gains over SLIMElastic are "
            f"{interval(paired(pairs, 'TunedWeighted', 'SLIMElastic'), 5, signed=True)} and "
            f"{interval(paired(pairs, 'TunedCombination', 'SLIMElastic'), 5, signed=True)}. "
            f"Weighted minus Combination is {interval(paired(pairs, 'TunedWeighted', 'TunedCombination'), 5, signed=True)}; this interval includes zero."
        ),
        "![NDCG versus K, seed means and 95% t intervals](plots/ndcg_by_k.png)",
        "##### Experiment 3: baseline contrasts",
        table(
            ["Model", "NDCG@10 minus Random [95% CI]", "NDCG@10 minus Pop [95% CI]"],
            [
                [name, interval(paired(pairs, name, "Random"), 5, signed=True), interval(paired(pairs, name, "Pop"), 5, signed=True)]
                for name in order[2:]
            ],
        ),
        "Every trained individual model and hybrid outperforms both baselines on NDCG@10, with positive paired intervals. Absolute differences express the size of the gain directly; percentage changes over near-zero Random scores would exaggerate its apparent size.",
        "##### Experiments 4–5: beyond-accuracy and uncertainty",
    ]
    section.append(
        table(
            ["Model", "Coverage@10 [95% CI]", "Novelty@10", "ILD@10", "Serendipity@10"],
            [
                [name, interval(metric(name, "coverage")), *[f"{metric(name, m)['mean']:.4f}" for m in ["novelty", "ild", "serendipity"]]]
                for name in order
            ],
        )
    )
    winners = []
    for name in ["coverage", "novelty", "ild", "serendipity"]:
        part = metrics[(metrics.experiment == "experiment2") & (metrics.k == 10) & (metrics.metric == name)]
        individual = part[part.category == "individual"].sort_values("mean").iloc[-1]
        hybrid = part[part.category == "hybrid"].sort_values("mean").iloc[-1]
        winners.append([name, f"{individual.model}: {individual['mean']:.4f}", f"{hybrid.model}: {hybrid['mean']:.4f}"])
    section.append(table(["Metric@10", "Highest individual mean", "Highest hybrid mean"], winners))
    corr = read("tradeoff_correlation_summary.csv")
    part = corr[(corr.k == 10) & (corr.scope == "non_baselines") & (corr.measure == "pearson")]
    section.append(
        table(
            ["Beyond metric", "Across-model Pearson r with NDCG [seed CI]"], [[r.beyond_accuracy_metric, interval(r, 3)] for _, r in part.iterrows()]
        )
    )
    section += [
        (
            "Serendipity includes relevance, so its strong correlation with NDCG is partly structural. "
            "Coverage, novelty and ILD do not establish a single best model. Weighted and Combination lie on the four-dimensional Pareto frontier in all five seeds."
        ),
        (
            "![Catalogue coverage versus K, across-seed intervals](plots/coverage_by_k.png)\n\n"
            "![Novelty versus K, across-seed intervals](plots/novelty_by_k.png)"
        ),
        (
            "The separate bootstrap uses seed 2020 only, resampling the same 943 user positions for every model and K. "
            "Its 990 point estimates include 90 coverage values with no user-bootstrap interval; 900 additive estimates have marginal intervals, and 7,650 model-pair contrasts have paired intervals. "
            "Coverage is point-only there because resampling users shrinks unique-item unions. The coverage intervals in the plots above are instead across independent seed runs."
        ),
        (
            "The saved Random exports have repeated score rows (four per run), so their low catalogue coverage describes those saved exports, not independent per-user uniform random lists. "
            "The report retains this baseline limitation and does not infer its cause."
        ),
        "#### Task 2.3 — Hybrid contribution analysis",
        (
            "Weighted diagnostics use TunedWeighted with candidate depth 100 and C=0.0001 from its selected settings. "
            "The regularization sweep is a sensitivity diagnostic, not a replacement for the selected model. "
            "The ten members include Pop and exclude Random. Coefficients depend on correlated member scores and are not causal attribution. "
            "Refit ablations remove a member, rebuild the candidate pool from the remaining members and refit their weights; fixed-weight ablations set that member's weight to zero while keeping the full candidate pool and other weights. "
            "Both are evaluated on held-out validation users, and a positive full-minus-ablated difference favours retaining the member."
        ),
        "##### Coefficients and member removal",
    ]
    bootstrap_pairs = pd.read_csv(root / "bootstrap" / f"seed_{protocol['bootstrap']['seed']}" / "paired_differences.csv")
    bootstrap_pairs = bootstrap_pairs.rename(columns={"difference": "mean"}).assign(experiment="experiment2")
    split_table = table(
        ["Seed 2020 contrast", "Paired user-bootstrap NDCG@10 difference [95% CI]"],
        [
            [f"{left} minus {right}", interval(paired(bootstrap_pairs, left, right), 5, signed=True)]
            for left, right in [("TunedWeighted", "SLIMElastic"), ("TunedCombination", "SLIMElastic"), ("TunedWeighted", "TunedCombination")]
        ],
    )
    section.insert(section.index("#### Task 2.3 — Hybrid contribution analysis"), split_table)
    coefficients = read("task23/weighted_coefficients_summary.csv")
    refit = read("task23/weighted_leave_one_out_summary.csv")
    fixed = read("task23/weighted_fixed_weight_ablation_summary.csv")
    member_order = coefficients[coefficients.measure == "absolute_magnitude_share"].sort_values("mean", ascending=False).member.tolist()
    section.append(
        table(
            ["Member", "Coefficient [95% CI]", "Absolute magnitude share"],
            [
                [
                    name,
                    interval(select(coefficients, member=name, measure="coefficient")),
                    f"{100 * select(coefficients, member=name, measure='absolute_magnitude_share')['mean']:.2f}%",
                ]
                for name in member_order
            ],
        )
    )
    section.append(
        table(
            ["Removed member", "Refit full − ablated NDCG [95% CI]", "Fixed weights full − ablated [95% CI]"],
            [
                [
                    name,
                    interval(select(refit, removed_member=name, measure="full_minus_ablated"), 5, signed=True),
                    interval(select(fixed, removed_member=name, measure="full_minus_ablated"), 5, signed=True),
                ]
                for name in member_order
            ],
        )
    )
    section += [
        (
            "SLIMElastic and NeuMF have positive intervals in both removal analyses. FISM and BPR have positive point estimates but their intervals include zero, so these experiments do not establish positive marginal gains for them. "
            "EASE has the largest coefficient share, yet neither removal analysis establishes a positive marginal gain. Coefficient magnitude and marginal contribution answer different questions."
        ),
        (
            "The regularization sweep has its highest mean validation NDCG at C=0.0001 (0.2692). Larger C weakens shrinkage and reduces mean NDCG to 0.2570 at C=1. "
            "Mean NDCG ranges from 0.2682 to 0.2692 for C between 1e-7 and 1e-4; paired contrasts would be needed to distinguish settings within this region. "
            "Strong shrinkage is consistent with redundant member signals and helps limit unstable cancellation between their scores."
        ),
        "##### Activity and hybrid architecture diagnostics",
    ]
    regularization = read("task23/weighted_regularization_summary.csv")
    sensitivity = regularization[regularization.measure == "cv_ndcg@10"].sort_values("C")
    section.insert(
        section.index("##### Activity and hybrid architecture diagnostics"),
        table(["Diagnostic C", "Validation-user CV NDCG@10 [95% seed CI]"], [[f"{r.C:g}", interval(r)] for _, r in sensitivity.iterrows()]),
    )
    activity = read("task23/weighted_by_activity_summary.csv")
    section.append(
        table(
            ["Activity quartile", "Global Weighted NDCG", "Best individual NDCG", "Paired difference [95% CI]"],
            [
                [
                    group,
                    f"{select(activity, activity_group=group, measure='weighted_ndcg@10')['mean']:.4f}",
                    f"{select(activity, activity_group=group, measure='best_individual_ndcg@10')['mean']:.4f}",
                    interval(select(activity, activity_group=group, measure="weighted_minus_best"), 5, signed=True),
                ]
                for group in range(1, 5)
            ],
        )
    )
    section += [
        (
            "Global Weighted has clearer positive differences in activity groups 3–4; groups 1–2 include zero. "
            "Separately refitting group-specific weights does not establish a gain: all four group-specific-minus-global intervals include zero."
        ),
        (
            "**Selection bias:** Mixed, Switching and Cascade diagnostics reuse the validation labels that tuned them and are optimistic in-sample descriptions. "
            "Weighted and Combination CV diagnostics also reuse hyperparameter-selection data; they are not nested CV or unbiased test estimates. No test labels are used to fit contribution models."
        ),
    ]
    mixed = read("task23/mixed_member_ablation_summary.csv")
    cascade = read("task23/cascade_stages_summary.csv")
    section.append(
        f"Mixed has validation NDCG {interval(select(mixed, removed_member=None, measure='validation_ndcg@10'))}. "
        "Switching selects the best member within each training-activity group using validation labels; matching the best validation member is therefore true by construction. "
        "Neither observation alone establishes a held-out routing gain."
    )
    section.append(
        table(
            ["Mixed removed member", "Validation NDCG@10 [95% CI]", "Full − ablated [95% CI]"],
            [
                [
                    name,
                    interval(select(mixed, removed_member=name, measure="validation_ndcg@10")),
                    interval(select(mixed, removed_member=name, measure="full_minus_ablated"), 5, signed=True),
                ]
                for name in member_order
            ],
        )
    )
    section.append(
        "Only FISM's removal from Mixed has a positive full-minus-ablated interval; the other intervals include zero. "
        "This validation diagnostic differs from Weighted's SLIMElastic/NeuMF pattern, indicating that a member's apparent contribution depends on the combination rule and selection procedure."
    )
    section.append(
        table(
            ["Cascade diagnostic", "Validation NDCG [95% seed CI]"],
            [[r.variant, interval(r)] for _, r in cascade[cascade.measure == "validation_ndcg@10"].iterrows()],
        )
    )
    section += [
        "##### Combination feature-block ablation",
        (
            "The classifier uses the selected Task 2 TunedCombination settings: unlimited depth (`depth=null`) and learning rate 0.03, fixed across feature variants and seeds. "
            "These five-seed settings differ from the single-run Task 1 depth-2 configuration; the two experiments should be interpreted separately. "
            "The user block contains age, gender, occupation and training count. The movie block contains genres, release year and training popularity. "
            "Occupation is categorical only in variants retaining the user block. Results are five-fold validation-user diagnostics, aggregated across seeds."
        ),
    ]
    feature = read("task23/combination_feature_ablation_summary.csv")
    variants = [
        "Member scores only",
        "Member scores plus user features",
        "Member scores plus movie features",
        "All member scores and user/movie side features",
    ]
    section.append(
        table(
            ["Feature variant", "Validation-user CV NDCG@10 [95% seed CI]"],
            [
                [
                    name,
                    interval(select(feature, variant=name, measure="cv_ndcg@10")),
                ]
                for name in variants
            ],
        )
    )
    section += [
        (
            "User features raise mean validation NDCG from 0.2368 to 0.2661, close to the full model's 0.2671. "
            "Movie features alone reach 0.2461. Within this classifier and candidate pool, the user block therefore accounts for the larger observed improvement. "
            "These differences between means are descriptive: marginal intervals do not test paired feature-block differences, and correlated features and interactions prevent additive causal attribution."
        ),
        "#### Task 2.4 — Hybrid behaviour and long-tail analysis",
        (
            "User routing, retrieval and exposure are evaluated on test interactions. Popularity groups are defined by training counts: "
            "head = top 20% of movies, middle = next 30%, tail = bottom 50%, with boundaries rebuilt per seed. "
            "Training activity and training tail-interest define user segments without using test labels."
        ),
        "##### Routing and candidate pipelines",
    ]
    routing = read("task24/routing_summary.csv")
    section.append(
        table(
            ["Routing measure", "NDCG@10 [95% seed CI]"],
            [[name, interval(select(routing, metric=name))] for name in ["oracle_ndcg@10", "combination_ndcg@10", "oracle_minus_combination"]],
        )
    )
    candidates = read("task24/candidate_pipeline_summary.csv")
    best = candidates[candidates.measure == "ndcg@10"].sort_values("mean", ascending=False).iloc[0]
    section += [
        "The oracle chooses using each user’s test outcomes and is a descriptive upper bound, not an achievable routing policy.",
        (
            f"The highest observed pipeline mean is {best.candidate_generator} retrieval at depth {best.depth} followed by {best.reranker} reranking: "
            f"NDCG {interval(best)}. The 8 × 6 × 5 sweep is exploratory and selected on test outcomes; it is not a validation-selected serving policy. "
            "Candidate pools exclude non-finite scores, so depth is an upper bound on the number of available items."
        ),
        "##### Exposure and specialized retrieval",
    ]
    exposure = read("task24/popularity_exposure_summary.csv")
    names = ["EASE", "SLIMElastic", "Weighted", "Combination", "LightGCN", "FISM", "NeuMF"]
    section.append(
        table(
            ["Model", "Head exposure", "Tail exposure"],
            [
                [
                    name,
                    f"{100 * select(exposure, model=name, bucket='head_top20pct', measure='exposure_share')['mean']:.2f}%",
                    f"{100 * select(exposure, model=name, bucket='tail_bottom50pct', measure='exposure_share')['mean']:.3f}%",
                ]
                for name in names
            ],
        )
    )
    specialized = read("task24/specialized_retrieval_summary.csv")
    deltas = read("task24/specialized_retrieval_deltas_summary.csv")
    strategies = specialized.strategy.drop_duplicates().tolist()
    srow = lambda strategy, measure: select(specialized, strategy=strategy, measure=measure)
    section.append(
        table(
            [
                "Candidate strategy",
                "Candidate coverage",
                "Tail candidate recall",
                "Paired recall gain vs base [95% CI, pp]",
                "NDCG@10",
                "Tail exposure",
            ],
            [
                [
                    name,
                    f"{100 * srow(name, 'candidate_catalogue_coverage')['mean']:.2f}%",
                    f"{100 * srow(name, 'tail_candidate_recall')['mean']:.2f}%",
                    "—"
                    if name == "Combination_top100_base"
                    else interval(select(deltas, strategy=name, measure="tail_candidate_recall_delta_eligible_users"), 2, scale=100, signed=True),
                    f"{srow(name, 'ndcg@10')['mean']:.5f}",
                    f"{100 * srow(name, 'tail_exposure')['mean']:.4f}%",
                ]
                for name in strategies
            ],
        )
    )
    tail = read("task24/tail_interest_segments_summary.csv")
    baseline = "Combination_top100_base"
    union = "Base_plus_LightGCN_and_genre_tail"
    trow = lambda group, strategy, measure: select(tail, tail_interest_quartile=group, strategy=strategy, measure=measure)
    section.append(
        table(
            ["Training tail-interest quartile", "Users with tail test items (mean)", "Base tail candidate recall", "Union tail candidate recall"],
            [
                [
                    group,
                    f"{trow(group, baseline, 'users_with_tail_test_items')['mean']:.0f}",
                    f"{100 * trow(group, baseline, 'mean_tail_candidate_recall')['mean']:.2f}%",
                    f"{100 * trow(group, union, 'mean_tail_candidate_recall')['mean']:.2f}%",
                ]
                for group in range(1, 5)
            ],
        )
    )
    section += [
        (
            "All strategies use the same SLIMElastic reranker. Adding specialized candidates increases tail candidate recall without materially changing final NDCG or tail exposure. "
            "Tail recall@10 remains about 0.060%; tiny tail-hit counts make conditional hit rates unstable. This supports a bottleneck in converting retrieved tail items into top-ten recommendations, not proof of an effective deployable tail-ranking policy."
        ),
        "### Interpretation and limitations",
        (
            "**Hybrid effectiveness.** Weighted and Combination improve mean NDCG over the strongest individual model, SLIMElastic, by about 2.8% and 2.6%, respectively. "
            "Their paired seed intervals support positive gains over SLIMElastic, while their difference from each other includes zero. "
            "Mixed and Cascade also reach higher mean NDCG than SLIMElastic, but Switching, Augmentation and Metalevel do not. "
            "Combining models is therefore useful for some architectures rather than a guarantee of improvement."
        ),
        (
            "**Sources of the gain.** Weighted's removal tests identify SLIMElastic and NeuMF as complementary contributors despite EASE having the largest coefficient. "
            "The Combination diagnostic shows a larger mean improvement from user features than movie features. "
            "These findings support combining different predictive signals and conditioning on user characteristics, while coefficient size alone is insufficient evidence of unique value. "
            "The activity analysis further suggests that a global blend helps users with longer histories more consistently than users with shorter histories; separately fitted group weights have no established advantage."
        ),
        (
            "**Accuracy and catalogue breadth.** LightGCN leads individual-model coverage and novelty, ItemKNN leads ILD, and Weighted leads hybrid serendipity. "
            "The accuracy leaders still allocate over 96% of recommendations to the head of the catalogue. "
            "Specialized retrieval raises tail candidate recall, particularly for users with greater training tail-interest, but a common SLIMElastic reranker leaves final tail exposure almost unchanged. "
            "For this pipeline, broadening retrieval alone is insufficient to translate catalogue access into recommendations."
        ),
        (
            "**Scope of inference.** Five seeds provide limited precision and combine split and training variability. "
            "Intervals are pointwise with no adjustment for the many models, metrics and ablations. Comparisons use paired differences rather than marginal-interval overlap. "
            "Validation contribution diagnostics reuse hyperparameter-selection data; Mixed, Switching and Cascade additionally reuse labels directly in selecting their configurations. "
            "They describe mechanisms but do not provide unbiased held-out gains. The test-selected candidate sweep and test-outcome oracle are exploratory bounds, not deployment policies. "
            "The seed-2020 bootstrap is conditional on one trained set of models and cannot replace seed-level uncertainty. "
            "The repeated Random score rows limit interpretation of its coverage, and sparse tail hits limit conditional tail-rate estimates. "
            "Undefined rates retain their actual number of contributing seeds; ILD is constant at K=1, so its correlations are undefined."
        ),
        table(
            ["Complete experiment output", "Contents"],
            [
                [
                    "[Task 2.2 metrics](../results/task2/reanalysis_cpu/aggregate/metric_summary.csv)",
                    "All model/metric/K combinations with seed means, SDs, sample counts and 95% intervals",
                ],
                [
                    "[Model differences](../results/task2/reanalysis_cpu/aggregate/paired_model_difference_summary.csv)",
                    "Paired seed contrasts for ranking and beyond-accuracy metrics",
                ],
                [
                    "[Task 2.3 tables](../results/task2/reanalysis_cpu/aggregate/task23/)",
                    "Coefficients, regularization, activity-specific policies and member removals, correlations, hybrid diagnostics and feature-block ablations",
                ],
                [
                    "[Task 2.4 tables](../results/task2/reanalysis_cpu/aggregate/task24/)",
                    "All candidate pipelines, routing groups, popularity buckets, exposure, specialized retrieval and tail-interest segments",
                ],
                [
                    "[Seed-2020 bootstrap](../results/task2/reanalysis_cpu/bootstrap/seed_2020/)",
                    "Point estimates, additive-metric intervals and paired user-bootstrap contrasts",
                ],
            ],
        ),
    ]
    original_text = notes.read_text()
    start = original_text.index("## Task 2 — Evaluation of Effectiveness")
    end = original_text.index("## Appendix A — Models", start)
    notes.write_text(original_text[:start] + "\n\n".join(section) + "\n\n" + original_text[end:])
    for name in ["ndcg", "coverage", "novelty", "serendipity"]:
        destination = notes.parent / "plots" / f"{name}_by_k.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aggregate / "plots" / f"{name}_by_k.png", destination)
    print(f"Updated {notes}; audited {checks['per_seed_metric_rows']} metric rows and {checks['checked_summary_rows']} summary rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/task2/reanalysis_cpu"))
    parser.add_argument("--notes", type=Path, default=Path("report/notes.md"))
    args = parser.parse_args()
    build(args.results.resolve(), args.notes)


if __name__ == "__main__":
    main()
