"""Resumable, fixed-configuration runs and seed-level summaries for Tasks 2.2–2.4."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.task22.task22_seeds import KS, REPO, SEEDS, _ci_summary, _prepare_workdir, aggregate as aggregate_task22
from source.data import MODELS

HYBRIDS = ("Weighted", "Mixed", "Cascade", "Switching", "Combination", "Augmentation", "Metalevel")
TABLES = {
    "task23": {
        "weighted_coefficients": ["member"],
        "weighted_regularization": ["C"],
        "weighted_coefficients_by_C": ["C", "member"],
        "weighted_leave_one_out": ["removed_member"],
        "weighted_fixed_weight_ablation": ["removed_member"],
        "member_score_correlations": ["left_member", "right_member"],
        "weighted_by_activity": ["activity_group"],
        "weighted_by_activity_policy": ["activity_group"],
        "weighted_activity_coefficients": ["activity_group", "member"],
        "weighted_leave_one_out_by_activity": ["activity_group", "removed_member"],
        "cascade_stages": ["variant"],
        "combination_feature_ablation": ["variant"],
        "mixed_member_ablation": ["removed_member"],
        "switching_by_user_group": ["activity_group"],
    },
    "task24": {
        "candidate_pipeline": ["candidate_generator", "reranker", "depth"],
        "popularity_buckets": ["model", "bucket"],
        "popularity_exposure": ["model", "bucket"],
        "routing_segments": ["segment", "segment_value", "model"],
        "specialized_retrieval": ["strategy"],
        "specialized_retrieval_deltas": ["strategy", "comparison"],
        "tail_interest_segments": ["tail_interest_quartile", "strategy"],
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def input_hashes():
    paths = {REPO / "uv.lock", REPO / "pyproject.toml"}
    for folder, pattern in (
        ("scripts", "*.py"),
        ("source", "*.py"),
        ("parameters", "*.yaml"),
        ("parameters", "*.json"),
        ("data/ml-100k", "ml-100k.*"),
    ):
        paths.update((REPO / folder).rglob(pattern))
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(paths) if p.is_file()}


def prepare(root, reference, seeds, ks, training_config=None, require_gpu=False):
    settings = {name: json.loads((reference / f"Tuned{name}.json").read_text()) for name in HYBRIDS}
    protocol = {
        "version": 2,
        "seeds": list(seeds),
        "ks": list(ks),
        "inputs": input_hashes(),
        "hybrid_settings": settings,
        "training_config": training_config.read_text() if training_config is not None else None,
        "require_gpu": require_gpu,
    }
    target = root / "run_protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Run inputs changed. Use a new --output directory to keep seed results comparable.")
    write_json(target, protocol)
    if protocol["training_config"] is not None:
        (root / "training_config.yaml").write_text(protocol["training_config"])
    for name, value in settings.items():
        write_json(root / "reference_metrics" / f"Tuned{name}.json", value)
    print(f"Prepared {root}: seeds={seeds}, K={ks}", flush=True)


def protocol_for(root):
    protocol = json.loads((root / "run_protocol.json").read_text())
    if protocol.get("version") != 2:
        raise ValueError("Legacy run protocol; prepare a new output directory for corrected analyses")
    if protocol["training_config"] is not None and (root / "training_config.yaml").read_text() != protocol["training_config"]:
        raise ValueError("Frozen training configuration changed")
    if protocol["inputs"] != input_hashes():
        raise ValueError("Code, parameters, lockfile, or data changed since prepare; create a new run directory.")
    for name, value in protocol["hybrid_settings"].items():
        if json.loads((root / "reference_metrics" / f"Tuned{name}.json").read_text()) != value:
            raise ValueError(f"Frozen settings changed: {name}")
    return protocol


def valid_marker(marker, base, signature):
    if not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text())
        return (
            saved["signature"] == signature
            and bool(saved["files"])
            and all((base / p).is_file() and digest(base / p) == value for p, value in saved["files"].items())
        )
    except (ValueError, KeyError, OSError):
        return False


def record(marker, base, signature, paths):
    files = {str(p.relative_to(base)): digest(p) for p in paths if p.is_file()}
    if not files:
        raise RuntimeError(f"No outputs for {marker}")
    write_json(marker, {"signature": signature, "files": files})


def signature(root, seed):
    return f"{digest(root / 'run_protocol.json')}:{seed}"


def execute(module, arguments, workdir):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(REPO / "scripts"), str(REPO)))
    env["MPLCONFIGDIR"] = str(workdir / "mplconfig")
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print(f"[{workdir.name}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=workdir, env=env, check=True)


def train(root, seed, protocol):
    workdir = root / "runs" / f"seed_{seed}"
    _prepare_workdir(workdir)
    for model in ["Random", *[name for name in MODELS if name != "Random"]]:
        marker = workdir / "stages" / f"train_{model}.json"
        sig = signature(root, seed)
        if valid_marker(marker, workdir, sig):
            print(f"Seed {seed}: reuse {model}", flush=True)
            continue
        arguments = [model, "--tuned", "--seed", seed]
        if protocol["training_config"] is not None:
            arguments += ["--config", root / "training_config.yaml"]
        if protocol["require_gpu"]:
            arguments += ["--require-gpu"]
        execute("scripts.export_scores", arguments, workdir)
        files = [workdir / f"results/tuned/{kind}/{model}.{ext}" for kind, ext in (("scores", "npz"), ("metrics", "json"))]
        files += list((workdir / "results/split").glob("*.tsv"))
        if model in ("LightGCN", "NGCF"):
            embeddings = list((workdir / "results/checkpoints").glob(f"*-{model}-*-embeddings.npz"))
            if not embeddings:
                raise RuntimeError(f"Missing {model} embeddings required by Meta-Level")
            files += embeddings
        if not all(p.is_file() for p in files) or len(list((workdir / "results/split").glob("*.tsv"))) != 3:
            raise RuntimeError(f"Incomplete export: {model}")
        record(marker, workdir, sig, files)


def training_complete(root, seed):
    workdir = root / "runs" / f"seed_{seed}"
    return all(valid_marker(workdir / "stages" / f"train_{name}.json", workdir, signature(root, seed)) for name in MODELS)


def analyze(root, seed, protocol):
    if not training_complete(root, seed):
        raise RuntimeError(f"Seed {seed}: training incomplete or artifacts changed")
    workdir = root / "runs" / f"seed_{seed}"
    sig = signature(root, seed)
    tasks = [
        ("hybrids", "scripts.task22.build_task22_repeat_hybrids", ["--reference-metrics", root / "reference_metrics"], "hybrid"),
        ("task22", "scripts.task22.task22_pipeline", ["--output", "results/task2/repeated_task22", "--ks", *protocol["ks"]], "task2/repeated_task22"),
        ("task23", "scripts.task23.analyze_contributions", [], "task23"),
        ("task24", "scripts.task24.task24_pipeline", [], "task24"),
    ]
    # A rebuilt upstream stage invalidates every downstream stage, even when files previously existed.
    upstream_changed = False
    for stage, module, args, folder in tasks:
        marker = workdir / "stages" / f"{stage}.json"
        if not upstream_changed and valid_marker(marker, workdir, sig):
            print(f"Seed {seed}: reuse {stage}", flush=True)
            continue
        upstream_changed = True
        (workdir / "seed_complete.json").unlink(missing_ok=True)
        execute(module, args, workdir)
        record(marker, workdir, sig, (workdir / "results" / folder).rglob("*"))
    write_json(
        workdir / "seed_complete.json", {"protocol_version": 2, "seed": seed, "ks": protocol["ks"], "complete": True, "tasks": ["2.2", "2.3", "2.4"]}
    )


def seed_complete(root, seed):
    workdir = root / "runs" / f"seed_{seed}"
    return (
        training_complete(root, seed)
        and (workdir / "seed_complete.json").exists()
        and all(
            valid_marker(workdir / "stages" / f"{stage}.json", workdir, signature(root, seed)) for stage in ("hybrids", "task22", "task23", "task24")
        )
    )


def aggregate_tables(root, seeds):
    """Aggregate seed-level means; never pool users or average within-run CI endpoints."""
    output = root / "aggregate"
    for task, tables in TABLES.items():
        folder = output / task
        folder.mkdir(parents=True, exist_ok=True)
        for name, keys in tables.items():
            frames = [pd.read_csv(root / "runs" / f"seed_{seed}" / "results" / task / f"{name}.csv").assign(seed=seed) for seed in seeds]
            combined = pd.concat(frames, ignore_index=True)
            if combined.duplicated(["seed", *keys]).any():
                raise ValueError(f"Duplicate seed/group observations in {task}/{name}")
            measures = [
                c
                for c in combined.select_dtypes(include="number")
                if c not in ["seed", *keys] and "ci95" not in c and not c.endswith("fold_sd") and c != "sd_across_folds"
            ]
            combined.to_csv(folder / f"{name}_per_seed.csv", index=False)
            summary = _ci_summary(combined, keys, measures)
            if (combined.groupby(keys, dropna=False).seed.nunique() != len(seeds)).any():
                raise ValueError(f"Missing seed/group observations in {task}/{name}")
            summary.to_csv(folder / f"{name}_summary.csv", index=False)
        routing, selections = [], []
        for seed in seeds:
            if task == "task24":
                row = json.loads((root / "runs" / f"seed_{seed}" / "results/task24/routing_summary.json").read_text())
                routing += [
                    {"seed": seed, "metric": k, "value": row[k]} for k in ("oracle_ndcg@10", "combination_ndcg@10", "oracle_minus_combination")
                ]
                routing += [{"seed": seed, "metric": f"win_share_{model}", "value": value} for model, value in row["model_win_rates"].items()]
            else:
                summary = json.loads((root / "runs" / f"seed_{seed}" / "results/task23/summary.json").read_text())
                for group, row in enumerate(summary["switching"], 1):
                    selections += [
                        {"seed": seed, "activity_group": group, "member": name, "selected": int(row["selected_member"] == name)}
                        for name in MODELS
                        if name != "Random"
                    ]
        if routing:
            frame = pd.DataFrame(routing)
            frame.to_csv(folder / "routing_per_seed.csv", index=False)
            _ci_summary(frame, ["metric"], ["value"]).to_csv(folder / "routing_summary.csv", index=False)
        if selections:
            frame = pd.DataFrame(selections)
            frame.to_csv(folder / "switching_selection_per_seed.csv", index=False)
            _ci_summary(frame, ["activity_group", "member"], ["selected"]).to_csv(folder / "switching_selection_summary.csv", index=False)


def aggregate(root, protocol):
    incomplete = [seed for seed in protocol["seeds"] if not seed_complete(root, seed)]
    if incomplete:
        raise RuntimeError(f"Refusing partial aggregate; incomplete seeds: {incomplete}")
    marker = root / "aggregate_complete.json"
    marker.unlink(missing_ok=True)
    aggregate_task22(root, protocol["seeds"], protocol["ks"])
    aggregate_tables(root, protocol["seeds"])
    write_json(
        root / "aggregate/all_tasks_protocol.json",
        {
            "seeds": protocol["seeds"],
            "ks": protocol["ks"],
            "tasks": ["2.2", "2.3", "2.4"],
            "uncertainty": "Mean, sample SD and 95% Student-t intervals across seed-level estimates; not user-bootstrap intervals.",
            "selection": "Individual and hybrid hyperparameters fixed from reference run; learned hybrid parameters refitted per seed.",
            "limitations": [
                "Five seeds are a small sample; split and training variability are combined.",
                "Task 2.3 feature-block ablation uses frozen TunedCombination depth and rate; validation diagnostics reuse selection data and are not nested-CV estimates.",
                "Task 2.4 oracle and candidate policies remain exploratory; aggregation does not validate deployment.",
                "Activity and popularity buckets are rebuilt within each seed; group sizes and boundaries may vary.",
            ],
            "undefined_metrics": "Undefined within-seed rates are excluded from that measure only; n_seeds reports its actual count.",
        },
    )
    record(marker, root, digest(root / "run_protocol.json"), (root / "aggregate").rglob("*"))
    print(f"All Task 2.2–2.4 aggregates complete: {root / 'aggregate'}", flush=True)


def status(root):
    if not (root / "run_protocol.json").exists():
        print(json.dumps({"complete": False, "reason": "No multi-seed run_protocol.json", "root": str(root)}, indent=2))
        return False
    protocol = protocol_for(root)
    complete = valid_marker(root / "aggregate_complete.json", root, digest(root / "run_protocol.json"))
    runs = {str(s): {"trained": training_complete(root, s), "complete": seed_complete(root, s)} for s in protocol["seeds"]}
    complete = complete and all(r["complete"] for r in runs.values())
    print(json.dumps({"complete": complete, "seeds": runs, "root": str(root)}, indent=2))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "train", "analyze", "aggregate", "status"), required=True)
    parser.add_argument("--output", type=Path, default=Path("results/task2/multiseed"))
    parser.add_argument("--reference-metrics", type=Path, default=Path("results/hybrid/metrics"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--training-config", type=Path, help="Optional RecBole override YAML, frozen during prepare")
    parser.add_argument("--require-gpu", action="store_true", help="Require CUDA during the prepared training phase")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase == "prepare":
        if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)) or sorted(args.ks) != list(KS):
            parser.error("Use at least two unique seeds and all cutoffs 1 3 5 10 20")
        prepare(root, args.reference_metrics.resolve(), args.seeds, args.ks, args.training_config, args.require_gpu)
    elif args.phase == "status":
        sys.exit(0 if status(root) else 1)
    else:
        protocol = protocol_for(root)
        if args.phase in ("train", "analyze"):
            if args.seed not in protocol["seeds"]:
                parser.error("--seed must be one of the prepared seeds")
            (train if args.phase == "train" else analyze)(root, args.seed, protocol)
        else:
            aggregate(root, protocol)


if __name__ == "__main__":
    main()
