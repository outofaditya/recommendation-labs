"""Analyze verified Task 2 exports on CPU without training base models or hybrids."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.task2.multiseed import aggregate_tables, digest, input_hashes, record, valid_marker, write_json
from scripts.task22.task22_seeds import REPO, aggregate as aggregate_task22


SOURCE_MODELS = ("Random", "Pop", "FISM", "NGCF", "LightGCN", "BPR", "NeuMF", "SLIMElastic", "ItemKNN", "UserKNN", "EASE")


def verify_analysis_inputs(output, protocol):
    """Verify the executed snapshot even when reporting code has since changed."""
    for relative, expected in protocol["analysis_inputs"].items():
        snapshot = output / "analysis_snapshot" / relative
        path = snapshot if snapshot.is_file() else REPO / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Executed analysis input missing or changed: {relative}")


def verify_source(source):
    """Validate original manifests against relocated files without changing them."""
    protocol_path = source / "run_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    signature = digest(protocol_path)
    evidence = {}
    snapshot = source.parents[2]
    for relative, expected in protocol["inputs"].items():
        path = snapshot / relative
        if not path.is_file() and relative.startswith("data/"):
            path = REPO / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Original input missing or changed: {relative}")
    for seed in protocol["seeds"]:
        work = source / "runs" / f"seed_{seed}"
        for stage in [*[f"train_{name}" for name in SOURCE_MODELS], "hybrids"]:
            marker = work / "stages" / f"{stage}.json"
            if not valid_marker(marker, work, f"{signature}:{seed}"):
                raise ValueError(f"Original stage artifacts failed validation: {marker}")
            evidence[str(marker.relative_to(source))] = digest(marker)
        for name in ("train", "valid", "test"):
            if not (work / "results/split" / f"{name}.tsv").is_file():
                raise ValueError(f"Missing seed {seed} split {name}")
    return protocol, evidence


def prepare(source, output, threads, bootstrap_reps):
    original, evidence = verify_source(source)
    # Match atomic side data used by the original training, including local data
    # accessed by load_side() rather than only exported score IDs.
    for relative, expected in original["inputs"].items():
        if relative.startswith("data/") and digest(REPO / relative) != expected:
            raise ValueError(f"Local data changed: {relative}")
    protocol = {
        "version": 1,
        "mode": "CPU reanalysis of verified saved exports; no individual or hybrid retraining",
        "source_root": str(source),
        "source_protocol_sha256": digest(source / "run_protocol.json"),
        "source_stage_marker_sha256": evidence,
        "source_inputs": original["inputs"],
        "hybrid_settings": original["hybrid_settings"],
        "seeds": original["seeds"],
        "ks": original["ks"],
        "analysis_inputs": input_hashes(),
        "threads_per_process": threads,
        "bootstrap": {"seed": original["seeds"][0], "reps": bootstrap_reps, "random_seed": 22022},
    }
    target = output / "reanalysis_protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Reanalysis inputs changed; use a new --output directory")
    # A redirected run.log may exist before the first prepare call.
    if not target.exists() and output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        raise ValueError("Output directory is not empty; use a new directory")
    write_json(target, protocol)
    for relative, expected in protocol["analysis_inputs"].items():
        path = REPO / relative
        if digest(path) != expected:
            raise ValueError(f"Analysis input changed during prepare: {relative}")
        snapshot = output / "analysis_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, snapshot)
    for seed in original["seeds"]:
        work = output / "runs" / f"seed_{seed}"
        results = work / "results"
        results.mkdir(parents=True, exist_ok=True)
        for name in ("tuned", "split", "checkpoints", "hybrid"):
            path = results / name
            expected = source / "runs" / f"seed_{seed}" / "results" / name
            if path.is_symlink():
                if path.resolve() != expected.resolve():
                    raise ValueError(f"Unexpected input link: {path}")
            elif path.exists():
                raise ValueError(f"Refusing to replace {path}")
            else:
                path.symlink_to(expected, target_is_directory=True)
        data = work / "data"
        if not data.exists():
            data.symlink_to(REPO / "data", target_is_directory=True)
        elif data.resolve() != (REPO / "data").resolve():
            raise ValueError(f"Unexpected data path: {data}")
        write_json(
            work / "reused_exports.json",
            {
                "source": str(source / "runs" / f"seed_{seed}"),
                "source_protocol_sha256": protocol["source_protocol_sha256"],
                "base_models_retrained": False,
                "hybrids_retrained": False,
            },
        )
    return protocol


def execute(module, arguments, work, logfile, threads):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(REPO / "scripts"), str(REPO))),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(work / "mplconfig"),
            **{
                name: str(threads)
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        }
    )
    logfile.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print(f"[{work.name}] START {module}; log={logfile}", flush=True)
    with logfile.open("w") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f"[{work.name}] DONE {module}", flush=True)


def signature(output, seed):
    return f"{digest(output / 'reanalysis_protocol.json')}:{seed}"


def run_seed(output, seed, protocol):
    work = output / "runs" / f"seed_{seed}"
    sig = signature(output, seed)
    stages = [
        ("task22", "scripts.task22.task22_pipeline", ["--output", "results/task2/repeated_task22", "--ks", *protocol["ks"]], "task2/repeated_task22"),
        ("task23", "scripts.task23.analyze_contributions", [], "task23"),
        ("task24", "scripts.task24.task24_pipeline", [], "task24"),
    ]
    (work / "seed_complete.json").unlink(missing_ok=True)
    for stage, module, arguments, folder in stages:
        marker = work / "stages" / f"{stage}.json"
        if valid_marker(marker, work, sig):
            print(f"[{work.name}] REUSE corrected {stage}", flush=True)
            continue
        execute(module, arguments, work, work / "logs" / f"{stage}.log", protocol["threads_per_process"])
        record(marker, work, sig, (work / "results" / folder).rglob("*"))
    write_json(
        work / "seed_complete.json",
        {
            "protocol_version": 2,
            "seed": seed,
            "ks": protocol["ks"],
            "complete": True,
            "tasks": ["2.2", "2.3", "2.4"],
            "reused_exports": True,
        },
    )


def bootstrap(output, protocol):
    seed = protocol["bootstrap"]["seed"]
    work = output / "runs" / f"seed_{seed}"
    marker = output / "stages" / "bootstrap.json"
    sig = digest(output / "reanalysis_protocol.json")
    if valid_marker(marker, output, sig):
        return
    destination = output / "bootstrap" / f"seed_{seed}"
    execute(
        "scripts.task22.task22_bootstrap",
        [
            "--output",
            destination,
            "--reps",
            protocol["bootstrap"]["reps"],
            "--ks",
            *protocol["ks"],
            "--seed",
            protocol["bootstrap"]["random_seed"],
        ],
        work,
        output / "logs" / "bootstrap.log",
        protocol["threads_per_process"],
    )
    record(marker, output, sig, destination.rglob("*"))


def aggregate(output, protocol):
    if not valid_marker(output / "stages/bootstrap.json", output, digest(output / "reanalysis_protocol.json")):
        raise ValueError("Cannot finalize incomplete or changed bootstrap outputs")
    for seed in protocol["seeds"]:
        work = output / "runs" / f"seed_{seed}"
        for stage in ("task22", "task23", "task24"):
            if not valid_marker(work / "stages" / f"{stage}.json", work, signature(output, seed)):
                raise ValueError(f"Cannot aggregate incomplete/changed stage {seed}/{stage}")
    # Recheck original artifacts and source code after the run as well.
    verify_source(Path(protocol["source_root"]))
    if input_hashes() != protocol["analysis_inputs"]:
        raise ValueError("Analysis code/data/configuration changed while running; aggregate refused")
    aggregate_task22(output, protocol["seeds"], protocol["ks"])
    aggregate_tables(output, protocol["seeds"])
    write_json(output / "aggregate" / "all_tasks_protocol.json", protocol)
    with (output / "aggregate" / "README.md").open("a") as stream:
        stream.write(
            "\nThese corrected analyses reuse verified original base-model and hybrid exports. "
            "No base model or final hybrid was retrained. Contribution ablations were refitted on validation. "
            "See ../reanalysis_protocol.json for original and current input hashes.\n"
        )
    record(output / "aggregate_complete.json", output, digest(output / "reanalysis_protocol.json"), (output / "aggregate").rglob("*"))
    print(f"COMPLETE CPU reanalysis: {output / 'aggregate'}", flush=True)


def status(output):
    """Check completion without preparing a directory or launching computation."""
    path = output / "reanalysis_protocol.json"
    if not path.is_file():
        return {"complete": False, "reason": "Missing reanalysis_protocol.json"}
    protocol = json.loads(path.read_text())
    sig = digest(path)
    stages = {}
    for seed in protocol["seeds"]:
        work = output / "runs" / f"seed_{seed}"
        stages[str(seed)] = {
            stage: valid_marker(work / "stages" / f"{stage}.json", work, f"{sig}:{seed}") for stage in ("task22", "task23", "task24")
        }
    errors = []
    try:
        verify_source(Path(protocol["source_root"]))
        if digest(Path(protocol["source_root"]) / "run_protocol.json") != protocol["source_protocol_sha256"]:
            raise ValueError("Original protocol identity changed")
        verify_analysis_inputs(output, protocol)
    except (ValueError, OSError, KeyError) as error:
        errors.append(str(error))
    boot = valid_marker(output / "stages/bootstrap.json", output, sig)
    final = valid_marker(output / "aggregate_complete.json", output, sig)
    return {
        "complete": not errors and boot and final and all(all(values.values()) for values in stages.values()),
        "seeds": stages,
        "bootstrap": boot,
        "aggregate": final,
        "input_errors": errors,
        "base_models_retrained": False,
        "final_hybrids_retrained": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Original verified results/task2/multiseed directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate new reanalysis directory")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--status", action="store_true", help="Only validate saved completion, artifacts and executed source")
    args = parser.parse_args()
    if args.status:
        result = status(args.output.resolve())
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["complete"] else 1)
    if args.source_root is None:
        parser.error("--source-root is required unless --status is used")
    source, output = args.source_root.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate non-nested directories")
    if args.threads < 1 or args.bootstrap_reps < 100:
        parser.error("Use positive threads and at least 100 bootstrap replicates")
    protocol = prepare(source, output, args.threads, args.bootstrap_reps)
    for seed in protocol["seeds"]:
        run_seed(output, seed, protocol)
    bootstrap(output, protocol)
    aggregate(output, protocol)


if __name__ == "__main__":
    main()
