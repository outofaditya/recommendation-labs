import os
import time
import argparse
from pathlib import Path
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

import yaml
import torch
import numpy as np
import pandas as pd
from export_scores import fit
from hyperopt.base import Domain
from recbole.config import Config
from recbole.trainer import HyperTuning
from hyperopt import tpe, Trials, STATUS_OK, JOB_STATE_DONE, space_eval

SEARCH = Path("parameters/search")
TUNED = Path("parameters/tuned")
TRIALS = Path("results/tuning")
PLAN = {
    "EASE": ("exhaustive", None),
    "ItemKNN": ("exhaustive", None),
    "UserKNN": ("exhaustive", None),
    "SLIMElastic": ("exhaustive", None),
    "BPR": ("bayes", 60),
    "NeuMF": ("bayes", 70),
    "LightGCN": ("bayes", 50),
    "FISM": ("bayes", 40),
    "NGCF": ("bayes", 40),
}
os.environ.setdefault("HYPEROPT_FMIN_SEED", "2020")


def plain(value):
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value.item() if hasattr(value, "item") else value


def trial(files, params, threads=None):
    if threads:
        torch.set_num_threads(threads)
    params = {k: plain(v) for k, v in params.items()}
    config = Config(config_file_list=files, config_dict=params)
    start = time.perf_counter()
    *_, valid, epochs = fit(config, saved=False)
    seconds = round(time.perf_counter() - start)
    capped = epochs >= config["epochs"]
    return params | valid | {"epochs": epochs, "capped": capped, "seconds": seconds}


def sequential(name, files, rows):
    def objective(config_dict, config_file_list, saved=False):
        rows.append(trial(config_file_list, config_dict))
        return {
            "model": name,
            "test_result": {},
            "valid_score_bigger": True,
            "best_valid_result": rows[-1],
            "best_valid_score": rows[-1]["ndcg@10"],
        }

    algo, evals = PLAN[name]
    tuner = HyperTuning(
        objective,
        algo=algo,
        max_evals=evals,
        early_stop=10**6,
        fixed_config_file_list=files,
        params_file=str(SEARCH / f"{name}.hyper"),
    )
    tuner.run()


# tpe in synchronous batches so every worker trains at once
def batched(name, files, rows, workers):
    space = HyperTuning._build_space_from_file(SEARCH / f"{name}.hyper")
    trials, rng, evals = Trials(), np.random.RandomState(2020), PLAN[name][1]
    domain = Domain(lambda params: 0, space)
    context = get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        while len(trials) < evals:
            docs = []
            for _ in range(min(workers, evals - len(trials))):
                seed = rng.randint(2**31 - 1)
                docs += tpe.suggest(trials.new_trial_ids(1), domain, trials, seed)
            vals = [{k: v[0] for k, v in d["misc"]["vals"].items() if v} for d in docs]
            params = [space_eval(space, v) for v in vals]
            batch = list(pool.map(trial, [files] * len(docs), params, [1] * len(docs)))
            for doc, row in zip(docs, batch):
                doc["state"] = JOB_STATE_DONE
                doc["result"] = {"loss": -row["ndcg@10"], "status": STATUS_OK}
            trials.insert_trial_docs(docs)
            trials.refresh()
            rows += batch
            print(name, len(trials), max(r["ndcg@10"] for r in rows), flush=True)


def tune(name, workers=1):
    rows = []
    base, common = Path("parameters") / f"{name}.yaml", SEARCH / "common.yaml"
    files = [str(base), str(common)]
    if workers > 1 and PLAN[name][0] == "bayes":
        batched(name, files, rows, workers)
    else:
        sequential(name, files, rows)

    TRIALS.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    trials.to_csv(TRIALS / f"{name}.csv", index=False)

    TUNED.mkdir(parents=True, exist_ok=True)
    keys = [line.split()[0] for line in (SEARCH / f"{name}.hyper").open()]
    best = {k: rows[trials.index[0]][k] for k in keys}
    config = (
        yaml.safe_load(base.read_text()) | yaml.safe_load(common.read_text()) | best
    )
    (TUNED / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        name,
        best,
        round(trials["ndcg@10"].iloc[0], 4),
        "capped",
        trials["capped"].sum(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("models", nargs="*", default=list(PLAN))
    args = parser.parse_args()
    for name in args.models:
        tune(name, args.workers)
