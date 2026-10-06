import os
import time
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import get_context
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import yaml
import torch
import numpy as np
import pandas as pd
from export_scores import fit
from source.data import RESULTS
from hyperopt.base import Domain
from recbole.config import Config
from recbole.trainer import HyperTuning
from recbole.trainer.hyper_tuning import _spacesize, exhaustive_search
from hyperopt import tpe, Trials, STATUS_OK, JOB_STATE_DONE, space_eval

SEARCH = Path("parameters/search")
TUNED = Path("parameters/tuned")
TRIALS = RESULTS / "tuning"
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


def plain(value):
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value.item() if hasattr(value, "item") else value


def trial(files, params, threads):
    torch.set_num_threads(threads)
    params = {k: plain(v) for k, v in params.items()}
    config = Config(config_file_list=files, config_dict=params)
    start = time.perf_counter()
    *_, valid, epochs = fit(config, saved=False)
    seconds = round(time.perf_counter() - start)
    capped = epochs >= config["epochs"]
    return params | valid | {"epochs": epochs, "capped": capped, "seconds": seconds}


# a free worker takes the next point at once and tpe learns from every finished trial
def search(name, files, workers):
    space = HyperTuning._build_space_from_file(SEARCH / f"{name}.hyper")
    algo, evals = PLAN[name]
    grid = algo == "exhaustive"
    suggest = partial(exhaustive_search, nbMaxSucessiveFailures=1000)
    suggest, evals = (suggest, _spacesize(space)) if grid else (tpe.suggest, evals)
    trials, rng = Trials(), np.random.RandomState(2020)
    domain = Domain(lambda params: 0, space)
    threads = max(1, torch.get_num_threads() // workers)
    rows, running = {}, {}
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        while len(rows) < evals:
            for _ in range(min(workers, evals - len(rows)) - len(running)):
                seed = rng.randint(2**31 - 1)
                [doc] = suggest(trials.new_trial_ids(1), domain, trials, seed)
                # grids see pending points so no point is drawn twice
                if grid:
                    trials.insert_trial_docs([doc])
                    trials.refresh()
                vals = {k: v[0] for k, v in doc["misc"]["vals"].items() if v}
                job = pool.submit(trial, files, space_eval(space, vals), threads)
                running[job] = doc
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for job in sorted(done, key=lambda job: running[job]["tid"]):
                doc = running.pop(job)
                row = rows[doc["tid"]] = job.result()
                doc["state"] = JOB_STATE_DONE
                doc["result"] = {"loss": -row["ndcg@10"], "status": STATUS_OK}
                if not grid:
                    trials.insert_trial_docs([doc])
                    trials.refresh()
            best = max(row["ndcg@10"] for row in rows.values())
            print(name, len(rows), best, flush=True)
    return [rows[tid] for tid in sorted(rows)]


def tune(name, workers):
    base, common = Path("parameters") / f"{name}.yaml", SEARCH / "common.yaml"
    files = [str(base), str(common)]
    rows = search(name, files, workers)

    TRIALS.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False, kind="stable")
    trials.to_csv(TRIALS / f"{name}.csv", index=False)

    TUNED.mkdir(parents=True, exist_ok=True)
    keys = [line.split()[0] for line in (SEARCH / f"{name}.hyper").open()]
    best = {k: rows[trials.index[0]][k] for k in keys}
    config = yaml.safe_load(base.read_text()) | yaml.safe_load(common.read_text()) | best
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
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("models", nargs="*", default=list(PLAN))
    args = parser.parse_args()
    for name in args.models:
        tune(name, args.workers)
