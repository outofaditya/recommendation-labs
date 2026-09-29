import time
import argparse
from pathlib import Path
from functools import partial
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
from recbole.trainer.hyper_tuning import _spacesize, exhaustive_search
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


# grids queue every point before training while tpe learns after each batch
def search(name, files, rows, workers):
    space = HyperTuning._build_space_from_file(SEARCH / f"{name}.hyper")
    algo, evals = PLAN[name]
    grid = algo == "exhaustive"
    suggest = partial(exhaustive_search, nbMaxSucessiveFailures=1000)
    suggest, evals = (suggest, _spacesize(space)) if grid else (tpe.suggest, evals)
    threads = max(1, torch.get_num_threads() // workers)
    trials, rng = Trials(), np.random.RandomState(2020)
    domain = Domain(lambda params: 0, space)
    context = get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        while len(trials) < evals:
            docs = []
            for _ in range(min(workers, evals - len(trials))):
                seed = rng.randint(2**31 - 1)
                docs += suggest(trials.new_trial_ids(1), domain, trials, seed)
                if grid:
                    trials.insert_trial_docs(docs[-1:])
                    trials.refresh()
            vals = [{k: v[0] for k, v in d["misc"]["vals"].items() if v} for d in docs]
            params = [space_eval(space, v) for v in vals]
            jobs = [files] * len(docs), params, [threads] * len(docs)
            batch = list(pool.map(trial, *jobs))
            for doc, row in zip(docs, batch):
                doc["state"] = JOB_STATE_DONE
                doc["result"] = {"loss": -row["ndcg@10"], "status": STATUS_OK}
            if not grid:
                trials.insert_trial_docs(docs)
                trials.refresh()
            rows += batch
            print(name, len(rows), max(r["ndcg@10"] for r in rows), flush=True)


def tune(name, workers=1):
    rows = []
    base, common = Path("parameters") / f"{name}.yaml", SEARCH / "common.yaml"
    files = [str(base), str(common)]
    search(name, files, rows, workers)

    TRIALS.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False, kind="stable")
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
