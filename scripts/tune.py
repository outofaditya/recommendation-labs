import os
import time
import argparse
from pathlib import Path

import yaml
import pandas as pd
from export_scores import fit
from recbole.config import Config
from recbole.trainer import HyperTuning

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


def tune(name):
    rows = []

    def objective(config_dict, config_file_list, saved=False):
        config_dict = {k: plain(v) for k, v in config_dict.items()}
        config = Config(config_file_list=config_file_list, config_dict=config_dict)
        start = time.perf_counter()
        *_, valid, epochs = fit(config, saved=False)
        seconds = round(time.perf_counter() - start)
        capped = epochs >= config["epochs"]
        rows.append(
            config_dict
            | valid
            | {"epochs": epochs, "capped": capped, "seconds": seconds}
        )
        return {
            "model": config["model"],
            "best_valid_score": valid["ndcg@10"],
            "valid_score_bigger": True,
            "best_valid_result": valid,
            "test_result": {},
        }

    algo, evals = PLAN[name]
    base = Path("parameters") / f"{name}.yaml"
    common = SEARCH / "common.yaml"
    tuner = HyperTuning(
        objective,
        algo=algo,
        max_evals=evals,
        early_stop=10**6,
        params_file=str(SEARCH / f"{name}.hyper"),
        fixed_config_file_list=[str(base), str(common)],
    )
    tuner.run()

    TRIALS.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    trials.to_csv(TRIALS / f"{name}.csv", index=False)

    TUNED.mkdir(parents=True, exist_ok=True)
    best = {k: plain(v) for k, v in tuner.best_params.items()}
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
    parser.add_argument("models", nargs="*", default=list(PLAN))
    for name in parser.parse_args().models:
        tune(name)
