import logging
import argparse
import warnings
from pathlib import Path

import yaml
import pandas as pd
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

SEARCH = Path("parameters/search")
TUNED = Path("parameters/tuned")
TRIALS = Path("results/tuning")
MODELS = [
    "EASE",
    "ItemKNN",
    "UserKNN",
    "SLIMElastic",
    "BPR",
    "NeuMF",
    "FISM",
    "NGCF",
    "LightGCN",
]
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def tune(name):
    rows = []

    def objective(config_dict, config_file_list, saved=True):
        result = objective_function(config_dict, config_file_list, saved)
        rows.append({**config_dict, **result["best_valid_result"]})
        return result

    base = Path("parameters") / f"{name}.yaml"
    common = SEARCH / "common.yaml"
    tuner = HyperTuning(
        objective,
        algo="exhaustive",
        early_stop=10**6,
        params_file=str(SEARCH / f"{name}.hyper"),
        fixed_config_file_list=[str(base), str(common)],
    )
    tuner.run()

    TRIALS.mkdir(parents=True, exist_ok=True)
    trials = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False)
    trials.to_csv(TRIALS / f"{name}.csv", index=False)

    TUNED.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(base.read_text())
    config |= yaml.safe_load(common.read_text()) | tuner.best_params
    (TUNED / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(name, tuner.best_params, round(trials["ndcg@10"].iloc[0], 4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=MODELS)
    for name in parser.parse_args().models:
        tune(name)
