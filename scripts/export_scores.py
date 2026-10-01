import json
import logging
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data.interaction import Interaction
from recbole.data import create_dataset, data_preparation
from source.data import RESULTS
from recbole.utils import init_seed, get_model, get_trainer

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def save_split(dataset, loaders):
    folder = RESULTS / "split"
    folder.mkdir(parents=True, exist_ok=True)
    uid, iid = dataset.uid_field, dataset.iid_field
    for name, loader in zip(["train", "valid", "test"], loaders):
        inter = loader.dataset.inter_feat
        frame = pd.DataFrame(
            {
                uid: dataset.id2token(uid, inter[uid].numpy()),
                iid: dataset.id2token(iid, inter[iid].numpy()),
            }
        )
        path = folder / f"{name}.tsv"
        if path.exists():
            saved = pd.read_csv(path, sep="\t", dtype=str)
            assert saved.equals(frame.astype(str)), f"{name} split differs"
        else:
            frame.to_csv(path, sep="\t", index=False)


def batch_scores(model, dataset, batch, device):
    try:
        inter = Interaction({dataset.uid_field: batch.to(device)})
        return model.full_sort_predict(inter).view(len(batch), -1)
    except NotImplementedError:
        items = torch.arange(dataset.item_num)
        inter = Interaction(
            {
                dataset.iid_field: items.repeat(len(batch)).to(device),
                dataset.uid_field: batch.repeat_interleave(len(items)).to(device),
            }
        )
        return model.predict(inter).view(len(batch), -1)


@torch.no_grad()
def full_scores(model, dataset, device):
    model.eval()
    users = torch.arange(1, dataset.user_num)
    rows = [batch_scores(model, dataset, b, device).cpu() for b in users.split(256)]
    return users, torch.cat(rows)[:, 1:].float().numpy()


def load(name, folder="parameters", overrides=None):
    path = Path(folder) / f"{name}.yaml"
    path = path if path.exists() else Path("parameters") / path.name
    checkpoints = {"checkpoint_dir": str(RESULTS / "checkpoints")}
    return Config(
        config_file_list=[str(path)], config_dict=checkpoints | (overrides or {})
    )


def fit(config, saved=True):
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    loaders = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, loaders[0].dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    _, valid = trainer.fit(loaders[0], loaders[1], saved=saved, show_progress=False)
    return dataset, loaders, model, trainer, valid, len(trainer.train_loss_dict)


def export(name, tuned=False):
    config = load(name, "parameters/tuned" if tuned else "parameters")
    dataset, loaders, model, trainer, best_valid, epochs = fit(config)
    save_split(dataset, loaders)
    test_result = trainer.evaluate(loaders[2], load_best_model=True)

    users, scores = full_scores(model, dataset, config["device"])
    items = np.arange(1, dataset.item_num)
    out = RESULTS / ("tuned" if tuned else "")
    folder = out / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{name}.npz",
        scores=scores,
        users=dataset.id2token(dataset.uid_field, users.numpy()),
        items=dataset.id2token(dataset.iid_field, items),
    )

    metrics = {"epochs": epochs, "valid": best_valid, "test": dict(test_result)}
    folder = out / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    print(name, metrics["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    export(parser.parse_args().model)
