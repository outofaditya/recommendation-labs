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
from recbole.utils import init_seed, get_model, get_trainer

RESULTS = Path("results")
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


def export(name, folder="parameters"):
    path = Path(folder) / f"{name}.yaml"
    config = Config(
        config_file_list=[
            str(path if path.exists() else Path("parameters") / path.name)
        ],
        config_dict={"checkpoint_dir": str(RESULTS / "checkpoints")},
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    loaders = data_preparation(config, dataset)
    train, valid, test = loaders
    save_split(dataset, loaders)

    model = get_model(config["model"])(config, train.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    _, best_valid = trainer.fit(train, valid, show_progress=False)
    test_result = trainer.evaluate(test, load_best_model=True)

    users, scores = full_scores(model, dataset, config["device"])
    items = np.arange(1, dataset.item_num)
    folder = RESULTS / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{name}.npz",
        scores=scores,
        users=dataset.id2token(dataset.uid_field, users.numpy()),
        items=dataset.id2token(dataset.iid_field, items),
    )

    metrics = {"valid": best_valid, "test": dict(test_result)}
    folder = RESULTS / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    print(name, metrics["test"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    export(parser.parse_args().model)
