"""Export the RecBole train/validation/test split without training a model.

This follows the corrected save_split implementation from the course
RecBole_DSAIT4335 repository.  It creates the RecBole dataloaders with the
configured seed and writes original user/item tokens to TSV files.
"""

import argparse
from pathlib import Path

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


def dump_interactions(data_loader, dataset, output):
    uid_field, iid_field = dataset.uid_field, dataset.iid_field
    inter_feat = data_loader.dataset.inter_feat
    frame = pd.DataFrame(
        {
            uid_field: dataset.id2token(uid_field, inter_feat[uid_field].numpy()),
            iid_field: dataset.id2token(iid_field, inter_feat[iid_field].numpy()),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)
    print(f"Saved {len(frame)} interactions -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("parameters/tuned/BPR.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/split"))
    parser.add_argument("--seed", type=int, help="Override the config seed")
    args = parser.parse_args()

    overrides = {
        "checkpoint_dir": str(Path("results/split_dataloader_cache")),
        "save_dataloaders": False,
    }
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = Config(config_file_list=[str(args.config)], config_dict=overrides)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    for name, data_loader in zip(("train", "valid", "test"), (train_data, valid_data, test_data), strict=True):
        dump_interactions(data_loader, dataset, args.output / f"{name}.tsv")
    print(f"Split seed: {config['seed']}; output: {args.output}")


if __name__ == "__main__":
    main()
