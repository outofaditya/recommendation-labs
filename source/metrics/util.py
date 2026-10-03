"""Task 2.1"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class RecommenderResultLoader:
    """Load one export directory, closing every NPZ after reading its arrays."""

    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def load_predictions(self, npz_filename):
        path = self.base_dir / "scores" / npz_filename
        with np.load(path, allow_pickle=False) as archive:
            users = archive["users"].astype(str)
            items = archive["items"].astype(str)
            scores = archive["scores"]
        if (
            users.ndim != 1
            or items.ndim != 1
            or scores.shape != (len(users), len(items))
        ):
            raise ValueError(f"Invalid score matrix or ID arrays: {path}")
        if len(set(users)) != len(users) or len(set(items)) != len(items):
            raise ValueError(f"Duplicate IDs: {path}")
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError(f"NaN or positive infinity: {path}")
        # Candidate-restricted hybrids may use -inf outside their pool.
        if self.base_dir.name != "hybrid" and not np.isfinite(scores).all():
            raise ValueError(f"Non-finite individual model scores: {path}")
        return users, items, scores

    def load_metrics(self, json_filename):
        return json.loads((self.base_dir / "metrics" / json_filename).read_text())

    def load_split_data_file(self, file_name="test.tsv"):
        return (
            pd.read_csv(self.base_dir / "split" / file_name, sep="\t", dtype=str)
            .iloc[:, :2]
            .to_numpy()
        )
