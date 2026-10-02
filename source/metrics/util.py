import os
import numpy as np
import json
from pathlib import Path
from typing import Any
from numpy.lib.npyio import NpzFile


class RecommenderResultLoader:
    """
    A resource-managed loader for recommender system evaluation outputs.
    Supports lazy loading of large recommendation matrices.
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self._npz_file: NpzFile | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Releases file handles"""
        if self._npz_file is not None:
            self._npz_file.close()

    def load_predictions(self, npz_filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads the users, items, and scores from the NPZ archive."""
        filepath = self.base_dir / "scores" / npz_filename
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prediction file not found: {filepath}")

        try:
            self._npz_file = np.load(filepath, allow_pickle=False)

            assert self._npz_file is not None, "._npz_file was none when it should have been leaded"
            users = self._npz_file["users"]
            items = self._npz_file["items"]
            scores = self._npz_file["scores"]

            if not (users.shape[0] == items.shape[0] == scores.shape[0] or scores.shape == (len(users), len(items))):
                raise ValueError(f"Shape mismatch: users {users.shape}, items {items.shape}, scores {scores.shape}")

            return users, items, scores

        except KeyError as e:
            raise KeyError(
                f"Expected array missing from NPZ archive: {e}. Available keys: {self._npz_file.files if self._npz_file else '_npz_file was none'}"
            )
        except OSError as e:
            raise RuntimeError(f"Failed to load numpy arrays safely: {e}")

    def load_metrics(self, json_filename) -> dict[str, Any]:
        """
        Loads pre-calculated metrics from a JSON file.
        """
        filepath = self.base_dir / "metrics" / json_filename

        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in metrics file: {e}")

    def load_split_data_file(self, file_name: str = "test.tsv"):
        path = Path(self.base_dir, "split", file_name)
        assert os.path.exists(path), f"path: {path} does not exist"
        return np.genfromtxt(fname=path, delimiter="\t", skip_header=1, filling_values=-1)

    def load_interaction_ratings(self, file_path: Path = Path("data/ml-100k/ml-100k.inter")) -> dict[tuple[int, int], float]:
        if not file_path.exists():
            raise FileNotFoundError(f"Interaction file not found: {file_path}")

        ratings_map = {}
        with open(file_path, "r") as f:
            header = f.readline().strip().split("\t")
            try:
                u_idx = next(i for i, col in enumerate(header) if "user" in col.lower())
                i_idx = next(i for i, col in enumerate(header) if "item" in col.lower())
                r_idx = next(i for i, col in enumerate(header) if "rating" in col.lower())
            except StopIteration:
                raise RuntimeError(f"Couldnt load the correct headers, found: {header}")

            for line_num, line in enumerate(f, start=2):
                parts = line.strip().split("\t")
                if len(parts) > max(u_idx, i_idx, r_idx):
                    try:
                        u = int(parts[u_idx])
                        i = int(parts[i_idx])
                        r = float(parts[r_idx])
                        ratings_map[(u, i)] = r
                    except ValueError:
                        raise ValueError(f"Failed to parse interaction row {line_num}: {line}")

        return ratings_map
