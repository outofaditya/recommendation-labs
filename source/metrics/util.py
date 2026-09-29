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
        self.close()

    def load_predictions(self, npz_filename, memory_map: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads the users, items, and scores from the NPZ archive."""
        filepath = self.base_dir / "scores" / npz_filename

        if not filepath.exists():
            raise FileNotFoundError(f"Prediction file not found: {filepath}")

        # Performance: mmap_mode='r' reads data from disk on-demand rather than loading
        # massive matrices directly into RAM.
        # mmap = 'r' if memory_map else None
        mmap = None

        try:
            self._npz_file = np.load(filepath, allow_pickle=False, mmap_mode=mmap)

            # The keys typically exclude the .npy extension when accessed via NpzFile,
            # but sometimes generators explicitly name them with the extension.
            # We check both to be safe.
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

    def close(self):
        """Releases file handles, essential when using memory mapping."""
        if self._npz_file is not None:
            self._npz_file.close()
