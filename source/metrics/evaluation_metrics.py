"""Metrics for evaluating predicted ratings on observed user-item pairs.

These metrics are deliberately separate from the top-k ranking metrics. A
prediction is evaluated against the original rating in ``ml-100k.inter``;
hidden train/validation interactions are not used as targets.
"""

from pathlib import Path

import numpy as np
import pandas as pd


class EvaluationMetrics:
    """Calculate pointwise rating-prediction errors."""

    @staticmethod
    def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        true = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        if true.shape != pred.shape:
            raise ValueError(f"True and predicted scores must have the same shape: {true.shape} != {pred.shape}")
        if true.size == 0:
            raise ValueError("At least one finite prediction is required")
        if not (np.isfinite(true).all() and np.isfinite(pred).all()):
            raise ValueError("MAE/MSE inputs must be finite")
        return true, pred

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error; lower is better."""
        true, pred = EvaluationMetrics._validate_arrays(y_true, y_pred)
        return float(np.mean(np.abs(true - pred)))

    @staticmethod
    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error; lower is better."""
        true, pred = EvaluationMetrics._validate_arrays(y_true, y_pred)
        return float(np.mean(np.square(true - pred)))

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root mean squared error; lower is better."""
        return float(np.sqrt(EvaluationMetrics.calculate_mse(y_true, y_pred)))

    @staticmethod
    def load_interaction_ratings(path: Path = Path("data/ml-100k/ml-100k.inter")) -> dict[tuple[str, str], float]:
        """Load original explicit ratings keyed by string user/item IDs."""
        if not path.exists():
            raise FileNotFoundError(f"Interaction file not found: {path}")
        frame = pd.read_csv(path, sep="\t", dtype=str)
        frame.columns = [column.split(":", 1)[0] for column in frame.columns]
        required = {"user_id", "item_id", "rating"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Interaction file is missing columns: {sorted(missing)}")
        ratings = frame[["user_id", "item_id", "rating"]].copy()
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise")
        if ratings.duplicated(["user_id", "item_id"]).any():
            raise ValueError("Interaction file contains duplicate user-item ratings")
        return {(row.user_id, row.item_id): float(row.rating) for row in ratings.itertuples(index=False)}

    @classmethod
    def calculate_test_metrics(
        cls,
        users: np.ndarray,
        items: np.ndarray,
        scores: np.ndarray,
        test_pairs: pd.DataFrame,
        ratings: dict[tuple[str, str], float],
    ) -> tuple[dict[str, float], dict[str, float | int]]:
        """Calculate MAE/MSE for finite predictions on test interactions.

        Hybrid exports may intentionally contain ``-inf`` outside their
        candidate pool. Such pairs are reported as unavailable rather than
        turning ``-inf`` into an artificial rating error.
        """
        user_index = {str(value): index for index, value in enumerate(users)}
        item_index = {str(value): index for index, value in enumerate(items)}
        true_values, predicted_values = [], []
        missing_ratings = 0
        unmapped_pairs = 0
        nonfinite_predictions = 0
        for row in test_pairs.itertuples(index=False):
            user, item = str(row[0]), str(row[1])
            if (user, item) not in ratings:
                missing_ratings += 1
                continue
            if user not in user_index or item not in item_index:
                unmapped_pairs += 1
                continue
            prediction = float(scores[user_index[user], item_index[item]])
            if not np.isfinite(prediction):
                nonfinite_predictions += 1
                continue
            true_values.append(ratings[(user, item)])
            predicted_values.append(prediction)

        if not predicted_values:
            raise ValueError("No finite test-pair predictions are available for MAE/MSE")
        true_array = np.asarray(true_values, dtype=float)
        prediction_array = np.asarray(predicted_values, dtype=float)
        metrics = {
            "mae": cls.calculate_mae(true_array, prediction_array),
            "mse": cls.calculate_mse(true_array, prediction_array),
        }
        diagnostics = {
            "test_pairs": len(test_pairs),
            "rated_test_pairs": len(test_pairs) - missing_ratings,
            "predicted_test_pairs": len(predicted_values),
            "missing_ratings": int(missing_ratings),
            "unmapped_pairs": int(unmapped_pairs),
            "nonfinite_predictions": int(nonfinite_predictions),
            "prediction_coverage": float(len(predicted_values) / len(test_pairs)) if len(test_pairs) else 0.0,
        }
        return metrics, diagnostics
