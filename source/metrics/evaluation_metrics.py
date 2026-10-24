"""Pointwise errors for explicit rating predictions on a common target scale.

Callers must supply finite, aligned rating predictions on the same observed
pairs. Raw implicit-feedback ranking scores are not valid inputs; Task 2.2
therefore does not call these metrics.
"""

import numpy as np


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
