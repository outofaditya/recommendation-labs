import numpy as np
from collections.abc import Iterable
from util import RecommenderResultLoader


class EvaluationMetrics:
    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return float(np.mean(np.square(y_true - y_pred)))

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(EvaluationMetrics.calculate_mse(y_true, y_pred)))

    @classmethod
    def evaluate_file(
        cls, prediction_file: str = "Random.npz", metrics: Iterable[str] = ("MAE", "MSE", "RMSE"), apply_id_offset: bool = True
    ) -> dict[str, float]:
        with RecommenderResultLoader(base_dir="results") as loader:
            users, items, predicted_scores = loader.load_predictions(npz_filename=prediction_file)
            test_pairs = loader.load_split_data_file()
            ground_truth_map = loader.load_interaction_ratings()

        user_id2idx = {int(raw_id): idx for idx, raw_id in enumerate(users)}
        item_id2idx = {int(raw_id): idx for idx, raw_id in enumerate(items)}

        user_indices = []
        item_indices = []
        true_scores = []
        missing_pairs = 0
        unmapped_pairs = 0

        for row in test_pairs:
            u_raw, i_raw = int(row[0]), int(row[1])

            if (u_raw, i_raw) in ground_truth_map:
                if u_raw in user_id2idx and i_raw in item_id2idx:
                    user_indices.append(user_id2idx[u_raw])
                    item_indices.append(item_id2idx[i_raw])
                    true_scores.append(ground_truth_map[(u_raw, i_raw)])
                else:
                    unmapped_pairs += 1
            else:
                missing_pairs += 1

        if missing_pairs > 0:
            print(f"Warning: {missing_pairs} pairs from the test split were not found in the ground truth interactions file.")
        if unmapped_pairs > 0:
            print(f"Warning: {unmapped_pairs} pairs from the test split were not found in the prediction matrix.")

        user_indices = np.array(user_indices)
        item_indices = np.array(item_indices)
        true_scores = np.array(true_scores)

        max_user_idx, max_item_idx = predicted_scores.shape
        if np.any(user_indices >= max_user_idx) or np.any(user_indices < 0):
            raise IndexError(
                f"Mapped user indices out of bounds (0-{max_user_idx - 1}). Found min: {np.min(user_indices)}, max: {np.max(user_indices)}."
            )
        if np.any(item_indices >= max_item_idx) or np.any(item_indices < 0):
            raise IndexError(
                f"Mapped item indices out of bounds (0-{max_item_idx - 1}). Found min: {np.min(item_indices)}, max: {np.max(item_indices)}."
            )

        results = {}
        metric_map = {"MAE": cls.calculate_mae, "MSE": cls.calculate_mse, "RMSE": cls.calculate_rmse}
        relevant_predictions = predicted_scores[user_indices, item_indices]
        for metric_name in metrics:
            if metric_name in metric_map:
                results[metric_name] = metric_map[metric_name](true_scores, relevant_predictions)
            else:
                raise KeyError(f"Unsupported metric: '{metric_name}'")

        return results


print(EvaluationMetrics.evaluate_file(prediction_file="EASE.npz"))
