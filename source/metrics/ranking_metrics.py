import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

from source.metrics.util import RecommenderResultLoader


class RankingMetrics:
    @staticmethod
    def calculate_hit(true_items: set[int], top_k_items: np.ndarray) -> float:
        """Hit@K: 1.0 if any true item is in the top K, else 0.0"""
        return 1.0 if any(item in true_items for item in top_k_items) else 0.0

    @staticmethod
    def calculate_precision(true_items: set[int], top_k_items: np.ndarray) -> float:
        """Precision@K: Proportion of recommended items that are relevant."""
        hits = sum(1 for item in top_k_items if item in true_items)
        return float(hits / len(top_k_items))

    @staticmethod
    def calculate_recall(true_items: set[int], top_k_items: np.ndarray) -> float:
        """Recall@K: Proportion of relevant items that were recommended."""
        if not true_items:
            return 0.0
        hits = sum(1 for item in top_k_items if item in true_items)
        return float(hits / len(true_items))

    @staticmethod
    def calculate_mrr(true_items: set[int], top_k_items: np.ndarray) -> float:
        """MRR@K: 1 / rank of the first relevant item."""
        for rank, item in enumerate(top_k_items, start=1):
            if item in true_items:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """F1-Score@K: Harmonic mean of Precision and Recall."""
        if precision + recall == 0.0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))

    @staticmethod
    def calculate_ndcg(true_items: set[int], top_k_items: np.ndarray) -> float:
        """NDCG@K: Position-discounted metric of relevance."""
        if not true_items:
            return 0.0

        # Binary relevance: 1 if hit, 0 if miss
        relevance = np.array([1 if item in true_items else 0 for item in top_k_items])

        # Discounted Cumulative Gain
        dcg = np.sum(relevance / np.log2(np.arange(2, len(top_k_items) + 2)))

        # Ideal DCG (if all relevant items were ranked at the very top)
        ideal_relevance = np.ones(min(len(true_items), len(top_k_items)))
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

        return float(dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def calculate_ap(true_items: set[int], top_k_items: np.ndarray) -> float:
        """
        Average Precision@K (AP@K): Averages the precision at each point a relevant item is found.
        MAP is the mean of this value across all users.
        """
        if not true_items:
            return 0.0

        hits = 0
        sum_precs = 0.0
        for i, item in enumerate(top_k_items, start=1):
            if item in true_items:
                hits += 1
                sum_precs += hits / i

        # Denominator can be len(true_items) or min(len(true_items), K) depending on strict formulation.
        # We use min() to not unfairly penalize users with more true items than K.
        return float(sum_precs / min(len(true_items), len(top_k_items)))

    @classmethod
    def evaluate_file(cls, prediction_file: str = "Random.npz", k: int = 10, apply_id_offset: bool = True) -> dict[str, float]:
        with RecommenderResultLoader(base_dir="results") as loader:
            _, _, predicted_scores = loader.load_predictions(npz_filename=prediction_file)
            test_pairs = loader.load_split_data_file()

        offset = 1 if apply_id_offset else 0
        user_test_items = defaultdict(set)
        for row in test_pairs:
            u, i = int(row[0]) - offset, int(row[1]) - offset
            user_test_items[u].add(i)

        metrics = defaultdict(list)
        num_items = predicted_scores.shape[1]

        for u, true_items in user_test_items.items():
            true_items = {i for i in true_items if 0 <= i < num_items}
            if not true_items:
                continue

            user_scores = predicted_scores[u]
            sorted_indices = np.argsort(user_scores)[::-1]
            top_k_items = sorted_indices[:k]

            # Compute standard metrics
            prec = cls.calculate_precision(true_items, top_k_items)
            rec = cls.calculate_recall(true_items, top_k_items)

            metrics["Hit"].append(cls.calculate_hit(true_items, top_k_items))
            metrics["Precision"].append(prec)
            metrics["Recall"].append(rec)
            metrics["F1"].append(cls.calculate_f1(prec, rec))
            metrics["MAP"].append(cls.calculate_ap(true_items, top_k_items))
            metrics["MRR"].append(cls.calculate_mrr(true_items, top_k_items))
            metrics["NDCG"].append(cls.calculate_ndcg(true_items, top_k_items))

        return {name: float(np.mean(values)) for name, values in metrics.items()}

    @classmethod
    def plot_roc_curve(cls, prediction_file: str = "Random.npz", apply_id_offset: bool = True):
        """Plots a global ROC curve and calculates global AUC"""
        with RecommenderResultLoader(base_dir="results") as loader:
            _, _, predicted_scores = loader.load_predictions(npz_filename=prediction_file)
            test_pairs = loader.load_split_data_file()

        offset = 1 if apply_id_offset else 0
        num_users, num_items = predicted_scores.shape

        ground_truth = np.zeros((num_users, num_items), dtype=np.int8)
        for row in test_pairs:
            u, i = int(row[0]) - offset, int(row[1]) - offset
            if 0 <= u < num_users and 0 <= i < num_items:
                ground_truth[u, i] = 1

        ground_truth_flat = ground_truth.flatten()
        scores_flat = predicted_scores.flatten()
        fpr, tpr, _ = roc_curve(ground_truth_flat, scores_flat)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("Global Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.show()


print(RankingMetrics.evaluate_file(prediction_file="EASE.npz"))
RankingMetrics.plot_roc_curve(prediction_file="EASE.npz")
