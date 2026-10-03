"""Task 2.1"""
import numpy as np


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
        relevance = np.array([1 if item in true_items else 0 for item in top_k_items])
        dcg = np.sum(relevance / np.log2(np.arange(2, len(top_k_items) + 2)))
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
        return float(sum_precs / min(len(true_items), len(top_k_items)))
