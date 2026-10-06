import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from source.metrics.util import RecommenderDataLoader


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

    @staticmethod
    def calculate_all_metrics(scores: np.ndarray, hidden: np.ndarray, relevant: np.ndarray, k: int = 10) -> dict[str, float]:
        masked = np.where(hidden, -np.inf, scores)  # Replace historical item scores with -infinity
        ranked = np.argsort(-masked, axis=1, kind="stable")[:, :k]
        hits = np.take_along_axis(relevant, ranked, axis=1)  # True if the ranked item is in the test set

        # Remove users users with no test data
        count = relevant.sum(axis=1)
        valid_users = count > 0
        hits = hits[valid_users]
        count = count[valid_users]

        found = hits.any(axis=1)
        hits_sum = hits.sum(axis=1)
        discount = 1 / np.log2(np.arange(2, k + 2))
        ideal = np.cumsum(discount)[np.minimum(count, k) - 1]
        cum_hits = hits.cumsum(axis=1)
        precisions_at_k = cum_hits / np.arange(1, k + 1)
        sum_precisdions = (precisions_at_k * hits).sum(axis=1)

        results = {
            f"hit@{k}": found,
            f"precision@{k}": hits_sum / k,
            f"recall@{k}": hits_sum / count,
            f"mrr@{k}": np.where(found, 1 / (hits.argmax(axis=1) + 1), 0),
            f"ndcg@{k}": (hits * discount).sum(axis=1) / ideal,
            f"f1@{k}": (2 * hits_sum) / (k + count),
            f"map@{k}": sum_precisdions / np.minimum(count, k),
        }

        return {name: float(values.mean()) for name, values in results.items()}

    @classmethod
    def _load_evaluation_data(cls, prediction_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal helper to load scores, align IDs, and build boolean matrices.
        Returns: (predicted_scores, hidden_mask, relevant_mask)
        """
        with RecommenderDataLoader(base_dir="results") as loader:
            users, items, predicted_scores = loader.load_predictions(npz_filename=prediction_file)
            num_users, num_items = predicted_scores.shape

            user_id2idx = {int(raw_id): idx for idx, raw_id in enumerate(users)}
            item_id2idx = {int(raw_id): idx for idx, raw_id in enumerate(items)}

            def build_dense_mask(filename: str) -> np.ndarray:
                mask = np.zeros((num_users, num_items), dtype=bool)
                try:
                    pairs = loader.load_split_data_file(filename)
                    for row in pairs:
                        if len(row) >= 2:
                            raw_u, raw_i = int(row[0]), int(row[1])
                            if raw_u in user_id2idx and raw_i in item_id2idx:
                                mask[user_id2idx[raw_u], item_id2idx[raw_i]] = True
                except (FileNotFoundError, AssertionError) as e:
                    print(f"Warning: {filename} skipped or failed. {e}")
                return mask

            relevant_mask = build_dense_mask("test.tsv")
            hidden_mask = build_dense_mask("train.tsv") | build_dense_mask("valid.tsv")

        return predicted_scores, hidden_mask, relevant_mask

    @classmethod
    def evaluate_file(cls, prediction_file: str = "Random.npz", k: int = 10) -> dict[str, float]:
        predicted_scores, hidden_mask, relevant_mask = cls._load_evaluation_data(prediction_file)
        return cls.calculate_all_metrics(scores=predicted_scores, hidden=hidden_mask, relevant=relevant_mask, k=k)

    @classmethod
    def plot_roc_curve(cls, prediction_file: str = "Random.npz"):
        """Plots a global ROC curve and calculates global AUC."""
        predicted_scores, hidden_mask, relevant_mask = cls._load_evaluation_data(prediction_file)

        unseen_mask = ~hidden_mask
        ground_truth_flat = relevant_mask[unseen_mask].astype(np.int8)
        scores_flat = predicted_scores[unseen_mask]

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
