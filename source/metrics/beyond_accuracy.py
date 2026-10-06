import numpy as np
from collections.abc import Callable

from source.metrics.util import RecommenderDataLoader


def jaccard_distance(genres_a: list[str], genres_b: list[str]) -> float:
    """Jaccard Similarity. Returns 0.0 if identical, 1.0 if completely disjoint."""
    set_a, set_b = set(genres_a), set(genres_b)
    union_len = len(set_a | set_b)
    if union_len == 0:
        return 0.0  # Both items have no genres
    return 1.0 - (len(set_a & set_b) / union_len)


class BeyondAccuracyMetrics:
    @staticmethod
    def build_genre_distance_matrix(
        internal_item_features: list[list[str]], num_items: int, distance_func: Callable[[list[str], list[str]], float]
    ) -> np.ndarray:
        """
        Precomputes the item-to-item distance matrix.
        Takes a list of features already aligned to the internal matrix indices.
        """
        distance_matrix = np.zeros((num_items, num_items), dtype=np.float32)

        for i in range(num_items):
            for j in range(i + 1, num_items):
                dist = distance_func(internal_item_features[i], internal_item_features[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    @staticmethod
    def calculate_ild(top_k_items: np.ndarray, distance_matrix: np.ndarray) -> float:
        """Intra-list Diversity (ILD@K): Average pairwise distance between recommended items."""
        _, k = top_k_items.shape
        if k <= 1:
            return 0.0

        k_distances = distance_matrix[top_k_items[:, :, None], top_k_items[:, None, :]]
        sum_distances = k_distances.sum(axis=(1, 2)) - np.trace(k_distances, axis1=1, axis2=2)
        user_ild = sum_distances / (k * (k - 1))

        return float(np.mean(user_ild))

    @staticmethod
    def calculate_novelty(top_k_items: np.ndarray, item_probabilities: np.ndarray) -> float:
        """Novelty@K (Self-Information): Evaluates how unexpected the recommendations are globally."""
        safe_probs = np.clip(item_probabilities, 1e-9, 1.0)
        self_information = -np.log2(safe_probs)

        user_novelty = np.mean(self_information[top_k_items], axis=1)
        return float(np.mean(user_novelty))

    @staticmethod
    def calculate_serendipity(
        top_k_items: np.ndarray, hits: np.ndarray, user_history: list[np.ndarray], distance_matrix: np.ndarray, alpha: float = 1.0
    ) -> float:
        """
        Serendipity@K: Measures recommendations that are BOTH relevant (Hits) AND surprising
        (distant from the user's historical profile).
        """
        num_users, _ = top_k_items.shape
        serendipity_scores = np.zeros(num_users)

        for u in range(num_users):
            history = user_history[u]
            if len(history) == 0:
                continue

            distances_to_history = distance_matrix[np.ix_(top_k_items[u], history)]
            unexpectedness = np.mean(distances_to_history, axis=1)
            user_serendipity = hits[u] * (unexpectedness**alpha)
            serendipity_scores[u] = np.mean(user_serendipity)

        return float(np.mean(serendipity_scores))

    @staticmethod
    def calculate_coverage(top_k_items: np.ndarray, total_catalog_size: int) -> float:
        """Catalog Coverage: Proportion of the total catalog that appears in any Top-K list."""
        unique_recommended = np.unique(top_k_items)
        return float(len(unique_recommended) / total_catalog_size)

    @classmethod
    def calculate_all_metrics(
        cls,
        scores: np.ndarray,
        hidden: np.ndarray,
        relevant: np.ndarray,
        distance_matrix: np.ndarray,
        k: int = 10,
    ) -> dict[str, float]:
        """Evaluate beyond-accuracy metrics on aligned score and mask arrays."""
        if k < 1:
            raise ValueError("k must be positive")
        if scores.shape != hidden.shape or scores.shape != relevant.shape:
            raise ValueError("scores, hidden and relevant must have the same shape")
        if distance_matrix.shape != (scores.shape[1], scores.shape[1]):
            raise ValueError("distance matrix must be square and aligned with score items")

        masked_scores = np.where(hidden, -np.inf, scores)
        top_k_items = np.argsort(-masked_scores, axis=1, kind="stable")[:, :k]
        hits = np.take_along_axis(relevant, top_k_items, axis=1)

        valid_users = relevant.sum(axis=1) > 0
        top_k_items = top_k_items[valid_users]
        hits = hits[valid_users]
        if not len(top_k_items):
            raise ValueError("No users with relevant test interactions")

        user_history = [np.flatnonzero(hidden[u]) for u in np.flatnonzero(valid_users)]
        item_interaction_counts = hidden.sum(axis=0)
        total_interactions = item_interaction_counts.sum()
        item_probabilities = (
            item_interaction_counts / total_interactions
            if total_interactions > 0
            else np.zeros(scores.shape[1])
        )

        return {
            f"coverage@{k}": cls.calculate_coverage(top_k_items, scores.shape[1]),
            f"novelty@{k}": cls.calculate_novelty(top_k_items, item_probabilities),
            f"ild@{k}": cls.calculate_ild(top_k_items, distance_matrix),
            f"serendipity@{k}": cls.calculate_serendipity(
                top_k_items, hits, user_history, distance_matrix, alpha=1.0
            ),
        }

    @classmethod
    def evaluate_file(cls, prediction_file: str = "Random.npz", k: int = 10) -> dict[str, float]:
        with RecommenderDataLoader(base_dir="results") as loader:
            users, items, predicted_scores = loader.load_predictions(npz_filename=prediction_file)
            num_users, num_items = predicted_scores.shape

            item_map = loader.load_item_info()

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

        internal_item_features = [[] for _ in range(num_items)]
        for raw_id, idx in item_id2idx.items():
            if raw_id in item_map:
                internal_item_features[idx] = item_map[raw_id][2]

        distance_matrix = cls.build_genre_distance_matrix(
            internal_item_features=internal_item_features, num_items=num_items, distance_func=jaccard_distance
        )

        return cls.calculate_all_metrics(
            predicted_scores, hidden_mask, relevant_mask, distance_matrix, k
        )


if __name__ == "__main__":
    print(BeyondAccuracyMetrics.evaluate_file(prediction_file="Random.npz"))
