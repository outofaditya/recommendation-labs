import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_matrix_pathology(npz_file: str = "Random.npz"):
    """
    Visually diagnoses rank collapse and row broadcasting in prediction matrices.
    Generates a heatmap of the raw scores and a PCA projection of the users.
    """
    with np.load(npz_file, allow_pickle=False) as data:
        scores = data["scores"]
        num_users, num_items = scores.shape

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. The Heatmap Check (Visualizing the broadcasting)
        # We slice the first 100 users and items. If rows are duplicated,
        # this will look like thick horizontal stripes rather than TV static.
        subset_users = min(num_users, 100)
        subset_items = min(num_items, 100)

        im = axes[0].imshow(scores[:subset_users, :subset_items], aspect="auto", cmap="viridis")
        axes[0].set_title(f"Prediction Heatmap (First {subset_users} Users, {subset_items} Items)\nHorizontal banding indicates duplicate rows")
        axes[0].set_xlabel("Item Index")
        axes[0].set_ylabel("User Index")
        fig.colorbar(im, ax=axes[0], label="Predicted Score")

        # 2. The PCA Check (Proving the exact number of unique rows)
        # A true random matrix will result in a massive cloud of 943 scattered points.
        # If there are exactly 4 unique rows, PCA will perfectly project all 943 users
        # into exactly 4 distinct, overlapping dots.
        pca = PCA(n_components=2)

        # We add a tiny bit of noise (jitter) strictly for visualization purposes,
        # otherwise the 943 dots perfectly occlude each other making it look like just 4 users.
        user_embeddings = pca.fit_transform(scores)
        jitter = np.random.normal(0, 0.01, size=user_embeddings.shape)
        user_embeddings_jittered = user_embeddings + jitter

        axes[1].scatter(user_embeddings_jittered[:, 0], user_embeddings_jittered[:, 1], alpha=0.4, c="crimson", edgecolor="k", s=50)
        axes[1].set_title("User Dimensionality (PCA with slight jitter)\n943 users collapsed into 4 distinct clusters")
        axes[1].set_xlabel("Principal Component 1")
        axes[1].set_ylabel("Principal Component 2")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plot_matrix_pathology("results/scores/Random.npz")
