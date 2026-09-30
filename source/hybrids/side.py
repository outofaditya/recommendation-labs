import numpy as np
import pandas as pd
from source.data import RESULTS, scores

K = 50
DATA = "data/ml-100k/ml-100k"


def atoms(kind, tokens):
    frame = pd.read_csv(f"{DATA}.{kind}", sep="\t", dtype=str)
    return frame.set_index(f"{kind}_id:token").loc[tokens]


# genres and release year per movie then age gender and occupation per user
def side():
    _, users, items = scores("Pop")
    movies, people = atoms("item", items), atoms("user", users)
    genres = movies["class:token_seq"].str.get_dummies(sep=" ")
    year = pd.to_numeric(movies["release_year:token"], errors="coerce")
    occupation = people["occupation:token"].astype("category").cat.codes
    person = [people["age:token"].astype(float), people["gender:token"] == "M"]
    movie = np.column_stack([genres, year]).astype(float)
    return np.column_stack(person + [occupation]).astype(float), movie


# the saved user embeddings that rebuild the tuned score table
def embeddings(name):
    table = scores(name)[0]
    for path in (RESULTS / "checkpoints").glob(f"*-{name}-*-embeddings.npz"):
        saved = np.load(path)
        users, items = saved["user_embeddings"][1:], saved["item_embeddings"][1:]
        if np.allclose(users @ items.T, table, atol=1e-4):
            return users
    raise FileNotFoundError(f"no {name} embeddings rebuild its scores")


# user neighbourhood as in recbole whose padding row decides ties
def userknn(vectors, train, k=K):
    vectors = np.pad(vectors.astype(np.float32), ((1, 0), (0, 0)))
    norms = np.sqrt((vectors * vectors).sum(axis=1))
    similarity = vectors @ vectors.T * (1 / (np.outer(norms, norms) + 1e-6))
    np.fill_diagonal(similarity, 0)
    keep = np.argpartition(-similarity, k - 1, axis=1)[1:, :k]
    weights = np.zeros(similarity[1:].shape, dtype=np.float32)
    np.put_along_axis(weights, keep, np.take_along_axis(similarity[1:], keep, 1), 1)
    return weights[:, 1:] @ train.astype(np.float32)
