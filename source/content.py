import numpy as np
import pandas as pd
from source.hybrids.side import atoms


# genre indicators and a weighted release decade per movie scaled to unit length
def features(items, decade=0.0, idf=False):
    movies = atoms("item", items)
    genres = movies["class:token_seq"].str.get_dummies(sep=" ").to_numpy(float)
    year = pd.to_numeric(movies["release_year:token"], errors="coerce")
    decades = pd.get_dummies(year // 10).to_numpy(float) * decade
    matrix = np.hstack([genres, decades])
    if idf:
        matrix *= np.log(len(matrix) / np.maximum((matrix > 0).sum(axis=0), 1))
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


# cosine to the profile of seen movies or summed cosine to the k closest seen movies
def predict(train, matrix, method="profile", k=None):
    train = train.astype(np.float32)
    matrix = matrix.astype(np.float32)
    if method == "profile":
        profile = train @ matrix
        profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
        return profile @ matrix.T
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, 0)
    if k:
        drop = np.argpartition(-similarity, k, axis=1)[:, k:]
        np.put_along_axis(similarity, drop, 0, axis=1)
    return train @ similarity.T
