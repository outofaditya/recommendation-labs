# Report Notes

## Setup

We use the course RecBole fork pinned to `439c5a8` with NumPy below 2 and pandas below 3 for compatibility. Ray Tune is added since the fork imports it without declaring it. MovieLens 100K is taken from the course fork as RecBole's download returns 403.

## Split

Each user's interactions are split at random into 80% train, 10% valid and 10% test with seed 2020. The split is identical for every model and the exporter verifies it. Models rank all movies a user has not seen in train.

## Task 1.1 — Individual Models

The course helpers save only the top-K so our exporter saves every user and movie score for the hybrids.

### Random

Random scores every movie uniformly without training. RecBole shares one random vector per user batch which we keep as shipped.

### Pop

Pop counts how often each movie appears in train and divides by the highest count. Star values are ignored. Every user gets the same ranking minus their seen movies which makes it the popularity bias extreme and a stronger baseline than Random.

### ItemKNN

ItemKNN computes cosine similarity between movies from their train audiences and keeps the 100 nearest per movie. A movie's score is the sum of its similarities to the movies the user watched. No weights are learned and `k` is the main setting to tune.

### UserKNN

UserKNN runs RecBole's ItemKNN with `knn_method: 'user'`. It computes cosine similarity between users from their train histories and keeps the 100 nearest per user. A movie's score is the sum of similarities of the neighbours who watched it.
