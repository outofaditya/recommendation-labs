# Report Notes

## Setup

We use the course RecBole fork pinned to `439c5a8` with NumPy below 2 and pandas below 3 for compatibility. Ray Tune is added since the fork imports it without declaring it. MovieLens 100K is taken from the course fork as RecBole's download returns 403.

## Split

Each user's interactions are split at random into 80% train, 10% valid and 10% test with seed 2020. The split is identical for every model and the exporter verifies it. Models rank every movie a user has not interacted with in train or earlier splits.

## Expectations

These are hypotheses to confirm or reject with our results.

Pop should beat Random by a wide margin because the top 10% of movies hold 42.7% of all ratings. NeuMF should roughly match a well tuned BPR since Rendle et al. (2020) showed a dot product is hard for an MLP to beat. LightGCN should beat NGCF as He et al. (2020) found its removed weights and activations only hurt on sparse data. EASE and SLIMElastic should lead on ml-100k since linear item models are strong on small dense data (Steck 2019) and they already led our one epoch smoke runs. BPR at 20 epochs is likely undertrained and should gain the most from tuning.

## Task 1.1 — Individual Models

The course helpers save only the top-K so our exporter saves every user and movie score for the hybrids.

### Results with Course Configs

| Model | NDCG@10 | Recall@10 | MRR@10 | Hit@10 |
| --- | --- | --- | --- | --- |
| EASE | 0.3295 | 0.2805 | 0.5277 | 0.8197 |
| SLIMElastic | 0.3235 | 0.2750 | 0.5231 | 0.8123 |
| UserKNN | 0.2873 | 0.2442 | 0.4834 | 0.7805 |
| ItemKNN | 0.2834 | 0.2470 | 0.4623 | 0.7847 |
| NeuMF | 0.2651 | 0.2245 | 0.4443 | 0.7434 |
| BPR | 0.2509 | 0.2147 | 0.4307 | 0.7349 |
| NGCF | 0.1881 | 0.1570 | 0.3216 | 0.6479 |
| LightGCN | 0.1441 | 0.1283 | 0.2718 | 0.5801 |
| FISM | 0.1409 | 0.1220 | 0.2669 | 0.5620 |
| Pop | 0.1034 | 0.0880 | 0.1951 | 0.4698 |
| Random | 0.0065 | 0.0056 | 0.0145 | 0.0636 |

Test set with the course configs untuned. The linear item models lead as expected. LightGCN falls below NGCF which contradicts our expectation and points to undertraining since every gradient trained model runs only 20 epochs at learning rate 0.001. NeuMF edges out BPR under the same budget. NGCF varies in the third decimal between runs despite the fixed seed. The exporter's pairwise fallback for NeuMF reproduces RecBole's recall exactly.

### Random

Random scores every movie uniformly without training. RecBole shares one random vector per user batch which we keep as shipped.

It is good only as a floor since chance hits about 6% of users. It fails everywhere else because it uses no signal at all.

### Pop

Pop counts how often each movie appears in train and divides by the highest count. Star values are ignored. Every user gets the same ranking minus their seen movies which makes it the popularity bias extreme and a stronger baseline than Random.

It is good for new users with few ratings since popular movies are a safe guess when nothing else is known. It fails on personal taste and niche movies because every user gets the same list and the long tail never appears.

### ItemKNN

ItemKNN computes cosine similarity between movies from their train audiences and keeps the 100 nearest per movie. A movie's score is the sum of its similarities to the movies the user watched. No weights are learned and `k` is the main setting to tune.

It is good for users with many ratings since each watched movie adds evidence and the reasons stay explainable. It fails on rarely rated movies because a few shared viewers give unreliable similarities.

### UserKNN

UserKNN runs RecBole's ItemKNN with `knn_method: 'user'`. It computes cosine similarity between users from their train histories and keeps the 100 nearest per user. A movie's score is the sum of similarities of the neighbours who watched it.

It is good for users whose taste matches a clear crowd since neighbours with many shared movies are reliable. It fails for users with unusual taste because their nearest neighbours share little and heavy raters dominate every neighbour list.

### BPR

BPR is matrix factorization with 64 dimensional user and movie embeddings trained by Bayesian Personalized Ranking. Each step pairs a watched movie with one uniformly sampled unwatched movie and pushes the watched score above it through `−log σ(pos − neg)` with Adam at learning rate 0.001. The course config trains only 20 epochs which is likely too few and a target for tuning.

It is good on sparse data since two users with no shared movies can still match through similar embeddings. It fails when undertrained or on rarely rated movies because their embeddings receive few updates and stay close to random.

### NeuMF

NeuMF keeps two 64 dimensional embeddings per user and movie. The GMF branch multiplies them element-wise and the MLP branch concatenates them through layers of 128 and 64 with ReLU and dropout 0.1. Both outputs are concatenated into one score trained pointwise with binary cross-entropy against one sampled negative per positive.

It is good when taste mixes features in ways a dot product cannot express. It fails on small data like ml-100k because the extra layers overfit and learning even a plain dot product is hard for them.

### LightGCN

LightGCN treats train as a bipartite graph of users and movies. Starting from 64 dimensional embeddings it runs 3 propagation layers where each node takes the neighbour average weighted by `1/√(deg_u × deg_i)` and averages all layers into the final embedding. Only the layer zero embeddings are learned through BPR loss with L2 weight 1e-4.

It is good for users with few ratings since hops borrow signal from neighbours of neighbours. It fails on niche taste when popular movies spread their signal through many paths which pushes toward popularity.

### NGCF

NGCF runs the same graph propagation as LightGCN but each of its 3 layers adds learned weights on the neighbour sum and on its element-wise product with the node, then LeakyReLU, message dropout 0.1 and rescaling. Layers are concatenated into 256 dimensions. LightGCN removes this machinery and usually performs better.

It is good in principle at modelling feature interactions through its product term. It fails in practice because the per layer weights and activations add parameters that overfit and slow training without adding useful signal.

### EASE

EASE learns one movie by movie weight table shared by all users. Each train cell is rebuilt as a weighted sum of the other cells in its row with the diagonal forced to zero so no movie predicts itself. The best weights come from a single least squares formula with an L2 penalty of `reg_weight` 250 so there are no epochs. A user's score for a movie is the sum of its weights over the movies they watched.

It is good on small dense data like ml-100k since weights are learned jointly and shared signal between co-watched movies is not counted twice. It fails on huge catalogues because inverting a movie by movie matrix becomes infeasible and it ignores order and time.
