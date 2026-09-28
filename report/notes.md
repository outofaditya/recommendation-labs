# Recommendation Systems — G12

Working draft of the final report. Main body sections follow the one page and 200 word limit per task. Model descriptions and implementation details sit in the appendix.

## Experimental Setup

**Data.** MovieLens 100K with 943 users, 1,682 movies and 100,000 ratings. Every user has at least 20 ratings with a median of 65. The matrix is 93.7% empty and popularity is concentrated as the top 10% of movies hold 42.7% of all interactions.

**Protocol.** Ratings are binarised into implicit feedback so any rating counts as one interaction regardless of stars. Each user's interactions are split at random into 80% train, 10% validation and 10% test with seed 2020. Models rank every movie the user has not interacted with in earlier splits and are scored on the top 10. All settings are chosen on validation and test is used once for the reported numbers.

**Metrics.** NDCG@10, Recall@10, MRR@10 and Hit@10. RecBole's values are reported in Task 1 and replaced by our own implementations in Task 2.

**Framework.** RecBole from the course fork pinned to commit `439c5a8`. Every model shares the identical split which our exporter verifies. Scores for every user and movie pair are exported so hybrids can combine full rankings instead of truncated top 10 lists.

## Task 1.1 — Individual Models

We run the eleven models provided with the course fork spanning four families. Random and Pop serve as non-personalised baselines. ItemKNN and UserKNN are neighbourhood models. BPR, NeuMF and FISM are latent factor models. LightGCN and NGCF are graph models. EASE and SLIMElastic are linear item-item models. All use the untuned course configurations.

| Family | Model | NDCG@10 | Recall@10 | MRR@10 | Hit@10 |
| --- | --- | --- | --- | --- | --- |
| Linear | EASE | **0.3295** | **0.2805** | **0.5277** | **0.8197** |
| Linear | SLIMElastic | 0.3235 | 0.2750 | 0.5231 | 0.8123 |
| Neighbourhood | UserKNN | 0.2873 | 0.2442 | 0.4834 | 0.7805 |
| Neighbourhood | ItemKNN | 0.2834 | 0.2470 | 0.4623 | 0.7847 |
| Latent Factor | NeuMF | 0.2651 | 0.2245 | 0.4443 | 0.7434 |
| Latent Factor | BPR | 0.2509 | 0.2147 | 0.4307 | 0.7349 |
| Graph | NGCF | 0.1881 | 0.1570 | 0.3216 | 0.6479 |
| Graph | LightGCN | 0.1441 | 0.1283 | 0.2718 | 0.5801 |
| Latent Factor | FISM | 0.1409 | 0.1220 | 0.2669 | 0.5620 |
| Baseline | Pop | 0.1034 | 0.0880 | 0.1951 | 0.4698 |
| Baseline | Random | 0.0065 | 0.0056 | 0.0145 | 0.0636 |

*Table 1. Test results with untuned course configurations.*

### Discussion

The ranking splits cleanly by how each model obtains its parameters. Models fitted in closed form or by counting lead because they cannot be undertrained. EASE and SLIMElastic learn every item-item weight jointly from co-occurrence which suits a small catalogue where each user has a median of 65 ratings. EASE edges out SLIMElastic because it permits negative weights and keeps a dense table while SLIMElastic forces non-negative sparse weights. The KNN models use the same co-occurrence signal but score each pair independently so correlated movies are counted twice.

Every gradient-trained model trails because the course budget of 20 epochs at learning rate 0.001 is short. LightGCN falling below NGCF contradicts He et al. (2020) and the budget explains it. LightGCN trains with batches of 4,096 so it takes about 400 optimiser steps in total against about 800 for NGCF and BPR at batch size 2,048 and its only learnable parameters are the embeddings. Tuning in Task 1.2 should close this gap.

Pop reaches a 47% hit rate against 6% for Random which reflects the concentrated popularity of the dataset rather than any personalisation.

### Hypotheses

| Hypothesis | Source | Status |
| --- | --- | --- |
| Pop beats Random by a wide margin | Concentrated popularity | Confirmed |
| Linear item models lead on ml-100k | Steck (2019) | Confirmed |
| NeuMF roughly matches a tuned BPR | Rendle et al. (2020) | Open until tuning |
| LightGCN beats NGCF | He et al. (2020) | Contradicted untuned |
| BPR gains the most from tuning | Short training budget | Open until tuning |

## Task 1.2 — Individual Tuning

Each model is tuned by exhaustive grid search with RecBole's `HyperTuning` over one or two settings per model. Every trial trains on train and is scored on validation. The winning configuration is stored in `parameters/tuned/` and every trial is exported for the appendix.

### Design Choices

| Choice | Reason |
| --- | --- |
| NDCG@10 for selection and early stopping | It rewards every hit and weights each by its rank position which matches the top 10 ranking task. MRR@10 used by the course configs only credits the first hit. |
| Exhaustive grid | Each model has one to three influential settings so a full grid is affordable and misses no combination. |
| RecBole `HyperTuning` | The course framework's own tuner keeps runs comparable with the fork and exports every trial. Test results it logs are never used for selection. |
| Up to 300 epochs with early stopping | The course budget of 20 epochs undertrained every gradient model. Validation is checked every 5 epochs and training stops after 5 checks without improvement. |
| Widen when a winner sits on the edge | A best value at the boundary of its grid may be beaten beyond it. Grids are widened and rerun until every winner is interior or at a natural floor such as zero dropout. Earlier trials are reused. |
| Settings kept fixed | Embedding sizes other than BPR's and regularisation weights stay at course values to keep grids small and comparisons fair. |

### Search Spaces

Final grids after widening.

| Model | Settings Searched |
| --- | --- |
| ItemKNN | `k` in 10 25 50 100 200 400 600 800 1200 and `shrink` in 0 10 50 |
| UserKNN | `k` in 10 25 50 100 200 400 and `shrink` in 0 10 50 |
| EASE | `reg_weight` in 10 50 100 250 500 1000 2000 |
| SLIMElastic | `alpha` in 0.01 0.05 0.1 0.2 0.5 and `l1_ratio` in 0.0001 0.0005 0.001 0.01 0.02 0.1 |
| BPR | `embedding_size` in 32 64 128 256 512 and `learning_rate` in 0.0005 0.001 0.005 0.01 |
| NeuMF | `dropout_prob` in 0 0.1 0.3 0.5 0.7 and `learning_rate` in 0.0001 0.0002 0.0005 0.001 0.005 |
| LightGCN | `n_layers` in 1 to 6 and `learning_rate` in 0.001 0.005 0.01 |
| NGCF | `message_dropout` in 0 0.1 0.3 and `learning_rate` in 0.0001 0.0002 0.0005 0.001 0.005 |
| FISM | `alpha` in 0 0.5 1 and `learning_rate` in 0.001 0.005 0.01 |

Random and Pop have no settings to tune.

### Tuning Rounds

**Round 1.** The initial grids ran 114 trials. Six of nine winners landed on a grid boundary which signalled the search had not yet bracketed the optimum.

| Model | Round 1 Winner | Valid NDCG@10 | Boundary Hit |
| --- | --- | --- | --- |
| SLIMElastic | `alpha` 0.1 and `l1_ratio` 0.001 | 0.2589 | Smallest `l1_ratio` |
| EASE | `reg_weight` 500 | 0.2568 | None |
| NGCF | `learning_rate` 0.0005 and `message_dropout` 0 | 0.2486 | Smallest `learning_rate` |
| LightGCN | `learning_rate` 0.005 and `n_layers` 4 | 0.2433 | Largest `n_layers` |
| BPR | `embedding_size` 128 and `learning_rate` 0.001 | 0.2411 | Largest `embedding_size` |
| UserKNN | `k` 50 and `shrink` 0 | 0.2355 | None |
| NeuMF | `dropout_prob` 0.3 and `learning_rate` 0.0005 | 0.2302 | Largest dropout and smallest `learning_rate` |
| ItemKNN | `k` 400 and `shrink` 10 | 0.2234 | Largest `k` |
| FISM | `alpha` 0.5 and `learning_rate` 0.005 | 0.1238 | None |

The longer training budget alone changed the picture. NGCF and LightGCN rose from the bottom half in Task 1.1 to within 0.01 of the linear models on validation which confirms the untuned graph models were undertrained rather than weak. The boundary hits also carry meaning. NeuMF preferring the strongest dropout and smallest step points to overfitting on a dataset of this size and ItemKNN preferring the largest neighbourhood suggests ml-100k is dense enough that more neighbours add signal rather than noise.

**Round 2.** Each boundary dimension was extended beyond its edge while keeping all earlier values so the grid stays exhaustive.

ItemKNN, SLIMElastic, LightGCN and NGCF settled inside their grids. The wider ItemKNN neighbourhoods of 600 to 1200 scored lower which confirms 400 as a true optimum. BPR and NeuMF moved to new boundaries at a lower learning rate so both received a third round.

**Round 3.** NeuMF settled at a learning rate of 0.0001 with 0.00005 scoring lower. BPR reached the new boundary of 1024 dimensions at a learning rate of 0.0001 but the interior runner-up at 512 dimensions and 0.0002 scores 0.2503 against 0.2506. We stop at this plateau since a further round would gain under 0.001 while doubling the model size again.

### Final Configurations

| Model | Winner | Trials | Valid NDCG@10 | Change from Course Config |
| --- | --- | --- | --- | --- |
| SLIMElastic | `alpha` 0.1 and `l1_ratio` 0.001 | 30 | 0.2589 | Weaker L1 for a denser table |
| EASE | `reg_weight` 500 | 7 | 0.2568 | Twice the regularisation |
| BPR | `embedding_size` 1024 and `learning_rate` 0.0001 | 36 | 0.2506 | Larger and slower with a longer budget |
| NGCF | `learning_rate` 0.0005 and `message_dropout` 0 | 15 | 0.2486 | Slower with no dropout |
| LightGCN | `n_layers` 5 and `learning_rate` 0.005 | 18 | 0.2443 | Two more layers and a faster step |
| UserKNN | `k` 50 and `shrink` 0 | 18 | 0.2355 | Half the neighbourhood |
| NeuMF | `dropout_prob` 0.5 and `learning_rate` 0.0001 | 30 | 0.2309 | Heavier dropout and a slower step |
| ItemKNN | `k` 400 and `shrink` 10 | 27 | 0.2234 | Four times the neighbourhood |
| FISM | `alpha` 0.5 and `learning_rate` 0.005 | 9 | 0.1238 | Partial normalisation and a faster step |

190 trials in total across three rounds.


## Appendix A — Model Descriptions

**Random.** Assigns every movie a uniform random score without training. Serves as the floor where chance alone hits about 6% of users.

**Pop.** Scores each movie by its train interaction count divided by the maximum count. Every user receives the same ranking minus their seen movies. It favours users with mainstream taste and never surfaces the long tail which makes it the extreme case of popularity bias.

**ItemKNN.** Computes cosine similarity between movies from their train audiences and keeps the 100 nearest per movie. A movie's score is the sum of its similarities to the user's history. It is explainable and strong for users with long histories but unreliable for rarely rated movies where a few shared viewers produce high similarity by chance.

**UserKNN.** The same computation over users with `knn_method: 'user'`. A movie's score is the sum of similarities of the 100 nearest users who watched it. It works when a user's taste matches a clear crowd but heavy raters overlap with nearly everyone and dominate neighbour lists.

**BPR.** Matrix factorization with 64 dimensional user and movie embeddings trained by Bayesian Personalized Ranking. Each step samples one unwatched movie per positive and minimises $-\log \sigma(\hat{s}_{ui} - \hat{s}_{uj})$. Shared embeddings let users without common movies still match but rarely rated movies receive few updates.

**NeuMF.** Two 64 dimensional embeddings per user and movie. A GMF branch takes their element-wise product and an MLP branch passes their concatenation through layers of 128 and 64 with ReLU and dropout 0.1. Both outputs feed one linear layer trained pointwise with binary cross-entropy. The extra capacity can model non-linear taste but tends to overfit small data.

**FISM.** Factored item similarity. A user is represented by the normalised sum of the embeddings of the movies they watched and a candidate is scored against it so item-item similarity is learned through low rank factors. Trained pointwise with binary cross-entropy.

**LightGCN.** Propagates 64 dimensional embeddings over the user-movie graph for three layers using the symmetric weight $1/\sqrt{|N(u)||N(i)|}$ and averages all layers into the final embedding. Only the layer zero embeddings are learned through BPR loss. Multi-hop propagation helps sparse users but popular movies spread signal through many paths.

**NGCF.** The same propagation with learned weight matrices on the neighbour sum and on its element-wise product with the node followed by LeakyReLU, message dropout 0.1 and normalisation. Layers are concatenated into 256 dimensions. The added parameters make training harder without adding signal on sparse data.

**EASE.** Learns one item-item weight matrix $B$ by minimising $\lVert R - RB \rVert^2 + \lambda \lVert B \rVert^2$ with a zero diagonal so no movie predicts itself. The solution is closed form with $\lambda = 250$. Strong on small dense catalogues and infeasible for very large ones.

**SLIMElastic.** The same reconstruction objective solved as one elastic net regression per movie with non-negative weights, `alpha` 0.2 and `l1_ratio` 0.02. The L1 term yields a sparse and interpretable weight matrix at the cost of losing negative associations.

## Appendix B — Implementation Notes

| Item | Detail |
| --- | --- |
| NumPy Below 2 | RecBole uses `np.float_` which NumPy 2 removed |
| pandas Below 3 | pandas 3 silently ignores RecBole's in-place missing value fill |
| Ray Tune | The fork imports it without declaring it |
| Local Dataset | RecBole's download returns 403 so ml-100k comes from the course fork |
| UserKNN | Provided as an ItemKNN configuration with `knn_method: 'user'` |
| NeuMF Export | NeuMF lacks full sort prediction so the exporter scores pairwise and reproduces RecBole's recall exactly |
| Random | RecBole draws one random vector per user batch so users in a batch share a ranking |
| NGCF | Results vary in the third decimal between runs despite the fixed seed |
| hyperopt 0.2.5 | Required by RecBole's tuner and needs setuptools below 81 for `pkg_resources` |
