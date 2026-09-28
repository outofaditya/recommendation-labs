# Recommendation Systems — G12

## Experimental Setup

**Data.** MovieLens 100K with 943 users, 1,682 movies and 100,000 ratings. Every user has at least 20 ratings with a median of 65. The matrix is 93.7% empty and popularity is concentrated as the top 10% of movies hold 42.7% of all interactions.

**Protocol.** Ratings are binarised into implicit feedback so any rating counts as one interaction regardless of stars. Each user's interactions are split at random into 80% train, 10% validation and 10% test. Models rank every movie the user has not interacted with in earlier splits and are scored on the top 10. All settings are chosen on validation and test is used only for reported numbers.

**Metrics.** NDCG@10, Recall@10, MRR@10 and Hit@10. RecBole's values are reported in Task 1 and replaced by our own implementations in Task 2.

**Framework.** RecBole from the course fork pinned to commit `439c5a8`. Every model shares the identical split which our exporter verifies. Scores for every user and movie pair are exported so hybrids can combine full rankings instead of truncated top 10 lists.

## Task 1 — Hybrid Recommender

We use the eleven models provided with the course fork spanning five families. Random and Pop are non-personalised baselines. ItemKNN and UserKNN are neighbourhood models. BPR, NeuMF and FISM are latent factor models. LightGCN and NGCF are graph models. EASE and SLIMElastic are linear item-item models.

### Hypotheses

| Hypothesis | Source |
| --- | --- |
| Pop beats Random by a wide margin | Concentrated popularity |
| Linear item models are strong on ml-100k | Steck (2019) |
| A tuned BPR matches or beats NeuMF | Rendle et al. (2020) |
| LightGCN beats NGCF | He et al. (2020) |

### 1.1 Individual Models

Each model is trained once with its course configuration at seed 2020 and evaluated on test. These numbers are the reference that tuning must improve on.

| Model | NDCG@10 | Recall@10 | MRR@10 | Hit@10 |
| --- | --- | --- | --- | --- |
| Random | 0.0065 | 0.0056 | 0.0145 | 0.0636 |
| Pop | 0.1034 | 0.0880 | 0.1951 | 0.4698 |
| ItemKNN | 0.2834 | 0.2470 | 0.4623 | 0.7847 |
| UserKNN | 0.2873 | 0.2442 | 0.4834 | 0.7805 |
| BPR | 0.2494 | 0.2085 | 0.4304 | 0.7275 |
| NeuMF | 0.2678 | 0.2271 | 0.4530 | 0.7667 |
| FISM | 0.1409 | 0.1220 | 0.2669 | 0.5620 |
| LightGCN | 0.1586 | 0.1388 | 0.2963 | 0.6098 |
| NGCF | 0.1842 | 0.1545 | 0.3137 | 0.6394 |
| EASE | 0.3295 | 0.2805 | 0.5277 | 0.8197 |
| SLIMElastic | 0.3235 | 0.2750 | 0.5231 | 0.8123 |

EASE and SLIMElastic lead and both neighbourhood models follow closely. Pop reaches 16 times the NDCG@10 of Random. All five trained models ran into the 20 epoch limit so they are likely undertrained and LightGCN falls below both KNN models. The course configurations may therefore rank the families by training budget rather than by capacity which motivates tuning.

### 1.2 Tuning

**Selection.** Every trial trains on train and is scored on validation NDCG@10. NDCG rewards every hit by its rank while MRR credits only the first one so it matches the goal of a good top 10 list. Test is never evaluated during tuning so no test information reaches the chosen settings.

**Training Budget.** The course configurations stop at 20 epochs which cuts the neural models short. Every trial may train up to 500 epochs with validation every 5 epochs and stops after 6 checks without gain. Each trial records the epochs it ran so we can prove that no chosen configuration hit the cap.

**Search Method.** Models with one or two settings are searched exhaustively over a full grid which is cheap and misses nothing inside the grid. Models with four to eight settings use TPE with a fixed budget of 40 to 70 trials since random search beats grids at equal budget (Bergstra and Bengio 2012) and TPE improves on it by learning from earlier trials. TPE is seeded so every search can be rerun exactly. Random and Pop have nothing to tune.

**Search Spaces.** Ranges are set wide in a single pass from each model's paper and RecBole's defaults. They span regularisation · embedding size · learning rate · batch size · negatives per positive and model specific depth or dropout. The full spaces are in Appendix B. A winner on the edge of its range is reported as a limitation.

**Robustness.** Each tuned configuration is retrained with 5 seeds from 2020 to 2024 where every seed draws a new split and a new initialisation. We report test mean ± std and treat a gap smaller than the spread as a tie.

| Model | Method | Trials | Valid NDCG@10 | Epochs | Capped |
| --- | --- | --- | --- | --- | --- |
| EASE | Grid | 17 | — | — | — |
| ItemKNN | Grid | 50 | — | — | — |
| UserKNN | Grid | 45 | — | — | — |
| SLIMElastic | Grid | 56 | — | — | — |
| BPR | TPE | 60 | — | — | — |
| NeuMF | TPE | 70 | — | — | — |
| LightGCN | TPE | 50 | — | — | — |
| NGCF | TPE | 40 | — | — | — |
| FISM | TPE | 40 | — | — | — |

| Model | Course NDCG@10 | Tuned NDCG@10 | Tuned Recall@10 |
| --- | --- | --- | --- |
| Random | 0.0065 | — | — |
| Pop | 0.1034 | — | — |
| ItemKNN | 0.2834 | — | — |
| UserKNN | 0.2873 | — | — |
| BPR | 0.2494 | — | — |
| NeuMF | 0.2678 | — | — |
| FISM | 0.1409 | — | — |
| LightGCN | 0.1586 | — | — |
| NGCF | 0.1842 | — | — |
| EASE | 0.3295 | — | — |
| SLIMElastic | 0.3235 | — | — |

### Observations

To be written from the results against the hypotheses above.

## Appendix A — Model Descriptions

**Random.** Assigns every movie a uniform random score without training. It is the floor where chance alone hits about 6% of users. RecBole draws one random vector per user batch so users in a batch share a ranking.

**Pop.** Scores each movie by its train interaction count divided by the maximum count. Every user receives the same ranking minus their seen movies. It favours mainstream taste and never surfaces the long tail which makes it the extreme case of popularity bias.

**ItemKNN.** Computes cosine similarity between movies from their train audiences and keeps the `k` nearest per movie. A movie's score is the sum of its similarities to the user's history. It is explainable and strong for users with long histories but unreliable for rarely rated movies where a few shared viewers produce high similarity by chance.

**UserKNN.** The same computation over users with `knn_method: 'user'`. A movie's score is the sum of similarities of the `k` nearest users who watched it. It works when a user's taste matches a clear crowd but heavy raters overlap with nearly everyone and dominate neighbour lists.

**BPR.** Matrix factorization with user and movie embeddings trained by Bayesian Personalized Ranking. Each step samples unwatched movies per positive and minimises $-\log \sigma(\hat{s}_{ui} - \hat{s}_{uj})$. Shared embeddings let users without common movies still match but rarely rated movies receive few updates.

**NeuMF.** Two embeddings per user and movie. A GMF branch takes their element-wise product and an MLP branch passes their concatenation through ReLU layers with dropout. Both outputs feed one linear layer trained pointwise with binary cross-entropy. The extra capacity can model non-linear taste but tends to overfit small data.

**FISM.** Factored item similarity. A user is represented by the sum of the embeddings of the movies they watched normalised by $|N(u)|^{\alpha}$ and a candidate is scored against it so item-item similarity is learned through low rank factors. Trained pointwise with binary cross-entropy.

**LightGCN.** Propagates embeddings over the user-movie graph using the symmetric weight $1/\sqrt{|N(u)||N(i)|}$ and averages all layers into the final embedding. Only the layer zero embeddings are learned through BPR loss. Multi-hop propagation helps sparse users but popular movies spread signal through many paths.

**NGCF.** The same propagation with learned weight matrices on the neighbour sum and on its element-wise product with the node followed by LeakyReLU, message dropout and normalisation. Layers are concatenated rather than averaged. The added parameters make training harder on sparse data.

**EASE.** Learns one item-item weight matrix $B$ by minimising $\lVert R - RB \rVert^2 + \lambda \lVert B \rVert^2$ with a zero diagonal so no movie predicts itself. The solution is closed form. Strong on small dense catalogues and infeasible for very large ones.

**SLIMElastic.** The same reconstruction objective solved as one elastic net regression per movie with non-negative weights. The L1 term yields a sparse and interpretable weight matrix at the cost of losing negative associations.

## Appendix B — Search Spaces

Log marks a range sampled on a log scale. Every other range is a list of choices or a uniform interval.

| Model | Setting | Space |
| --- | --- | --- |
| EASE | `reg_weight` | 1 · 5 · 10 · 25 · 50 · 100 · 200 · 300 · 400 · 500 · 600 · 800 · 1000 · 1500 · 2000 · 3000 · 5000 |
| ItemKNN | `k` | 10 · 20 · 50 · 100 · 200 · 300 · 400 · 500 · 700 · 1000 |
| ItemKNN | `shrink` | 0 · 10 · 50 · 100 · 200 |
| UserKNN | `k` | 10 · 20 · 50 · 100 · 200 · 300 · 400 · 500 · 700 |
| UserKNN | `shrink` | 0 · 10 · 50 · 100 · 200 |
| SLIMElastic | `alpha` | 0.001 · 0.005 · 0.01 · 0.05 · 0.1 · 0.2 · 0.5 · 1 |
| SLIMElastic | `l1_ratio` | 0.0001 · 0.0005 · 0.001 · 0.005 · 0.01 · 0.05 · 0.1 |
| BPR | `embedding_size` | 32 · 64 · 128 · 256 · 512 · 1024 |
| BPR | `learning_rate` | 1e-5 to 1e-2 log |
| BPR | `weight_decay` | 0 · 1e-6 · 1e-5 · 1e-4 · 1e-3 |
| BPR | `train_batch_size` | 512 · 1024 · 2048 · 4096 |
| BPR | Negatives | 1 · 2 · 4 |
| NeuMF | `mf_embedding_size` · `mlp_embedding_size` | 16 · 32 · 64 · 128 |
| NeuMF | `mlp_hidden_size` | [64,32] · [128,64] · [256,128] · [128,64,32] · [256,128,64] |
| NeuMF | `dropout_prob` | 0 to 0.7 |
| NeuMF | `learning_rate` | 1e-5 to 1e-2 log |
| NeuMF | `weight_decay` | 0 · 1e-6 · 1e-5 · 1e-4 · 1e-3 |
| NeuMF | `train_batch_size` | 512 · 1024 · 2048 |
| NeuMF | Negatives | 1 · 2 · 4 |
| LightGCN | `embedding_size` | 32 · 64 · 128 · 256 |
| LightGCN | `n_layers` | 1 to 6 |
| LightGCN | `reg_weight` | 1e-6 to 1e-2 log |
| LightGCN | `learning_rate` | 1e-4 to 1e-2 log |
| LightGCN | `train_batch_size` | 1024 · 2048 · 4096 |
| NGCF | `embedding_size` | 32 · 64 · 128 |
| NGCF | `hidden_size_list` | [64] · [64,64] · [64,64,64] · [64,64,64,64] · [128,128] · [128,128,128] |
| NGCF | `message_dropout` | 0 to 0.5 |
| NGCF | `node_dropout` | 0 · 0.1 · 0.2 |
| NGCF | `reg_weight` | 1e-6 to 1e-2 log |
| NGCF | `learning_rate` | 1e-4 to 1e-2 log |
| FISM | `embedding_size` | 32 · 64 · 128 · 256 |
| FISM | `alpha` | 0 to 1 |
| FISM | `reg_weights` | 1e-6 · 1e-5 · 1e-4 · 1e-3 · 1e-2 |
| FISM | `learning_rate` | 1e-4 to 1e-2 log |

Kept fixed: LightGCN uses one negative as in its paper · SLIMElastic keeps non-negative weights which define SLIM · all evaluation settings.
