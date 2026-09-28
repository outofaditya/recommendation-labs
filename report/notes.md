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
