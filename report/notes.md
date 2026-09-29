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

**Parallel Search.** Trials run in parallel processes and a free worker starts the next trial at once. Grid points are drawn in the same seeded order as RecBole's own search so every trial and winner is identical at any worker count. With one worker TPE reproduces RecBole's sequential search exactly. With more workers TPE picks a new trial without the results of trials still running which trades a little search efficiency for speed. BPR, NeuMF and LightGCN were tuned with one worker. NGCF was tuned in batches of 4. FISM was tuned with 5 workers on one RTX 4090 since RecBole's FISM scores one user at a time and needed hours per trial on CPU. The thread count per trial is fixed by the worker count since a different thread count changes results in the fourth decimal.

**Hardware.** The other eight models were tuned on an 8 core laptop CPU. The final runs and seed repeats of all eleven models ran on the same GPU so every reported test number comes from one machine.

**Robustness.** Each tuned configuration is retrained with 5 seeds from 2020 to 2024 where every seed draws a new split and a new initialisation. We report test mean ± std and treat a gap smaller than the spread as a tie.

| Model | Method | Trials | Valid NDCG@10 | Winner Epochs | Capped Trials |
| --- | --- | --- | --- | --- | --- |
| EASE | Grid | 17 | 0.2581 | — | — |
| ItemKNN | Grid | 50 | 0.2234 | — | — |
| UserKNN | Grid | 45 | 0.2355 | — | — |
| SLIMElastic | Grid | 56 | 0.2589 | — | — |
| BPR | TPE | 60 | 0.2535 | 220 | 8 |
| NeuMF | TPE | 70 | 0.2501 | 170 | 2 |
| LightGCN | TPE | 50 | 0.2504 | 270 | 5 |
| NGCF | TPE | 40 | 0.2523 | 220 | 7 |
| FISM | TPE | 40 | 0.2406 | 125 | 3 |

Grid models train in closed form or in one pass so epochs do not apply. No winner hit the 500 epoch cap.

| Model | Course NDCG@10 | Tuned NDCG@10 | Tuned Recall@10 | 5 Seeds NDCG@10 |
| --- | --- | --- | --- | --- |
| Random | 0.0065 | 0.0065 | 0.0056 | 0.0072 ± 0.0013 |
| Pop | 0.1034 | 0.1032 | 0.0850 | 0.1097 ± 0.0187 |
| ItemKNN | 0.2834 | 0.2783 | 0.2370 | 0.2773 ± 0.0045 |
| UserKNN | 0.2873 | 0.2877 | 0.2524 | 0.2844 ± 0.0095 |
| BPR | 0.2494 | 0.3197 | 0.2683 | 0.3138 ± 0.0064 |
| NeuMF | 0.2678 | 0.3116 | 0.2637 | 0.3043 ± 0.0056 |
| FISM | 0.1409 | 0.2875 | 0.2489 | 0.2840 ± 0.0064 |
| LightGCN | 0.1586 | 0.3146 | 0.2661 | 0.3091 ± 0.0039 |
| NGCF | 0.1842 | 0.3201 | 0.2745 | 0.3112 ± 0.0058 |
| EASE | 0.3295 | 0.3309 | 0.2808 | 0.3266 ± 0.0055 |
| SLIMElastic | 0.3235 | 0.3240 | 0.2769 | 0.3288 ± 0.0066 |

Random and Pop keep their course configurations. Their tuned column is the same configuration rerun on the GPU.

### Observations

Tuning lifts every trained model. The five neural and factor models gain between 0.04 and 0.16 NDCG@10 and LightGCN and FISM double their course scores. This supports the view that the course runs were cut short at 20 epochs. The four grid models barely move since their course settings were already close to the best ones. ItemKNN drops by 0.005 on test which is about one seed spread.

Against the hypotheses:

| Hypothesis | Verdict |
| --- | --- |
| Pop beats Random by a wide margin | Holds with 0.110 against 0.007 |
| Linear item models are strong on ml-100k | Holds as SLIMElastic and EASE lead and tie within their spread |
| A tuned BPR matches or beats NeuMF | Holds with 0.314 against 0.304 and a gap larger than either spread |
| LightGCN beats NGCF | Not supported as they tie with NGCF 0.002 ahead |

After tuning the ranking is SLIMElastic and EASE first then BPR · NGCF · LightGCN and NeuMF within 0.01 of each other then UserKNN and FISM then ItemKNN. The gap between the best linear model and the best trained model shrinks from 0.06 in the course runs to 0.015 across seeds.

**Limitations.** Some winners sit on the edge of their range: BPR and LightGCN at the largest batch size · LightGCN and FISM at the largest embedding size · NGCF at the largest embedding size and node dropout · NeuMF at the smallest embeddings. Wider ranges could help these models a little. Trials with very low learning rates sometimes reached the 500 epoch cap but none was a winner. Tuning ran on two machines which can change which setting wins by small margins but not the reported test numbers. SLIMElastic ties at `l1_ratio` 0.001 and 0.0001 on validation NDCG@10 and the earlier trial wins. NGCF is not bit exact on the GPU as two identical runs at seed 2020 scored 0.3201 and 0.3206 while every other model repeated exactly.

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
