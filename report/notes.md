# Recommendation Systems — G12

Working notes behind the report. Every choice is recorded with its reason and every number comes from `results/`.

## Experimental Setup

**Data.** MovieLens 100K with 943 users · 1,682 movies and 100,000 ratings. Every user has at least 20 ratings with a median of 65. The matrix is 93.7% empty and the top 10% of movies hold 42.7% of all interactions.

**Protocol.** Ratings are binarised so any rating counts as one interaction. Each user's interactions are split at random into 80% train · 10% validation and 10% test. Models rank every movie the user has not seen in earlier splits and are scored on the top 10. Settings are chosen on validation and test is used only for reported numbers.

**Metrics.** NDCG@10 · Recall@10 · MRR@10 and Hit@10. Task 1 reports RecBole's values. Task 2 replaces them with our own implementations.

**Framework.** RecBole from the course fork pinned to commit `439c5a8` with its data and model configurations unchanged. Every model shares one split which the exporter verifies. Scores for every user and movie pair are exported so hybrids combine full rankings instead of top 10 lists.

## Task 1 — Hybrid Recommender

We run all eleven models of the course fork across five families. Random and Pop are non-personalised baselines. ItemKNN and UserKNN are neighbourhood models. BPR · NeuMF and FISM are latent factor models. LightGCN and NGCF are graph models. EASE and SLIMElastic are linear item-item models. Appendix A describes each.

| Hypothesis | Source |
| --- | --- |
| Pop beats Random by a wide margin | Concentrated popularity |
| Linear item models are strong on ml-100k | Steck (2019) |
| A tuned BPR matches or beats NeuMF | Rendle et al. (2020) |
| LightGCN beats NGCF | He et al. (2020) |
| A weighted hybrid beats its best member | Models from different families make different errors (Burke 2002) |

### 1.1 Individual Models

Each model is trained once with its course configuration at seed 2020 and evaluated on test. These numbers are the reference for tuning.

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

**Observations.**
- **Linear item models lead.** EASE and SLIMElastic learn one weight per movie pair from co-occurrence. The catalogue holds only 1,682 movies with about 60 ratings each so a full item-item model is cheap to learn and well supported by data.
- **Neighbourhood models follow.** ItemKNN and UserKNN use the same co-occurrence signal but with fixed cosine weights instead of learned ones.
- **Pop reaches 16 times Random.** Popularity is concentrated so recommending the same hits to everyone already finds many test movies.
- **Trained models are cut short.** All five stopped at the 20 epoch limit and LightGCN · NGCF and FISM fall below both KNN models. The course settings rank these families by training budget rather than capacity which motivates tuning.

*Insert figure: validation NDCG@10 against epoch for the five trained models under their course configurations · expected to show the curves still rising at the 20 epoch limit.*

### 1.2 Tuning

**Selection.** Every trial trains on train and is scored on validation NDCG@10. NDCG rewards every hit by its rank while MRR credits only the first one so it matches the goal of a good top 10 list. Test is never evaluated during tuning.

**Training Budget.** Every trial may train up to 500 epochs with validation every 5 epochs and stops after 6 checks without gain. Each trial records its epochs so we can show that no winner hit the cap.

**Search Method.** Models with one or two settings are searched over a full grid which is cheap and misses nothing inside it. Models with four to eight settings use TPE with 40 to 70 trials since random search beats grids at equal budget (Bergstra and Bengio 2012) and TPE improves on it by learning from earlier trials. The search is seeded so it can be rerun exactly. Random and Pop have nothing to tune.

**Search Spaces.** Ranges are set wide in one pass from each model's paper and RecBole's defaults. They cover regularisation · embedding size · learning rate · batch size · negatives per positive and model specific depth or dropout (Appendix B). A winner on the edge of its range is reported as a limitation.

**Parallel Search.** Trials run in parallel processes and a free worker starts the next trial at once. Grid points follow RecBole's own seeded order so every trial and winner is identical at any worker count. With one worker TPE reproduces RecBole's sequential search exactly. With more workers TPE picks a trial without the results of trials still running which trades a little search efficiency for speed. BPR · NeuMF and LightGCN used one worker · NGCF batches of 4 and FISM 5 workers on an RTX 4090 since RecBole's FISM scores one user at a time and needed hours per trial on CPU. Threads per trial are fixed by the worker count since a different thread count changes results in the fourth decimal.

**Hardware.** The other eight models were tuned on an 8 core laptop CPU. The final runs and seed repeats of all eleven models ran on one GPU so every reported test number comes from one machine.

**Robustness.** Each tuned configuration is retrained with seeds 2020 to 2024 where every seed draws a new split and initialisation. We report test mean ± std and treat a gap smaller than the spread as a tie.

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

**Observations.**
- **Budget was the bottleneck.** Winners of the trained models stop between 125 and 270 epochs against the course limit of 20. LightGCN and FISM double their course NDCG@10 and all five now sit within 0.05 of the linear models.
- **Grid models were already near their best.** Their validation curves are flat near the top. EASE peaks at `reg_weight` 300 next to the course value of 250 and the five best ItemKNN settings lie within 0.001. The course ItemKNN setting sits on that plateau so its 0.005 test drop after tuning is selection noise.
- **Full rank item models still lead.** FISM learns the same kind of item-item similarity as EASE but through low rank factors and its winner takes the largest embedding size on offer. The item-item signal on this catalogue needs high rank which a full weight matrix gives for free.
- **Extra capacity does not pay.** NeuMF's winner takes the smallest embeddings on offer and still trails BPR. Its MLP branch adds parameters that ml-100k cannot fill which matches Rendle et al. (2020).
- **Graph simplification does not matter here.** NGCF ties LightGCN once its extra weights are held in check by dropout and regularisation. LightGCN's advantage was shown on larger and sparser data (He et al. 2020).

*Insert figure: validation NDCG@10 against epochs run for every TPE trial per model · shows every top trial running far past 20 epochs.*

*Insert figure: validation NDCG@10 against `reg_weight` for EASE · shows the flat optimum around the course value.*

| Hypothesis | Verdict |
| --- | --- |
| Pop beats Random by a wide margin | Holds with 0.110 against 0.007 |
| Linear item models are strong on ml-100k | Holds as SLIMElastic and EASE lead and tie within their spread |
| A tuned BPR matches or beats NeuMF | Holds with 0.314 against 0.304 and a gap larger than either spread |
| LightGCN beats NGCF | Not supported as they tie with NGCF 0.002 ahead |

**Limitations.**
- Some winners sit on the edge of their range: BPR and LightGCN at the largest batch size · LightGCN · NGCF and FISM at the largest embedding size · NGCF at the largest node dropout · NeuMF at the smallest embeddings.
- Trials with very low learning rates sometimes reached the 500 epoch cap but none was a winner.
- Tuning ran on two machines which can change which setting wins by small margins but not the reported test numbers.
- SLIMElastic ties at `l1_ratio` 0.001 and 0.0001 on validation NDCG@10 and the earlier trial wins.
- NGCF is not bit exact as identical runs differ in the fourth decimal on CPU and GPU while every other model repeats its metrics exactly.

### 1.3 Weighted Hybrid

**Idea.** The hybrid score of a user and movie is a weighted sum of the individual models' scores for that pair. Regression learns the weights on validation so the mix that best finds held-out movies wins.

**Members.** The nine tuned models and Pop. Random is left out since its scores carry no signal.

**Scaling.** Each model's scores are standardised per user over that user's candidate movies. Raw scales differ widely between models and only the order within a user matters for ranking.

**Rows.** Each user's candidates are the union of every member's top 100 movies outside train which covers 79% of validation movies in about 215,000 rows. A row holds the ten scaled scores of one candidate and its label is 1 when the movie is in the user's validation set. Candidates are used instead of the whole catalogue since the fit then learns the order near the top of the list which is all NDCG@10 sees. This mirrors the ranking stage of industrial recommenders where retrievers propose candidates and a ranker orders them.

**Regression.** Logistic regression with L2 regularisation. It fits the 0 or 1 label directly and each weight reads as how much a model's score moves the odds of a hit. L2 keeps the weights of near-duplicate models such as EASE and SLIMElastic stable instead of trading them off. Its strength is chosen by cross-validation.

**Test.** The weights are frozen. For each user the train and validation movies are hidden · the candidates are rebuilt from the remaining movies and ordered by the weighted sum and the top 10 is evaluated on test exactly as for the single models.

**Metrics.** The hybrid is not a RecBole model so its metrics come from our own module. It reproduces RecBole's test metrics exactly for the nine tuned models on both course and tuned runs. Random draws fresh scores at every call so its exported table is a different draw. Pop ties at rank 10 for 822 of 943 users so its top 10 depends on tie order and our ranking breaks ties by movie order to stay deterministic. RecBole's Pop also counts the sampled training negatives so its ranking matches true popularity at a rank correlation of only 0.90 which matters for the baseline comparison in 2.2.

**Stability.** Five-fold cross-validation over users fits the weights on 80% of users and scores the rest in turn. The spread of each weight across folds shows how stable its contribution is for 2.3 and the held-out score picks the L2 strength. The final weights use all validation users.

**Development.** Choices below were made on validation only.
- Fitting on every movie outside train lost to SLIMElastic even on validation with 0.2509 against 0.2589. The fit spent its effort separating hits from obvious misses deep in the tail which motivated the candidate rows.
- On candidates the L2 strength drives the ranking. Weak regularisation fits the 0 or 1 labels well but ranks poorly as correlated members cancel out with LightGCN at −0.90 against NGCF at 0.55. Strong regularisation pulls the weights toward a balanced mix that ranks better than any member.

| L2 Strength C | Validation NDCG@10 |
| --- | --- |
| 100 | 0.2511 |
| 1 | 0.2511 |
| 0.01 | 0.2529 |
| 0.0001 | 0.2640 |
| SLIMElastic alone | 0.2589 |

These values are in-sample so C is chosen by five-fold user cross-validation before test is used for the final number.

*Insert figure: validation NDCG@10 against C for the hybrid with the best member as a line · shows ranking quality rising as the weights shrink toward a balanced mix.*

**Limitations.**
- The members were tuned on the same validation set so their validation scores are slightly optimistic. At worst the weights end up a little off which lowers the test score and never inflates it since test stays unseen. Stacking with out-of-fold retraining of every member would remove this at a cost of hours for a small effect.
- The hybrid uses the seed 2020 split only since score tables exist for that split alone.

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
