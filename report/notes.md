# Recommendation Systems — G12

Working notes behind the report. Every choice has its reason and every number comes from `results/`.

## Experimental Setup

**Data.** MovieLens 100K with 943 users · 1,682 movies and 100,000 ratings. Every user has at least 20 ratings. The matrix is 93.7% empty and the top 10% of movies hold 42.7% of interactions.

**Protocol.** Any rating counts as one interaction. Each user's interactions are split at random into 80% train · 10% validation and 10% test. Models rank every movie the user has not seen and are scored on the top 10. Settings are chosen on validation and test is used only for reported numbers.

**Metrics.** NDCG@10 · Recall@10 · MRR@10 and Hit@10. Our own module in `source/metrics/` reproduces RecBole's values exactly for the nine tuned models.

**Framework.** RecBole from the course fork at commit `439c5a8` with its data and configurations unchanged. All models share one verified split and export a score for every user and movie so hybrids combine full rankings.

## Task 1 — Hybrid Recommender

Eleven models from five families: Random and Pop as baselines · ItemKNN and UserKNN as neighbourhood models · BPR · NeuMF and FISM as latent factor models · LightGCN and NGCF as graph models · EASE and SLIMElastic as linear item-item models. Content-based models are excluded by the brief.

| Hypothesis | Source |
| --- | --- |
| Pop beats Random by a wide margin | Concentrated popularity |
| Linear item models are strong on ml-100k | Steck (2019) |
| A tuned BPR matches or beats NeuMF | Rendle et al. (2020) |
| LightGCN beats NGCF | He et al. (2020) |
| A weighted hybrid beats its best member | Different families make different errors (Burke 2002) |
| Different members win for different user groups | Short histories favour broad signals over item-item ones |

### 1.1 Individual Models

Each model trains once with its course configuration at seed 2020.

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
- **Linear item models lead.** EASE and SLIMElastic learn one weight per movie pair. With 1,682 movies and about 60 ratings each a full item-item model is cheap and well supported.
- **Neighbourhood models follow.** ItemKNN and UserKNN use the same co-occurrence signal with fixed cosine weights instead of learned ones.
- **Pop reaches 16 times Random.** Concentrated popularity lets one ranking for everyone find many test movies.
- **Trained models are cut short.** All five stop at the 20 epoch limit (`epochs` in `results/metrics/*.json`) so the course settings rank them by training budget rather than capacity.

### 1.2 Tuning

**Method.**
- **Selection.** Validation NDCG@10 since it credits every hit by rank while MRR credits only the first.
- **Budget.** Up to 500 epochs with validation every 5 and a stop after 6 checks without gain. Every trial records its epochs to show no winner hit the cap.
- **Search.** Full grids for models with one or two settings. Seeded TPE with 40 to 70 trials for four to eight settings since random search beats grids at equal budget (Bergstra and Bengio 2012) and TPE learns from earlier trials.
- **Spaces.** Set wide in one pass from each paper and RecBole's defaults (Appendix B). A winner on an edge is a limitation.
- **Parallelism.** A free worker starts the next trial at once. Grids are identical at any worker count and one worker reproduces RecBole's sequential TPE. FISM was tuned with 5 workers on an RTX 4090 since its scoring takes hours per trial on CPU. All final runs used that GPU.
- **Robustness.** Each winner is retrained with seeds 2020 to 2024 which redraw split and initialisation. A gap smaller than the spread is a tie.

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

**Observations.**
- **Budget was the bottleneck.** Trained winners stop between 125 and 270 epochs against the course limit of 20. LightGCN and FISM double their NDCG@10.
- **Grid models were near their best.** EASE peaks at `reg_weight` 300 beside the course value of 250 and the five best ItemKNN settings lie within 0.001. ItemKNN's 0.005 test drop is selection noise on that plateau.
- **Full rank item models still lead.** FISM learns item similarity through low rank factors and takes the largest embedding on offer so this catalogue needs the high rank a full weight matrix gives for free.
- **Extra capacity does not pay.** NeuMF takes the smallest embeddings on offer and still trails BPR which matches Rendle et al. (2020).
- **Graph simplification does not matter here.** NGCF ties LightGCN once dropout and regularisation hold its extra weights in check. LightGCN's gain was shown on larger sparser data (He et al. 2020).

*Figure: validation NDCG@10 against epochs for every TPE trial from `results/tuning/{BPR,NeuMF,LightGCN,NGCF,FISM}.csv` · shows every top trial running far past 20 epochs.*

*Figure: validation NDCG@10 against `reg_weight` from `results/tuning/EASE.csv` · shows the flat optimum around the course value.*

| Hypothesis | Verdict |
| --- | --- |
| Pop beats Random by a wide margin | Holds with 0.110 against 0.007 |
| Linear item models are strong on ml-100k | Holds as SLIMElastic and EASE lead and tie |
| A tuned BPR matches or beats NeuMF | Holds with 0.314 against 0.304 beyond either spread |
| LightGCN beats NGCF | Not supported as they tie |

**Limitations.**
- Winners on an edge: BPR and LightGCN at the largest batch · LightGCN · NGCF and FISM at the largest embedding · NGCF at the largest node dropout · NeuMF at the smallest embeddings.
- Some low learning rate trials hit the 500 epoch cap but none won.
- Eight models were tuned on CPU and FISM on GPU which can shift close winners but not the reported test numbers.
- SLIMElastic ties at `l1_ratio` 0.001 and 0.0001 and the earlier trial wins.
- NGCF differs in the fourth decimal between identical runs while every other model repeats exactly.

### 1.3 Weighted Hybrid

**Method.**
- **Score.** A weighted sum of the members' scores for a user and movie with weights learned by regression on validation.
- **Members.** The nine tuned models and Pop. Random carries no signal.
- **Scaling.** Each member is standardised per user over the movies outside train since raw scales differ and only the order within a user matters.
- **Candidates.** The union of every member's top 100 movies per user as in the ranking stage of industrial recommenders. The fit then learns the order near the top which is all NDCG@10 sees. They cover 79% of validation movies in about 215,000 rows.
- **Regression.** Logistic regression with L2 on a 0 or 1 label for validation movies. Each weight reads as a member's effect on the odds of a hit and L2 keeps near-duplicate members stable.
- **Strength.** Five-fold cross-validation over users picks C and the spread across folds measures weight stability. Final weights use all users and test is scored once.

Fitting on every movie outside train lost to SLIMElastic even on validation (0.2509 against 0.2589) since the fit spent its effort on obvious misses in the tail. On candidates weak regularisation lets correlated members cancel out (LightGCN −0.90 against NGCF 0.55) while strong regularisation pulls the weights toward a balanced mix.

| C | Cross-Validated NDCG@10 |
| --- | --- |
| 1 | 0.2508 |
| 0.01 | 0.2521 |
| 0.001 | 0.2571 |
| 0.0001 | 0.2642 |
| 0.00001 | 0.2631 |
| 0.0000001 | 0.2630 |

*Figure: cross-validated NDCG@10 against C from `cv` in `results/hybrid/metrics/Weighted.json` with SLIMElastic's 0.2589 as a line · shows ranking rising as weights shrink toward a balanced mix.*

| Model | Test NDCG@10 | Test Recall@10 |
| --- | --- | --- |
| Weighted Hybrid | 0.3378 | 0.2879 |
| EASE | 0.3309 | 0.2808 |
| SLIMElastic | 0.3240 | 0.2769 |

| Member | Weight | Spread Across Folds |
| --- | --- | --- |
| EASE | 0.139 | 0.002 |
| NeuMF | 0.138 | 0.003 |
| SLIMElastic | 0.111 | 0.003 |
| FISM | 0.079 | 0.003 |
| Pop | 0.035 | 0.004 |
| BPR | 0.030 | 0.002 |
| NGCF | 0.029 | 0.002 |
| LightGCN | 0.003 | 0.001 |
| UserKNN | −0.006 | 0.002 |
| ItemKNN | −0.017 | 0.002 |

**Observations.**
- **The hybrid wins narrowly.** It gains 0.007 over EASE which is about one seed spread so the hypothesis holds but only just.
- **Weight goes to members that err differently.** Full item-item weights (EASE and SLIMElastic) · a non-linear MLP (NeuMF) and low rank item similarity (FISM) view the data differently so their mistakes overlap less.
- **Embedding models share one slot.** BPR · NGCF and LightGCN all score by an embedding dot product so once one is in the others add little.
- **Neighbourhood models only correct.** ItemKNN and UserKNN repeat the co-occurrence signal that EASE and SLIMElastic learn better.
- **Shrinkage beats a free fit.** The pointwise loss rewards separating hits from misses rather than ordering the top which matches the blending literature.
- **Weights are stable.** No weight moves more than 0.004 across folds so 2.3 can read them directly.

*Figure: correlation between members' scores on candidates from `results/tuned/scores/*.npz` with train movies hidden via `results/split/train.tsv` · expected to show the embedding and item-item models as two blocks.*

**Limitations.**
- Members were tuned on the same validation set which can only lower the test score. Full stacking would cost hours of retraining for a small effect.
- Only the seed 2020 split has score tables.
- 21% of validation movies fall outside the candidates and can never be ranked. The depth of 100 is a setting for 1.5.

### 1.4 Other Hybrids

The remaining six designs of Burke (2002) on the tuned members and the 1.3 protocol. The first three combine finished score tables. The last three reach inside a model or bring in genres and demographics that no member has seen.

**Method.**
- **Switching.** Users fall into three equal groups by train size. Each group uses the member with the best validation NDCG@10 in that group so it tests whether families win for different users.
- **Mixed.** Every member's ranking is fused by reciprocal rank fusion with k = 60 (Cormack et al. 2009). It learns nothing so it shows what the 1.3 weights add.
- **Cascade.** EASE proposes its top 50 and NeuMF reorders them. The two carry the largest 1.3 weights and come from different families.
- **Feature Combination.** A gradient-boosted classifier on the 1.3 candidates over the 10 member scores plus each movie's genres year and popularity and each user's age gender occupation and activity. It tests whether a learner can find where each member is right.
- **Feature Augmentation.** EASE's top 10 unseen movies per user join train as pseudo-interactions and UserKNN is refit on the denser matrix. It tests whether the strongest model can fix the sparse overlap that user neighbourhoods depend on.
- **Meta-Level.** UserKNN takes its neighbours from the cosine of LightGCN's learned user embeddings instead of raw co-ratings. It tests whether graph propagation finds better peers than direct overlap.

**Proof.** Combination with a linear learner and no side features reproduces Weighted exactly at cross-validated 0.2642 and test 0.3378. The shared neighbourhood on raw ratings reproduces tuned UserKNN for all 943 users once RecBole's padding row is kept since it decides ties at the 50th peer. The saved LightGCN embeddings reproduce its score table to 1e-5.

| Model | Test NDCG@10 | Test Recall@10 |
| --- | --- | --- |
| Weighted (1.3) | 0.3378 | 0.2879 |
| Mixed | 0.3316 | 0.2776 |
| EASE | 0.3309 | 0.2808 |
| Feature Combination | 0.3305 | 0.2852 |
| Switching | 0.3239 | 0.2726 |
| Cascade | 0.3142 | 0.2656 |
| Meta-Level | 0.3023 | 0.2563 |
| Feature Augmentation | 0.2981 | 0.2540 |
| UserKNN | 0.2877 | 0.2524 |

| Train Movies | Switching Member |
| --- | --- |
| 16 to 34 | LightGCN |
| 34 to 92 | SLIMElastic |
| 93 to 591 | EASE |

**Observations.**
- **Short histories go to the graph model.** LightGCN wins for the lightest users since propagation borrows signal from similar users when a user's own history is thin. Item-item models win once histories are long enough to match movie to movie so the hypothesis holds.
- **Switching still trails EASE.** Each group picks from about 314 users so a small validation edge does not carry to test.
- **Equal votes tie the best member.** Fusion counts weak members such as Pop and the neighbourhood models as much as strong ones. The 1.3 weights add about 0.006 on top by muting them.
- **The last stage sets the order.** The cascade shortlist holds half of all test movies but NeuMF orders it worse than EASE does so the result lands beside NeuMF alone.
- **Side data is already inside the members.** Genres and demographics lift a linear learner by only 0.001 in cross-validation from 0.2642 to 0.2654. The default booster reaches 0.3508 in-sample but 0.2520 in cross-validation so its non-linearity overfits more than the side data gives.
- **Densifying helps the users who need it least.** Augmentation lifts UserKNN by 0.010 but its lightest third falls from 0.2450 to 0.2356 while the middle and heaviest thirds gain 0.015 and 0.025. Ten pseudo-interactions are up to 38% of a light user's profile. The hypothesis fails.
- **Learned peers beat co-rated peers but not their source.** Meta-Level lifts UserKNN by 0.015 with the heaviest third gaining 0.037 and the lightest none. It still trails LightGCN at 0.3146 so the embeddings rank better directly than as a similarity.
- **Only linear weights clearly win.** Choosing one member per user or per stage throws away the others while weighting keeps every signal in proportion.

*Figure: validation NDCG@10 of every member per train size group from `results/tuned/scores/*.npz` with `results/split/{train,valid}.tsv` · shows the winner shifting from LightGCN to the item-item models.*

**Limitations.**
- Groups and stages and the booster and the pseudo count are fixed by hand as settings for 1.5.

## Appendix A — Models

| Model | How It Scores |
| --- | --- |
| Random | Uniform random scores drawn afresh at every call |
| Pop | Train count per movie. RecBole also counts sampled negatives so it matches true popularity at a rank correlation of 0.90 |
| ItemKNN | Sum of cosine similarities between a movie and the user's history over the `k` nearest movies |
| UserKNN | Sum of cosine similarities of the `k` nearest users who watched the movie |
| BPR | Dot product of user and movie embeddings trained to rank a seen movie above a sampled unseen one |
| NeuMF | Element-wise product branch plus an MLP branch over user and movie embeddings trained on 0 or 1 labels |
| FISM | Sum of the user's movie embeddings scaled by $\lvert N(u) \rvert^{-\alpha}$ against the candidate's embedding |
| LightGCN | Embeddings propagated over the user-movie graph and averaged across layers trained with the BPR loss |
| NGCF | Graph propagation with learned weights · non-linearity and dropout with layers concatenated |
| EASE | Closed-form item-item matrix minimising $\lVert R - RB \rVert^2 + \lambda \lVert B \rVert^2$ with a zero diagonal |
| SLIMElastic | The same objective as one non-negative elastic net per movie giving a sparse matrix |

## Appendix B — Search Spaces

Log marks a log scale range. Other ranges are choices or uniform intervals.

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

Fixed: LightGCN uses one negative as in its paper and SLIMElastic keeps non-negative weights.
