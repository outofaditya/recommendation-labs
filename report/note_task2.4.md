# Task 2.4 — Explaining hybrid recommender behaviour

Task 2.2 compared overall effectiveness, and Task 2.3 examined member contributions. In Task 2.4, we first ran three analyses at the user, candidate-pipeline, and item-popularity levels. We then added Experiment 5, a specialized long-tail candidate source, and Experiment 6, a comparison by users' training-time tail interest. Together, they identify whether additional tail items can be retrieved and whether the current reranker can use them.

1. **Per-user model wins:** measured how model performance varies across users and activity groups, and estimated the upper bound for user-specific routing.
2. **Candidate recall versus reranking:** tested how candidate-generator choice and pool depth affect retrieval recall and final ranking quality.
3. **Popularity and long-tail exposure:** measured which popularity groups appear in recommendations and whether exposed tail items match held-out interactions.

## Experiment 1 — Per-user model wins and routing headroom

### Analysis

For each of 943 test users, calculate NDCG@10 for six tuned individual models (EASE, SLIMElastic, LightGCN, FISM, BPR, NeuMF) and two hybrids (Weighted and Feature Combination). Group users into quartiles by training interaction count and profile genre entropy. The model with the highest per-user NDCG is recorded as that user's winner. The oracle score averages each user's best test NDCG and is an upper bound only: it uses test outcomes to select the model and is not a deployable routing policy.

### Outputs

- `results/task24/routing_per_user.csv`: per-user model NDCG and winner.
- `results/task24/routing_segments.csv`: win rates and mean NDCG by activity and genre-entropy quartile.
- `results/task24/routing_summary.json`: overall win shares and oracle gap with user-bootstrap interval.
- `results/task24/routing_wins_by_activity.png`: model-win rates by activity quartile.

The oracle reaches NDCG@10 **0.4365**, versus **0.3399** for Feature Combination. The paired user-bootstrap gap is **0.0966**, with a 95% interval of **[0.0908, 0.1030]**. Across users, EASE is the best single model for 23.97%, NeuMF for 17.82%, SLIMElastic for 13.47%, LightGCN for 13.04%, FISM for 11.66%, BPR for 8.59%, Feature Combination for 8.27%, and Weighted for 3.18%.

| Activity quartile | EASE | LightGCN | SLIMElastic | Weighted | Feature Combination |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1, least active | 0.2610 | 0.2490 | 0.2612 | 0.2658 | **0.2694** |
| 2 | 0.2754 | 0.2435 | 0.2609 | 0.2746 | **0.2770** |
| 3 | 0.2953 | 0.2842 | 0.2879 | **0.3034** | 0.3001 |
| 4, most active | 0.4918 | 0.4814 | 0.4859 | 0.5061 | **0.5131** |

### Explanation

The oracle gap indicates substantial user-level variation: no single model is best for every user. However, the deployed value of routing is unknown until a gate is fitted on validation users and evaluated on held-out users. Feature Combination has the highest mean NDCG in three of the four activity quartiles; Weighted leads the third. LightGCN is not the mean-NDCG leader for the least-active users, so the simple rule “route sparse users to the graph model” is not supported here. EASE wins the largest share of users in the first two activity quartiles, while NeuMF has the highest win share in the most-active quartile (19.9%). These per-user wins motivate a validation-trained gate, but do not establish that the gate will generalize.

The executed per-user comparison includes six individual models plus Weighted and Feature Combination. It does not include the saved Switching export.

## Experiment 2 — Candidate recall versus reranking quality

### Analysis

Use each selected individual or hybrid export to generate unseen candidates at depths 20, 50, 100, 200, and full available pool. Rerank those candidates with each of the six individual exports, which provide scores across the catalogue. This is important because hybrid exports have finite scores only within their saved candidate pools; treating their `-inf` values as scores for other candidates would create invalid rankings. The run evaluates 240 generator/reranker/depth combinations. Reranker score availability is effectively complete (minimum `ranker_available_fraction` 0.99997).

### Outputs

- `results/task24/candidate_pipeline.csv`: candidate recall, catalogue coverage, NDCG, conditional NDCG, and score availability for every combination.
- `results/task24/candidate_pipeline_top20.csv`: highest-NDCG combinations.
- `results/task24/protocol.json`: run settings, input hashes, and limitations.

The highest observed NDCG@10 is **0.3351**, from LightGCN candidate generation at depth 20 followed by SLIMElastic reranking. For that same pair:

| Candidate depth | Candidate recall | NDCG@10 | Conditional NDCG@10* |
| ---: | ---: | ---: | ---: |
| 20 | 0.3940 | **0.3351** | 0.3696 |
| 50 | 0.5817 | 0.3260 | 0.3371 |
| 100 | 0.7414 | 0.3251 | 0.3286 |
| 200 | 0.8716 | 0.3242 | 0.3249 |
| Full | 1.0000 | 0.3240 | 0.3240 |

*Conditional NDCG is averaged over users whose candidate pool contains at least one relevant test item.

The same pattern appears for NeuMF retrieval and EASE reranking: recall rises from 0.3910 at depth 20 to 0.7384 at depth 100 and 1.0000 for the full catalogue; NDCG changes from 0.3328 to 0.3315 and 0.3309, respectively.

### Explanation

Deeper pools recover more relevant items, but do not necessarily improve the final top ten. In the examples above, the smaller pool gives the reranker a more focused set and yields higher NDCG. The result shows that retrieval recall and final ranking quality measure different stages. A production system should select retrieval depth jointly with ranking quality and serving cost; recall alone is not a reason to increase depth. The reported best pair is descriptive because it was identified on this test split, not selected on validation data.

## Experiment 3 — Popularity exposure and long-tail relevance

### Analysis

Rank catalogue items by training interaction count and assign the top 20% to head, the next 30% to middle, and the bottom 50% to tail. For selected individual models and hybrids, measure the share of top-10 positions exposed to each bucket and the recall of relevant test items within each bucket. These metrics distinguish how often items are shown from whether shown items match held-out interactions.

### Outputs

- `results/task24/popularity_buckets.csv`: bucket sizes, training-interaction share, exposure, NDCG contribution, and within-bucket recall.
- `results/task24/popularity_exposure.csv`: recommendation exposure by model and popularity bucket.
- `results/task24/popularity_exposure.png`: stacked exposure chart.

The head bucket contains 337 items (20% of the catalogue) but accounts for **65.00%** of training interactions. Top-10 exposure is more concentrated:

| Model | Head exposure | Middle exposure | Tail exposure | Tail recall@10* |
| --- | ---: | ---: | ---: | ---: |
| EASE | 98.77% | 1.23% | 0.00% | 0.00% |
| Weighted | 98.19% | 1.77% | 0.04% | 0.45% |
| Feature Combination | 96.45% | 3.49% | 0.06% | 0.45% |
| LightGCN | 92.60% | 6.89% | 0.51% | 0.98% |
| FISM | 91.73% | 7.99% | 0.29% | 0.46% |
| NeuMF | 89.71% | 9.19% | 1.10% | 0.96% |

*Within-bucket recall is averaged over users who have at least one relevant item in that popularity bucket.

### Explanation

Every selected model exposes the head bucket more than its 65% share of training interactions would suggest. NeuMF has the largest tail exposure (1.10%) and LightGCN has the highest measured tail recall (0.98%, narrowly above NeuMF at 0.96%). Feature Combination has the strongest overall hybrid ranking result, but its tail exposure is only 0.06% and its tail recall is 0.45%. Its aggregate accuracy gain therefore does not amount to a long-tail gain. A long-tail product objective would need an explicit exposure or retrieval policy, evaluated against its relevance cost.

## Experiment 5 — Specialized long-tail candidate retrieval

### Analysis

Start with the top 100 finite-scoring candidates from the saved Feature Combination export. Add 20 unique items beyond this base pool from one source at a time: LightGCN, NeuMF, or a genre-aware tail retriever. The genre-aware retriever ranks unseen tail items by overlap with genre preferences estimated from each user's training history; training popularity breaks ties. Also test the union of LightGCN and genre-aware additions. Rerank every pool with the same SLIMElastic scores to isolate candidate-source effects. Tail items use the train-defined bottom-50% popularity bucket.

### Outputs

- `results/task24/specialized_retrieval.csv`: candidate-pool size and coverage, tail candidate recall, NDCG, overall recall, tail exposure, tail recall, and tail hit rate.
- `results/task24/specialized_retrieval_per_user.csv`: per-user strategy results.
- `results/task24/specialized_retrieval_deltas.csv`: paired user-bootstrap differences from the base pool.
- `results/task24/protocol.json`: candidate construction and run settings.

| Candidate strategy | Mean pool size | Tail candidate recall | NDCG@10 | Tail exposure | Tail recall@10 | Tail recommendations / hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Feature Combination top 100 | 100.0 | 11.41% | 0.3240 | 0.0212% | 0.30% | 2 / 1 |
| + 20 unique LightGCN items | 120.0 | 14.89% | 0.3240 | 0.0212% | 0.30% | 2 / 1 |
| + 20 unique NeuMF items | 120.0 | 16.71% | 0.3240 | 0.0212% | 0.30% | 2 / 1 |
| + 20 genre-aware tail items | 120.0 | 15.95% | 0.3240 | 0.0212% | 0.30% | 2 / 1 |
| + LightGCN and genre-aware tail items | 139.9 | 19.42% | 0.3240 | 0.0212% | 0.30% | 2 / 1 |

Relative to the base pool, the genre-aware source raises tail candidate recall by **4.54 percentage points** (paired user-bootstrap 95% CI **[2.84, 6.46]**). NeuMF adds 5.30 points (95% CI [3.42, 7.37]), LightGCN adds 3.48 points (95% CI [2.11, 5.06]), and the LightGCN plus genre-tail union adds 8.01 points (95% CI [5.75, 10.54]). Candidate-recall comparisons are averaged over users with at least one held-out tail interaction.

### Explanation

The specialized source retrieves additional relevant tail items: candidate recall rises from 11.41% to 15.95%, and the union reaches 19.42%. However, the fixed SLIMElastic reranker produces the same NDCG@10, overall Recall@10, tail exposure, and tail recall for every strategy at the reported precision. Across 943 users, only two tail items appear in the final top tens and one is relevant. The new candidates enter the pool but are not promoted into the final list. This locates the current bottleneck at reranking for this setup, while showing that the specialized source can improve retrieval coverage. The tail hit rate is 1/2, so raw counts are included because this estimate is highly unstable.

## Experiment 6 — Results by users' training-time tail interest

### Analysis

Compute each user's share of training interactions that belong to the train-defined tail bucket, then split users into four equal-sized quartiles. Compare the same candidate strategies as Experiment 5 within each group. Report tail candidate recall among users with at least one held-out tail interaction, along with the size of that user subset. Training interactions determine groups; test interactions are only used for evaluation.

### Outputs

- `results/task24/tail_interest_segments.csv`: segment-level candidate recall, tail exposure, tail recall, and NDCG.
- `results/task24/tail_interest_per_user.csv`: user-level segment and strategy results.

| Tail-interest quartile | Mean tail share in train history | Users with tail test items | Base candidate recall | + Genre-tail source | + LightGCN source | + NeuMF source | + Both sources |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, lowest | 0.06% | 22 / 236 | 15.91% | **22.73%** | 15.91% | 15.91% | **22.73%** |
| 2 | 2.80% | 72 / 236 | 6.02% | **13.43%** | 6.02% | 8.80% | **13.43%** |
| 3 | 5.70% | 93 / 235 | 4.97% | 9.01% | 7.66% | 9.68% | **11.72%** |
| 4, highest | 13.26% | 147 / 236 | 17.45% | 20.57% | 23.66% | 25.16% | **26.75%** |

The segment file also reports bootstrap intervals for each strategy's candidate recall. They are wide in quartile 1 because only 22 users have tail test items: base recall is 15.91% (95% CI [2.27, 31.82]) and genre-tail recall is 22.73% ([6.82, 40.91]). In quartile 4, the base is 17.45% ([12.25, 23.11]) and the LightGCN plus genre-tail union is 26.75% ([20.86, 32.82]). These are intervals for each strategy's group mean, not paired intervals for the difference.

NDCG@10 is unchanged across candidate strategies within each quartile: 0.2735, 0.3061, 0.3503, and 0.3661. Tail exposure and tail recall are also unchanged: the only observed tail hits are in quartile 1, where mean tail recall is 4.55% and mean tail exposure is 0.0847%. Quartiles 2–4 have no tail item in the final top ten.

### Explanation

The genre-aware source improves tail candidate recall in the first two groups, with its largest gain in quartile 2 (6.02% to 13.43%). In quartile 3, the LightGCN plus genre-tail union reaches 11.72%. In the highest-interest quartile, model-based retrieval is stronger: NeuMF reaches 25.16%, and the union reaches 26.75%. This suggests that useful retrieval sources may vary with users' training histories, but these group patterns have not been used to train or validate a routing policy.

The unchanged NDCG and tail exposure reinforce the Experiment 5 diagnosis: candidate additions improve tail retrieval, but SLIMElastic does not surface those items. A reranker that incorporates tail relevance or an exposure-aware ranking policy is needed before retrieval gains become visible to users. Any such policy should be selected on validation users and evaluated for its NDCG cost.

## Overall interpretation

Task 2.4 ran five analyses: per-user model wins, candidate-depth effects, popularity exposure, specialized tail retrieval, and results by users' training-time tail interest. The first showed user-level variation, while the oracle score remains only an upper bound. The second showed that deeper pools raise recall without necessarily improving NDCG@10. The third found that Feature Combination's strong aggregate accuracy does not translate into long-tail exposure: 96.45% of its recommendations come from the top 20% of items, and only 0.06% from the bottom half. Experiments 5 and 6 showed that additional candidate sources can improve tail candidate recall and that the most useful source varies across tail-interest groups. In this run, the fixed reranker did not turn those extra candidates into additional tail recommendations or higher NDCG.

For this split, Feature Combination is the strongest default by aggregate NDCG and by mean NDCG in three activity quartiles; Weighted has the highest mean NDCG in the remaining quartile. Candidate depth should be selected using retrieval recall and reranking quality together. A long-tail product objective requires an explicit ranking policy and a measured relevance cost. Experiments 5 and 6 locate the immediate bottleneck in reranking: retrieval improved, while the final recommendation lists did not.

## Protocol and limitations

The run was executed from the repository root with:

```bash
uv run python -m scripts.task24.task24_pipeline
```

It uses the corrected shared MovieLens 100K split, 943 test users, K=10, and fixed saved exports. The oracle uses test outcomes, Switching was omitted from the per-user comparison, and the run does not measure retraining variance or online feedback. Candidate-source and depth comparisons are exploratory on one test split; they have not been selected on validation users and should not be treated as production policies. See `results/task24/protocol.json` for input hashes and run metadata.

## Planned extension — Stability under history and score perturbations

This experiment was not executed in the current Task 2.4 run. It is a follow-up to test whether the observed model differences and recommendation lists are stable under small changes that can occur in a live system.

For each user, perturb the input in three controlled ways: remove one randomly selected training interaction, remove the newest 10% of interactions when timestamps are available, and add small fixed-seed noise to member scores. For each perturbation, compare the original and perturbed top-10 lists using overlap and rank correlation, and measure changes in NDCG@10, catalogue coverage, and any selected route or member weights. Repeat interaction removal and score noise over multiple fixed seeds and summarize user-level means and confidence intervals.

This would establish whether modest offline gains are robust enough to justify extra routing complexity, and could inform refresh cadence and monitoring. It should be reported as a separate follow-up once implemented; no stability result is claimed here.
