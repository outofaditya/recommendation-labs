# Task 2.4 — Hybrid Behaviour and Long-Tail Analysis

## Experimental setting

The analyses use five runs (seeds 2020–2024), MovieLens 100K (943 users and 1,682 items), K=10, and fixed tuned Task 1 configurations. Means and 95% Student-t intervals are across runs. Routing evaluates per-user results for EASE, SLIMElastic, LightGCN, FISM, BPR, NeuMF, Weighted, and Combination. The candidate pipeline tests 8 generators × 6 individual rerankers × 5 depths (20, 50, 100, 200, and the full available pool). Popularity groups are defined by training counts: top 20%, next 30%, and bottom 50%. Tail-retrieval experiments start from Combination's top 100 unseen candidates, add 20 unique candidates from specialized sources, and use the same SLIMElastic scores to rerank every pool.

## Experiment 1 — User-level routing and activity segments

An oracle chooses the best model for each user using that user's test outcomes. It is an upper bound and cannot be deployed as a routing rule.

| NDCG@10 measure | Mean [95% CI] |
| --- | ---: |
| Per-user oracle across eight models | 0.4335 [0.4248, 0.4422] |
| Combination | 0.3373 [0.3310, 0.3437] |
| Oracle − Combination | 0.0961 [0.0933, 0.0990] |

The largest per-user win share belongs to EASE (24.37% [23.43, 25.31]), followed by SLIMElastic (15.52% [13.54, 17.51]), LightGCN (13.72% [13.19, 14.25]), and NeuMF (13.43% [11.98, 14.87]). Preference varies by user, but this test-based oracle gap is not an estimate of achievable routing gain.

By training activity, the highest mean segment NDCG comes from SLIMElastic in group 1 (0.2719), Weighted in groups 2–3 (0.2724 and 0.3053), and Combination in group 4 (0.5051). Their intervals overlap within the groups, so these point leaders do not establish a stable routing policy.

## Experiment 2 — Candidate generation and reranking

The highest mean NDCG in the 240-setting pipeline sweep is LightGCN retrieval at depth 20 followed by SLIMElastic reranking: NDCG@10 0.3332 [0.3259, 0.3406], candidate recall 0.3858 [0.3759, 0.3956], and conditional NDCG 0.3714. Combination retrieval at depth 20 with the same reranker has higher candidate recall (0.4097 [0.3998, 0.4196]) but lower NDCG (0.3316 [0.3234, 0.3399]). This test-split sweep is exploratory, not a validation-selected serving policy.

## Experiment 3 — Popularity exposure

The top 20% of items account for about 64.8% of training interactions, yet receive most top-10 recommendations. Values below are mean exposure shares over the five runs.

| Model | Head exposure (top 20%) | Tail exposure (bottom 50%) |
| --- | ---: | ---: |
| EASE | 98.66% | 0.000% |
| SLIMElastic | 98.26% | 0.019% |
| Weighted | 98.19% | 0.013% |
| Combination | 96.36% | 0.081% |
| LightGCN | 91.67% | 0.573% |
| FISM | 92.12% | 0.250% |
| NeuMF | 93.37% | 1.122% |

All selected models remain head-heavy; higher overall ranking accuracy does not imply substantial long-tail exposure.

The K-sweep plots below show catalogue-wide coverage and novelty, which describe discovery breadth and item popularity. The table above is the direct measure of head-versus-tail exposure.

![Catalogue coverage by recommendation-list length](plots/coverage_by_k.png)

![Novelty by recommendation-list length](plots/novelty_by_k.png)

## Experiments 4–5 — Specialized tail retrieval and tail-interest groups

The base candidate set is Combination's top 100. Adding 20 unique candidates increases catalogue coverage and tail candidate recall, but the fixed SLIMElastic reranker barely changes the final list:

| Candidate strategy | Candidate catalogue coverage | Tail candidate recall | Paired recall gain vs base [95% CI] | NDCG@10 | Tail exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Combination top 100 | 69.58% | 10.82% | — | 0.32890 | 0.0191% |
| + LightGCN top 20 | 76.62% | 15.51% | +4.69 pp [3.96, 5.43] | 0.32888 | 0.0191% |
| + NeuMF top 20 | 81.90% | 16.99% | +6.18 pp [5.69, 6.67] | 0.32890 | 0.0191% |
| + Genre-preference tail top 20 | 75.26% | 15.94% | +5.13 pp [4.05, 6.20] | 0.32890 | 0.0191% |
| + LightGCN and genre-tail | 80.55% | 20.22% | +9.40 pp [7.95, 10.85] | 0.32888 | 0.0191% |

Tail recall@10 does not change (mean 0.060%); the base produces only 1.8 tail recommendations and 0.2 tail hits per run across 943 users. Retrieval therefore finds relevant tail items that the fixed reranker rarely puts in the top ten. The extremely small tail-hit count makes the hit rate unstable.

Tail candidate recall for the base and the LightGCN-plus-genre union, by users' training-time tail-interest quartile, is:

| Tail-interest quartile | Users with tail test items (mean) | Base recall | Union recall | Change |
| --- | ---: | ---: | ---: | ---: |
| 1 (lowest interest) | 21 | 15.70% | 19.94% | +4.23 pp |
| 2 | 65 | 2.45% | 9.25% | +6.80 pp |
| 3 | 94 | 7.40% | 15.72% | +8.32 pp |
| 4 (highest interest) | 146 | 16.02% | 28.02% | +12.01 pp |

Retrieval gains are largest in the highest-interest group, but final NDCG and tail exposure remain effectively unchanged within groups. Groups 1 and 2 have few users with held-out tail items.

## Main result 

Across these experiments, user preferences differ and additional sources improve tail candidate recall, but neither the oracle nor the candidate sweep establishes a deployable policy. The repeated runs show that the current fixed reranker is the bottleneck for converting added tail candidates into recommendations. A tail-aware ranking method should be evaluated with validation-based selection and its accuracy cost measured. 