# Task 2.4 — Explaining Hybrid Recommender Behaviour

## Repeated-run design

The behavioural analyses were repeated for seeds 2020–2024. Each run rebuilt the fixed model set and repeated the routing, candidate, popularity, specialized-retrieval, and tail-interest analyses. Values are means over five runs; intervals are 95% Student-t intervals across runs. The activity and tail-interest groups are formed from training data in each run. These intervals summarize seed-to-seed variation and are not user-bootstrap intervals.

## Per-user model preference and routing headroom

The oracle selects the best model for each user using that user's test outcomes. It is an upper bound, not an implementable routing policy.

| Measure | Mean | 95% CI |
| --- | ---: | ---: |
| Per-user oracle NDCG@10 | 0.4335 | [0.4248, 0.4422] |
| Feature Combination NDCG@10 | 0.3373 | [0.3310, 0.3437] |
| Oracle − Combination | 0.0961 | [0.0933, 0.0990] |

EASE has the largest per-user win share at 24.4% [23.4%, 25.3%], followed by SLIMElastic at 15.5% [13.5%, 17.5%], LightGCN at 13.7% [13.2%, 14.3%], and NeuMF at 13.4% [12.0%, 14.9%]. The remaining wins are distributed among FISM, BPR, Combination, and Weighted. This consistent spread shows that the best model varies by user, but the oracle gap overstates attainable routing gains because it uses test outcomes.

The leading model by mean segment NDCG changes with activity: SLIMElastic is narrowly highest in group 1 (0.2719), Weighted in groups 2 and 3 (0.2724 and 0.3053), and Combination in group 4 (0.5051). The leading models' intervals overlap within each segment, so the point leaders do not establish a robust gate.

## Candidate retrieval and reranking

In the candidate-pipeline sweep, the highest mean NDCG@10 was LightGCN retrieval at depth 20 followed by SLIMElastic reranking: 0.3332 [0.3259, 0.3406], with candidate recall 0.3858 [0.3759, 0.3956]. Combination retrieval at depth 20 with the same reranker had candidate recall 0.4097 [0.3998, 0.4196] and NDCG@10 0.3316 [0.3234, 0.3399]. These are exploratory comparisons across many pipeline configurations; they do not identify a validated serving policy.

Retrieval changes do not guarantee changes in the final top ten. With a fixed SLIMElastic reranker, the Combination top-100 pool recovered 10.82% of relevant tail items among users with held-out tail items. Adding 20 unique candidates from LightGCN, NeuMF, or a genre-tail source, or combining LightGCN and genre-tail sources, increased mean tail candidate recall:

| Candidate strategy | Tail candidate recall | Paired gain vs. base (95% CI) | NDCG@10 | Tail exposure in top 10 |
| --- | ---: | ---: | ---: | ---: |
| Combination top 100 (base) | 10.82% | — | 0.32890 | 0.0191% |
| + LightGCN top 20 | 15.51% | +4.69 pp [3.96, 5.43] | 0.32888 | 0.0191% |
| + NeuMF top 20 | 16.99% | +6.18 pp [5.69, 6.67] | 0.32890 | 0.0191% |
| + Genre-preference tail top 20 | 15.94% | +5.13 pp [4.05, 6.20] | 0.32890 | 0.0191% |
| + LightGCN and genre-tail | 20.22% | +9.40 pp [7.95, 10.85] | 0.32888 | 0.0191% |

All additions left NDCG and tail exposure effectively unchanged under the fixed reranker. The added candidates were retrieved, but almost none reached the recommendation list: the base produced an average of 1.8 tail recommendations and 0.2 tail hits across 943 users per run. This points to reranking as the bottleneck in this experiment, rather than candidate availability alone. The tiny hit count also makes the tail-hit rate unstable.

## Popularity exposure

Items are grouped by training popularity: head is the top 20% (337 items), middle is the next 30% (504), and tail is the bottom 50% (841). Exposure is the share of top-ten recommendation positions assigned to a bucket.

| Model | Head exposure | Tail exposure |
| --- | ---: | ---: |
| EASE | 98.66% | 0.00% |
| SLIMElastic | 98.26% | 0.019% |
| Weighted | 98.19% | 0.019% |
| Combination | 96.36% | 0.083% |
| LightGCN | 91.67% | 0.573% |
| FISM | 92.12% | 0.250% |
| NeuMF | 93.37% | 1.12% |

Even for the less head-concentrated neural models, tail items occupy a small fraction of recommendation positions. Strong overall NDCG therefore does not imply broad long-tail exposure.

The following K-sweep plots show catalogue-level coverage and novelty for context. They summarize discovery breadth and item popularity, while the exposure table above directly measures head-versus-tail placement.

![Catalogue coverage by recommendation-list length](../results/task2/multiseed/aggregate/plots/coverage_by_k.png)

![Novelty by recommendation-list length](../results/task2/multiseed/aggregate/plots/novelty_by_k.png)

## Tail interest segments

Users were divided into quartiles by the share of their training interactions involving tail items. The union of LightGCN and genre-tail additions increased tail candidate recall over the Combination top-100 base in every quartile:

| Tail-interest quartile | Users with tail test items (mean) | Base tail candidate recall | + LightGCN and genre-tail | Change |
| --- | ---: | ---: | ---: | ---: |
| 1 (lowest interest) | 21 | 15.70% | 19.94% | +4.23 pp |
| 2 | 65 | 2.45% | 9.25% | +6.80 pp |
| 3 | 94 | 7.40% | 15.72% | +8.32 pp |
| 4 (highest interest) | 146 | 16.02% | 28.02% | +12.01 pp |

These averages show the largest retrieval gain among high-tail-interest users, but candidate additions did not materially change NDCG, top-ten tail exposure, or tail recall in the segments. Quartiles 1 and 2 have especially few users with held-out tail items, so their tail estimates are less informative.

## Interpretation and limits

The repeated runs confirm two distinct behaviours: model preference varies across users, and specialized retrieval expands the pool of relevant tail candidates. Yet the fixed ranking stage rarely promotes those extra candidates, and popularity exposure remains strongly concentrated in the head. A follow-up should evaluate a tail-aware reranker or ranking feature using validation-based selection and report the resulting accuracy cost. The oracle and the pipeline sweep are exploratory; neither establishes a deployable routing or retrieval policy. Five seeds remain a small sample. Full per-seed routing, candidate, popularity, and segment summaries are in [`results/task2/multiseed/aggregate/task24/`](../results/task2/multiseed/aggregate/task24/).
