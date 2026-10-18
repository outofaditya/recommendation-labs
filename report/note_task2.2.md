# Task 2.2 — Evaluation of Effectiveness

## Repeated-run protocol

Results aggregate five runs (seeds 2020–2024) at K = 1, 3, 5, 10, and 20. Each run refits the nine tuned individual models and rebuilds the seven hybrid designs using the fixed reference hyperparameters; learned hybrid parameters are refit from that run's validation data. Tables below focus on K=10. Values are seed means; intervals are two-sided 95% Student-t intervals across the five runs. They describe run-to-run variability for this setup, not user-bootstrap intervals. The same test protocol is applied in each run.

## Ranking effectiveness at K=10

| Model | Type | NDCG@10 (mean [95% CI]) | Recall@10 | Precision@10 | MRR@10 | Hit@10 | MAP@10 | F1@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | baseline | 0.0079 [0.0068, 0.0089] | 0.0058 | 0.0065 | 0.0197 | 0.0602 | 0.0029 | 0.0050 |
| Pop | baseline | 0.1095 [0.0881, 0.1309] | 0.0947 | 0.0856 | 0.2081 | 0.4899 | 0.0508 | 0.0729 |
| ItemKNN | individual | 0.2773 [0.2717, 0.2829] | 0.2358 | 0.1845 | 0.4632 | 0.7631 | 0.1680 | 0.1674 |
| UserKNN | individual | 0.2845 [0.2727, 0.2962] | 0.2453 | 0.1912 | 0.4718 | 0.7811 | 0.1707 | 0.1744 |
| BPR | individual | 0.3138 [0.3058, 0.3217] | 0.2672 | 0.2124 | 0.5065 | 0.8098 | 0.1935 | 0.1926 |
| NeuMF | individual | 0.3034 [0.2981, 0.3087] | 0.2588 | 0.2077 | 0.4917 | 0.7958 | 0.1858 | 0.1871 |
| FISM | individual | 0.2840 [0.2761, 0.2918] | 0.2485 | 0.1935 | 0.4677 | 0.7913 | 0.1676 | 0.1773 |
| LightGCN | individual | 0.3091 [0.3042, 0.3140] | 0.2614 | 0.2112 | 0.5005 | 0.8034 | 0.1902 | 0.1900 |
| NGCF | individual | 0.3118 [0.3050, 0.3186] | 0.2659 | 0.2126 | 0.5044 | 0.8083 | 0.1917 | 0.1918 |
| EASE | individual | 0.3266 [0.3198, 0.3335] | 0.2783 | 0.2201 | 0.5218 | 0.8165 | 0.2048 | 0.1998 |
| SLIMElastic | individual | 0.3289 [0.3206, 0.3371] | 0.2789 | 0.2201 | 0.5252 | 0.8176 | 0.2080 | 0.1998 |
| TunedWeighted | hybrid | 0.3380 [0.3304, 0.3455] | 0.2875 | 0.2276 | 0.5355 | 0.8303 | 0.2144 | 0.2066 |
| TunedCombination | hybrid | 0.3373 [0.3310, 0.3437] | 0.2861 | 0.2276 | 0.5351 | 0.8314 | 0.2140 | 0.2060 |
| TunedMixed | hybrid | 0.3343 [0.3267, 0.3419] | 0.2809 | 0.2251 | 0.5331 | 0.8199 | 0.2123 | 0.2030 |
| TunedCascade | hybrid | 0.3314 [0.3236, 0.3392] | 0.2816 | 0.2227 | 0.5277 | 0.8246 | 0.2092 | 0.2023 |
| TunedSwitching | hybrid | 0.3224 [0.3168, 0.3280] | 0.2721 | 0.2178 | 0.5188 | 0.8110 | 0.2021 | 0.1969 |
| TunedAugmentation | hybrid | 0.2971 [0.2877, 0.3066] | 0.2514 | 0.1976 | 0.4925 | 0.7892 | 0.1815 | 0.1797 |
| TunedMetalevel | hybrid | 0.3005 [0.2914, 0.3096] | 0.2509 | 0.2022 | 0.4927 | 0.7858 | 0.1859 | 0.1812 |

Weighted is the leading hybrid at K=10, with Combination close behind; both exceed the best individual, SLIMElastic, by about 0.009 and 0.008 NDCG points respectively. Weighted also leads hybrid Recall, MRR, MAP, and F1, while Combination has the highest hybrid Hit@10. At other cutoffs, Weighted leads at K=1 and 3, Mixed narrowly leads at K=5, and Combination leads at K=20. Thus the preferred blend depends somewhat on list length, although these leaders are based on point means and the intervals overlap.

![Mean NDCG by recommendation-list length across the five runs](../results/task2/multiseed/aggregate/plots/ndcg_by_k.png)

The K sweep shows how the accuracy ranking changes with recommendation-list length; the table above gives the detailed K=10 comparison.

## Beyond-accuracy results and trade-offs

| Model | Coverage@10 | Novelty@10 | ILD@10 | Serendipity@10 |
| --- | ---: | ---: | ---: | ---: |
| LightGCN | 0.3740 | 8.7910 | 0.7852 | 0.1708 |
| FISM | 0.3593 | 8.7637 | 0.8032 | 0.1567 |
| TunedCombination | 0.2881 | 8.5959 | 0.8038 | 0.1845 |
| TunedWeighted | 0.2411 | 8.5257 | 0.8068 | 0.1846 |
| SLIMElastic | 0.2352 | 8.5400 | 0.8125 | 0.1791 |
| TunedCascade | 0.2517 | 8.5614 | 0.8083 | 0.1808 |

Among hybrids, Combination covers the largest share of the catalogue; Weighted has the highest mean serendipity, effectively tied with Combination. Individual LightGCN and FISM have higher coverage and novelty, while EASE and SLIMElastic have higher intra-list diversity than the hybrids. Accuracy and catalogue coverage have only a weak positive association across non-baseline configurations at K=10 (mean Pearson r=0.12); novelty and ILD correlations are also weak. Serendipity correlates strongly with NDCG (r=0.99), but this metric includes relevance and should not be interpreted as independent evidence of diversity.

![Catalogue coverage by recommendation-list length](../results/task2/multiseed/aggregate/plots/coverage_by_k.png)

![Novelty by recommendation-list length](../results/task2/multiseed/aggregate/plots/novelty_by_k.png)

![Serendipity by recommendation-list length](../results/task2/multiseed/aggregate/plots/serendipity_by_k.png)

Across the five runs, Weighted and Combination are the most consistent high-accuracy hybrid choices at K=10. No model leads every beyond-accuracy dimension, so selection depends on whether accuracy, catalogue coverage, diversity, or serendipity is the priority. Relative percentage gains over Random are omitted because its near-zero NDCG makes them misleading.

## Uncertainty and interpretation

The aggregate reports seed mean, sample standard deviation, and 95% Student-t interval. Five runs are still a small sample, and both split generation and training randomness vary across seeds. Hybrid hyperparameters remain fixed from the reference selection; these results measure repeatability under those choices rather than uncertainty from retuning. Intervals are descriptive and overlapping intervals do not establish formal pairwise equivalence or difference. Per-seed metrics, baseline contrasts, ranks, frontiers, and correlations are in [`results/task2/multiseed/aggregate/`](../results/task2/multiseed/aggregate/).
