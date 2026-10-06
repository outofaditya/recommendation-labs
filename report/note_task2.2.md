# Task 2.2 — Evaluation of Effectiveness
7 accuracy: NDCG、Recall、Precision、MRR、Hit、MAP、F1

4 beyond-accuracy: Coverage、Novelty、ILD、Serendipity

## Experiment 1 - Compare tuned individual models

**Setup.** Evaluate the 9 tuned Task 1 individual models, plus 2 baseline Random and Pop on the same saved MovieLens 100K split (943 test users). 
Apply the Task 2.1 accuracy and beyond-accuracy metrics at K = 10, treating test interactions as binary relevance. 
Average user-level metrics across users; resolve score ties by exported item-column order. 

**Results.** All metrics below are measured at K = 10; higher values indicate better ranking accuracy.

| Model | NDCG | Recall | Precision | MRR | Hit | MAP | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 0.0065 | 0.0050 | 0.0052 | 0.0159 | 0.0520 | 0.0024 | 0.0044 |
| Pop | 0.1004 | 0.0850 | 0.0828 | 0.1913 | 0.4719 | 0.0457 | 0.0680 |
| ItemKNN | 0.2783 | 0.2370 | 0.1829 | 0.4636 | 0.7667 | 0.1691 | 0.1670 |
| UserKNN | 0.2877 | 0.2524 | 0.1909 | 0.4770 | 0.7932 | 0.1722 | 0.1760 |
| BPR | 0.3197 | 0.2683 | 0.2137 | 0.5197 | 0.8155 | 0.1989 | 0.1934 |
| NeuMF | 0.3108 | 0.2618 | 0.2116 | 0.5004 | 0.7964 | 0.1935 | 0.1903 |
| FISM | 0.2873 | 0.2488 | 0.1911 | 0.4852 | 0.8006 | 0.1693 | 0.1765 |
| LightGCN | 0.3146 | 0.2661 | 0.2131 | 0.5138 | 0.8240 | 0.1932 | 0.1922 |
| NGCF | 0.3183 | 0.2717 | 0.2157 | 0.5131 | 0.8144 | 0.1975 | 0.1951 |
| EASE | **0.3309** | **0.2808** | **0.2208** | **0.5316** | 0.8229 | **0.2078** | **0.2011** |
| SLIMElastic | 0.3240 | 0.2769 | 0.2185 | 0.5135 | 0.8134 | 0.2037 | 0.1981 |

Beyond-accuracy results use catalogue coverage, self-information novelty, genre-Jaccard intra-list diversity (ILD), and relevance-weighted genre-distance serendipity.

| Model | Coverage@10 | Novelty@10 | ILD@10 | Serendipity@10 |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.0357 | **12.6636** | 0.6868 | 0.0043 |
| Pop | 0.0357 | 8.0322 | **0.8151** | 0.0691 |
| ItemKNN | 0.1581 | 8.4162 | 0.8134 | 0.1501 |
| UserKNN | 0.1742 | 8.4174 | 0.8112 | 0.1564 |
| BPR | 0.3050 | 8.6081 | 0.8016 | 0.1733 |
| NeuMF | 0.3573 | 8.6935 | 0.7926 | 0.1723 |
| FISM | **0.3722** | 8.7661 | 0.8042 | 0.1547 |
| LightGCN | 0.3585 | 8.7456 | 0.7863 | 0.1723 |
| NGCF | 0.3109 | 8.6048 | 0.7958 | 0.1749 |
| EASE | 0.2325 | 8.5177 | 0.8099 | **0.1791** |
| SLIMElastic | 0.2354 | 8.5315 | 0.8118 | 0.1775 |

**Discussion.** 
```
All nine tuned models outperform both naïve baselines on every accuracy metric. 

**EASE leads on six of seven**, while LightGCN has the highest Hit@10. 

FISM reaches the widest catalogue coverage (0.3722), and EASE has the highest serendipity (0.1791), showing that the most accurate model can still recommend relevant items that differ from user history. 

Random's extreme novelty (12.6636) does not imply useful recommendations: its serendipity is only 0.0043 as novelty receives no relevance requirement whereas serendipity does. 

Pop combines very low coverage (0.0357) with high ILD (0.8151); one repeated popular list can span several genres while covering little of the catalogue. BPR exceeds NeuMF on every accuracy metric, but NeuMF provides broader coverage and greater novelty. 

JC: EASE is the strongest individual-model accuracy benchmark, but different models each have their strengths in coverage, novelty, and serendipity.
```
**Limitations.** These are single-split results, without significance testing. 

**Reproduce:** `uv run python -m scripts.compare_individuals`

**Source outputs:** `results/task2/experiment1/summary.csv` (11 models and 11 metrics), `comparison.md`, and `protocol.json` (metric definitions, evaluation settings, input hashes and baseline diagnostics).

## Experiment 2 - Compare tuned hybrids with individual models

**Setup.** Reuse 9 tuned individual exports, 7 tuned hybrid exports and both baselines Random and Pop on the same local split as Experiment 1 (943 test users, 1,682 items). 

Evaluate the seven Task 2.1 accuracy metrics and four beyond-accuracy metrics at K = 10. 

EASE is a descriptive reference as it has the highest observed individual NDCG on this split.

**Results.** The tables reproduce the local `uv run python -m scripts.compare_hybrids` output to four decimal places. All seven hybrid rows use tuned configurations. Bold accuracy values mark the highest score across all 18 models; the beyond-accuracy table emphasizes comparison with EASE.

| Model | NDCG@10 | Recall@10 | Precision@10 | MRR@10 | Hit@10 | MAP@10 | F1@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random (baseline) | 0.0065 | 0.0050 | 0.0052 | 0.0159 | 0.0520 | 0.0024 | 0.0044 |
| Pop (baseline) | 0.1004 | 0.0850 | 0.0828 | 0.1913 | 0.4719 | 0.0457 | 0.0680 |
| ItemKNN (individual) | 0.2783 | 0.2370 | 0.1829 | 0.4636 | 0.7667 | 0.1691 | 0.1670 |
| UserKNN (individual) | 0.2877 | 0.2524 | 0.1909 | 0.4770 | 0.7932 | 0.1722 | 0.1760 |
| BPR (individual) | 0.3197 | 0.2683 | 0.2137 | 0.5197 | 0.8155 | 0.1989 | 0.1934 |
| NeuMF (individual) | 0.3108 | 0.2618 | 0.2116 | 0.5004 | 0.7964 | 0.1935 | 0.1903 |
| FISM (individual) | 0.2873 | 0.2488 | 0.1911 | 0.4852 | 0.8006 | 0.1693 | 0.1765 |
| LightGCN (individual) | 0.3146 | 0.2661 | 0.2131 | 0.5138 | 0.8240 | 0.1932 | 0.1922 |
| NGCF (individual) | 0.3183 | 0.2717 | 0.2157 | 0.5131 | 0.8144 | 0.1975 | 0.1951 |
| EASE (individual reference) | 0.3309 | 0.2808 | 0.2208 | 0.5316 | 0.8229 | 0.2078 | 0.2011 |
| SLIMElastic (individual) | 0.3240 | 0.2769 | 0.2185 | 0.5135 | 0.8134 | 0.2037 | 0.1981 |
| Weighted | 0.3375 | 0.2890 | 0.2268 | **0.5363** | 0.8346 | 0.2126 | 0.2069 |
| Mixed | 0.3360 | 0.2822 | 0.2249 | 0.5360 | 0.8282 | 0.2141 | 0.2035 |
| Cascade | 0.3282 | 0.2831 | 0.2231 | 0.5165 | 0.8303 | 0.2057 | 0.2030 |
| Switching | 0.3209 | 0.2728 | 0.2175 | 0.5144 | 0.8208 | 0.1995 | 0.1966 |
| Feature Combination | **0.3399** | **0.2921** | **0.2306** | 0.5322 | **0.8409** | **0.2157** | **0.2098** |
| Feature Augmentation | 0.3027 | 0.2562 | 0.1968 | 0.5075 | 0.7879 | 0.1858 | 0.1804 |
| Meta-Level | 0.3053 | 0.2632 | 0.2053 | 0.5007 | 0.8144 | 0.1865 | 0.1863 |

| Model | Coverage@10 | Novelty@10 | ILD@10 | Serendipity@10 |
| --- | ---: | ---: | ---: | ---: |
| EASE (individual reference) | 0.2325 | 8.5177 | **0.8099** | 0.1791 |
| Weighted | 0.2461 | 8.5243 | 0.8073 | 0.1837 |
| Mixed | 0.2366 | 8.4623 | 0.8066 | 0.1828 |
| Cascade | 0.2503 | 8.5532 | 0.8084 | 0.1807 |
| Switching | 0.2473 | 8.5579 | 0.8069 | 0.1768 |
| Feature Combination | **0.2848** | **8.5807** | 0.8039 | **0.1867** |
| Feature Augmentation | 0.2004 | 8.4134 | 0.8067 | 0.1607 |
| Meta-Level | 0.1968 | 8.4452 | 0.8057 | 0.1678 |

**Discussion.** 
```
All seven tuned hybrids outperform Random and Pop on every accuracy metric. 

Feature Combination, Weighted and Mixed also exceed EASE on all seven accuracy metrics. 

Feature Combination leads on six and simultaneously improves EASE's coverage (0.2848 versus 0.2325), novelty (8.5807 versus 8.5177), and serendipity (0.1867 versus 0.1791), with a small ILD decrease (0.8039 versus 0.8099). 

Weighted shows the same direction and has the highest MRR (0.5363). Mixed improves accuracy and serendipity but reduces novelty. 

Cascade retrieves more relevant items than EASE but ranks them less favourably by NDCG, MRR, and MAP. 

Switching, Augmentation, and Meta-Level fall below EASE on every accuracy metric; 
Augmentation and Meta-Level also narrow catalogue coverage. 

JC: hybrid approach may not automatically lead to improvement. Weighted and Feature Combination methods perform best probably as they preserve the complementary signals from multiple models, whereas structures such as Switching and Augmentation may lose information from the stronger model.
```

**Limitations.** Results describe one saved local split without significance testing or repeated training. Coefficient analysis remains a separate experiment.

**Reproduce:** `uv run python -m scripts.compare_hybrids` (or `.venv/bin/python -m scripts.compare_hybrids` in the existing environment).

**Source outputs:** `results/task2/experiment2/summary.csv` (all 18 models and 11 metrics), `hybrid_vs_individual.csv` (all 63 hybrid/individual pairs across 11 metrics), `comparison.md`, and `protocol.json` (settings, candidate diagnostics, limitations and input hashes).

## Experiment 3 - Quantify improvement over naïve baselines

**Setup.** Compare every tuned individual and hybrid with the saved Random and Pop exports using the 11 metrics from Experiment 2. For each model, baseline, and metric, calculate the absolute difference $\Delta=m-b$ and relative change $100(m-b)/b$. All included metrics are **higher-is-better**, so a **negative value means the model scores below that baseline**.

**Results.** NDCG@10 improvements are shown against both baselines. Absolute differences remain interpretable when a baseline is near zero; percentages against Random are mechanically very large.

| Model | Type | Δ vs Random | % vs Random | Δ vs Pop | % vs Pop |
| --- | --- | ---: | ---: | ---: | ---: |
| ItemKNN | Individual | 0.2718 | 4157.4% | 0.1779 | 177.2% |
| UserKNN | Individual | 0.2812 | 4300.8% | 0.1873 | 186.5% |
| BPR | Individual | 0.3132 | 4789.9% | 0.2193 | 218.4% |
| NeuMF | Individual | 0.3043 | 4654.3% | 0.2104 | 209.5% |
| FISM | Individual | 0.2808 | 4294.6% | 0.1869 | 186.1% |
| LightGCN | Individual | 0.3080 | 4711.4% | 0.2142 | 213.3% |
| NGCF | Individual | 0.3118 | 4768.4% | 0.2179 | 217.0% |
| EASE | Individual | 0.3244 | 4961.2% | 0.2305 | 229.5% |
| SLIMElastic | Individual | 0.3175 | 4855.5% | 0.2236 | 222.6% |
| Weighted | Hybrid | 0.3310 | 5062.0% | 0.2371 | 236.1% |
| Mixed | Hybrid | 0.3294 | 5038.7% | 0.2355 | 234.6% |
| Cascade | Hybrid | 0.3217 | 4920.2% | 0.2278 | 226.9% |
| Switching | Hybrid | 0.3144 | 4808.7% | 0.2205 | 219.6% |
| Feature Combination | Hybrid | **0.3334** | **5099.5%** | **0.2395** | **238.5%** |
| Feature Augmentation | Hybrid | 0.2961 | 4529.1% | 0.2022 | 201.4% |
| Meta-Level | Hybrid | 0.2987 | 4569.1% | 0.2048 | 204.0% |

The complete output contains the same absolute and relative comparisons for Recall, Precision, MRR, Hit, MAP, F1, coverage, novelty, ILD, and serendipity. Against Pop, selected trade-offs are:

| Model | Coverage % | Novelty % | ILD % | Serendipity % |
| --- | ---: | ---: | ---: | ---: |
| EASE | +551.7% | +6.0% | -0.6% | +159.4% |
| Weighted | +590.0% | +6.1% | -1.0% | +166.0% |
| Mixed | +563.3% | +5.4% | -1.0% | +164.7% |
| Feature Combination | **+698.3%** | **+6.8%** | -1.4% | **+170.4%** |

**Discussion.** 
```
Every tuned model improves all seven accuracy metrics over both baselines. 

Pop is the more informative accuracy reference: Feature Combination improves its NDCG by 0.2395, or 238.5%, followed by Weighted at 236.1% and Mixed at 234.6%. 

Their gains over Random exceed 5,000% as Random NDCG is only 0.0065; this scale should not be confused with statistical evidence. 

Beyond accuracy exposes different behaviour. 

Feature Combination improves Pop's coverage by 698.3%, novelty by 6.8%, and serendipity by 170.4%, but its ILD is 1.4% lower. 

Thus, the strongest hybrid is broader, more novel, and more likely to produce relevant unexpected items than Pop, while Pop's single repeated ranking happens to contain a slightly more genre-diverse set of ten movies.

JC: All tuned models outperform both Random and Pop on accuracy; 
however, the extremely large percentage gains over Random are merely a mathematical artifact and should not be treated as statistical evidence. 
In terms of beyond-accuracy, the strongest hybrid is broader, more novel, and more serendipitous than Pop, but its ILD is slightly lower, indicating a mild trade-off between accuracy and diversity, and evaluation must be conducted metric by metric, with an appropriate reference, and by examining both absolute differences and relative changes.
```

**Limitations.** These are descriptive differences on one split, without confidence intervals or significance tests. 

**Reproduce:** `uv run python -m scripts.quantify_baselines`

**Source outputs:** `results/task2/experiment3/baseline_comparison.csv` (352 model-baseline-metric comparisons), `ndcg_summary.csv`, `comparison.md`, and `protocol.json`.

## Experiment 4 - Compare beyond-accuracy performance

**Setup.** Rank all 18 fixed exports separately on catalogue Coverage@10, self-information Novelty@10, genre-Jaccard ILD@10, and relevance-weighted Serendipity@10. 
Higher is better for each metric. Minimum ranks are shared by ties. 
Report category leaders for baselines, individual models, and hybrids, and retain a Pareto frontier containing models that no other model exceeds on all four dimensions. 

**Category leaders.** Global rank is shown in parentheses.

| Metric | Baseline leader | Individual leader | Hybrid leader |
| --- | --- | --- | --- |
| Coverage@10 | Random / Pop: 0.0357 (17) | **FISM: 0.3722 (1)** | Feature Combination: 0.2848 (6) |
| Novelty@10 | **Random: 12.6636 (1)** | FISM: 8.7661 (2) | Feature Combination: 8.5807 (7) |
| ILD@10 | **Pop: 0.8151 (1)** | ItemKNN: 0.8134 (2) | Cascade: 0.8084 (6) |
| Serendipity@10 | Pop: 0.0691 (17) | EASE: 0.1791 (5) | **Feature Combination: 0.1867 (1)** |

**Per-metric ranks.** Frontier means that no other model is at least as high on all four metrics and strictly higher on one, mainly use to measure the trade-off between accuracy and beyond-accuracy.

| Model | Type | Coverage rank | Novelty rank | ILD rank | Serendipity rank | Frontier |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Random | Baseline | 17 | **1** | 18 | 18 | Yes |
| Pop | Baseline | 17 | 18 | **1** | 17 | Yes |
| ItemKNN | Individual | 16 | 16 | 2 | 16 | Yes |
| UserKNN | Individual | 15 | 15 | 4 | 14 | No |
| BPR | Individual | 5 | 5 | 14 | 9 | Yes |
| NeuMF | Individual | 3 | 4 | 16 | 10 | Yes |
| FISM | Individual | **1** | 2 | 12 | 15 | Yes |
| LightGCN | Individual | 2 | 3 | 17 | 11 | Yes |
| NGCF | Individual | 4 | 6 | 15 | 8 | Yes |
| EASE | Individual | 12 | 12 | 5 | 5 | Yes |
| SLIMElastic | Individual | 11 | 10 | 3 | 6 | Yes |
| Weighted | Hybrid | 9 | 11 | 7 | 2 | Yes |
| Mixed | Hybrid | 10 | 13 | 10 | 3 | No |
| Cascade | Hybrid | 7 | 9 | 6 | 4 | Yes |
| Switching | Hybrid | 8 | 8 | 8 | 7 | Yes |
| Feature Combination | Hybrid | 6 | 7 | 13 | **1** | Yes |
| Feature Augmentation | Hybrid | 13 | 17 | 9 | 13 | No |
| Meta-Level | Hybrid | 14 | 14 | 11 | 12 | No |

**Discussion.** 
```
No model leads every beyond-accuracy dimension. 

FISM covers the most catalogue items and is the most novel non-random model, but ranks only 15th on serendipity as many unusual recommendations are not relevant test hits. 

Feature Combination leads serendipity and all hybrids in coverage and novelty, suggesting that combining member scores with side features broadens the relevant recommendations. 

Cascade has the highest hybrid ILD, while ItemKNN and Pop lead their categories: genre variety within a list can remain high even when catalogue coverage is narrow. 

Random's novelty rank is not evidence of quality as it ranks last on serendipity, and its four repeated score rows make the baseline especially limited. 

14 models remain on the four-dimensional Pareto frontier, indicating clear trade-offs among different beyond-accuracy metrics, and no single model dominates all others across all four dimensions.

- High novelty does not mean the recommendations are relevant.
- High ILD does not mean high coverage.
- High coverage does not mean high serendipity.
- Serendipity considers both relevance and unexpectedness, and therefore comes closer to ‘useful novelty’.
```

**Limitations.** Rankings describe one saved split and do not express uncertainty. 

**Reproduce:** `uv run python -m scripts.compare_beyond_accuracy`

**Source outputs:** `results/task2/experiment4/summary.csv`, `metric_rankings.csv`, `category_leaders.csv`, `pareto_frontier.csv`, `comparison.md`, and `protocol.json`.

## Experiment 5 - Examine accuracy–beyond-accuracy trade-offs

**Setup.** Use NDCG@10 as the primary accuracy measure and compare it separately with Coverage@10, Novelty@10, ILD@10, and Serendipity@10. 

Calculate Pearson linear correlation and Spearman rank correlation across 
(1) all 18 exports 
and (2) the 16 non-baseline models. 

For each accuracy/beyond-accuracy pair, identify the two-dimensional Pareto frontier: a model remains when no other model is at least as high on both metrics and strictly higher on one. 
Correlations are descriptive across configurations, without significance or causal claims.

![Accuracy and beyond-accuracy trade-offs](../results/task2/experiment5/tradeoffs.png)

| Scope | Beyond metric | Pearson | Spearman |
| --- | --- | ---: | ---: |
| All 18 models | Coverage@10 | 0.756 | 0.401 |
| All 18 models | Novelty@10 | -0.710 | 0.061 |
| All 18 models | ILD@10 | 0.713 | 0.015 |
| All 18 models | Serendipity@10 | 0.997 | 0.994 |
| 16 non-baselines | Coverage@10 | 0.174 | 0.147 |
| 16 non-baselines | Novelty@10 | 0.088 | 0.112 |
| 16 non-baselines | ILD@10 | -0.105 | -0.003 |
| 16 non-baselines | Serendipity@10 | 0.988 | 0.991 |

| Pair with NDCG@10 | Non-baseline Pareto frontier |
| --- | --- |
| Coverage@10 | Feature Combination, BPR, NGCF, LightGCN, FISM |
| Novelty@10 | Feature Combination, BPR, LightGCN, FISM |
| ILD@10 | Feature Combination, Weighted, EASE, SLIMElastic, ItemKNN |
| Serendipity@10 | Feature Combination |

**Discussion.** 
```
Across all 18 models, NDCG seems strongly positively related to coverage and ILD, and negatively related to novelty; 
However, after removing Random and Pop, all three associations become weak (absolute Pearson at most 0.174 and absolute Spearman at most 0.147). 

Random's extreme novelty and Pop's high ILD therefore drive much of the all-model pattern. 

The Pareto frontiers reveal actual choices: moving from Feature Combination toward FISM gains coverage or novelty while sacrificing NDCG, whereas moving toward ItemKNN gains ILD. 

JC: The trade-off between accuracy and beyond-accuracy is existed, but it is between local models instead global.

Feature Combination improves EASE by 0.0090 NDCG, 0.0523 coverage, 0.0630 novelty, and 0.0076 serendipity, while losing 0.0060 ILD. 
JC: It alone leads both NDCG and serendipity, but this is **partly structural** as the serendipity formula multiplies unexpectedness by test hits,and feature combination has a higher NDCG (higher hits), thereforem serendipity is structurally higher. 


Thus, the local results show 
- no general accuracy trade-off with coverage or novelty, 
- a small accuracy–ILD tension, 
- and a definition-dependent positive relationship with serendipity.
```
**Limitations.** No confidence intervals or repeated-split tests are available, and pairwise frontiers ignore the other three beyond-accuracy dimensions. Serendipity is not statistically independent of accuracy under the implemented definition.

**Reproduce:** `uv run python -m scripts.analyze_tradeoffs`

**Source outputs:** `results/task2/experiment5/correlations.csv`, `pairwise_frontiers.csv`, `reference_deltas.csv`, `tradeoffs.png`, `comparison.md`, and `protocol.json`.
