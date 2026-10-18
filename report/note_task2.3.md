# Task 2.3 — Hybrid Contribution Analysis

## Repeated-run design

The contribution and architecture analyses were repeated for seeds 2020–2024. For each seed, the individual-model scores and hybrid components were rebuilt; learned hybrid parameters were refit using that seed's validation data. Task 2.3 reports validation cross-validation diagnostics for contribution and ablation analyses. Values are means over five runs with 95% Student-t intervals unless stated otherwise. These intervals summarize seed variability in this fixed experimental design; they are not user-bootstrap intervals.

## Global weighted blend: coefficients and ablations

The weighted blend standardizes member scores per user and fits a regularized logistic model using five validation-user folds. Coefficients are conditional on the included, correlated members and candidate construction; absolute coefficient shares are descriptive weights, not causal importance.

| Member | Mean coefficient | 95% CI | Absolute coefficient share |
| --- | ---: | ---: | ---: |
| EASE | 0.1139 | [0.0430, 0.1847] | 22.37% |
| NeuMF | 0.1035 | [0.0365, 0.1704] | 19.54% |
| SLIMElastic | 0.0887 | [0.0354, 0.1419] | 18.19% |
| FISM | 0.0775 | [0.0278, 0.1272] | 15.06% |
| Pop | 0.0342 | [0.0106, 0.0578] | 6.21% |
| BPR | 0.0259 | [0.0108, 0.0410] | 5.70% |
| NGCF | 0.0202 | [0.0081, 0.0324] | 4.47% |
| ItemKNN | -0.0131 | [-0.0253, -0.0009] | 3.61% |
| UserKNN | -0.0020 | [-0.0100, 0.0060] | 3.52% |
| LightGCN | 0.0005 | [-0.0042, 0.0051] | 1.34% |

EASE, NeuMF, SLIMElastic, and FISM together account for about 75% of mean absolute coefficient share. The size and sign of a coefficient alone do not establish marginal benefit: ablations provide a more direct, though still conditional, comparison.

| Removed member | Refitted blend: full − ablated NDCG@10 (95% CI) | Fixed-weight blend: full − ablated NDCG@10 (95% CI) |
| --- | ---: | ---: |
| SLIMElastic | +0.00496 [0.00203, 0.00790] | +0.00402 [0.00126, 0.00678] |
| FISM | +0.00213 [0.00024, 0.00403] | +0.00211 [0.00014, 0.00407] |
| NeuMF | +0.00150 [0.00032, 0.00268] | +0.00181 [0.00071, 0.00291] |
| BPR | +0.00059 [0.00009, 0.00109] | +0.00076 [0.00016, 0.00136] |
| NGCF | +0.00034 [-0.00039, 0.00106] | +0.00044 [-0.00035, 0.00123] |
| LightGCN | +0.00036 [-0.00003, 0.00076] | +0.00032 [-0.00017, 0.00082] |
| EASE | -0.00027 [-0.00223, 0.00169] | -0.00056 [-0.00239, 0.00127] |

The two ablation views agree that SLIMElastic, FISM, and NeuMF have the clearest positive contribution across seeds; BPR's contribution is smaller but positive. EASE has a large coefficient share but no positive marginal effect in these ablations. Refitted ablation removes the member and refits the remaining blend; fixed-weight ablation preserves the fit and candidate pool while zeroing one coefficient. Their results answer different questions.

Cross-validation selected regularization C=0.0001, with mean NDCG@10 0.2692 [0.2613, 0.2772]. The next-best tested value, C=1e-5, averaged 0.2684 [0.2593, 0.2775]; stronger regularization at C=0.001 and above had lower mean scores. The C intervals overlap, so the small advantage at 0.0001 is not a strong separation across five seeds.

## Activity-conditioned blends

Activity groups are based on training interaction counts. A global weighted blend versus each group's best individual model gives:

| Activity group (training interactions) | Weighted NDCG@10 | Best individual NDCG@10 | Weighted − best (95% CI) |
| --- | ---: | ---: | ---: |
| 1 (16–27) | 0.2370 | 0.2419 | -0.00490 [-0.01129, 0.00150] |
| 2 (27–53) | 0.2391 | 0.2390 | +0.00014 [-0.00563, 0.00592] |
| 3 (53–120) | 0.2442 | 0.2380 | +0.00616 [0.00078, 0.01153] |
| 4 (120–591) | 0.3570 | 0.3502 | +0.00678 [0.00482, 0.00874] |

The global blend is most helpful in groups 3 and 4; the intervals for groups 1 and 2 include zero. Refitting separate group-specific weights instead of using the global blend changed NDCG by -0.00181, +0.00091, +0.00003, and +0.00044 across groups 1–4; every corresponding interval includes zero. These runs do not show a reliable benefit from a more complex activity-specific weighting policy.

## Hybrid architecture diagnostics

The aggregate NDCG curves provide context for how the fixed hybrid designs compare across recommendation-list lengths; the contribution tables above focus on validation diagnostics at K=10.

![Mean NDCG by recommendation-list length across the five runs](../results/task2/multiseed/aggregate/plots/ndcg_by_k.png)

**Mixed fusion.** The full mixed model averaged validation NDCG@10 0.2675. Removing FISM reduced it by 0.00126 [0.00052, 0.00200]. The estimated drops for SLIMElastic (+0.00134) and NeuMF (+0.00132) were less certain because their intervals include zero. Most other member removals had small, uncertain effects. Equal-rank fusion therefore does not imply equal contribution.

**Cascade.** On validation, the first-stage full ranking averaged NDCG@10 0.2550 [0.2479, 0.2621]. Shortlisting 20 candidates and reranking them raised it to 0.2650 [0.2577, 0.2723], compared with 0.2638 [0.2561, 0.2716] for the second-stage model's full ranking. The mean supports complementary retrieval and reranking roles, while the overlapping intervals call for caution.

**Feature Combination.** The fixed shallow-tree feature diagnostic averaged NDCG@10 0.2686 with member scores plus user and movie features, versus 0.2683 with member scores plus movie features, 0.2366 with scores plus user features, and 0.2358 with member scores only. Movie-side features explain most of the gain in this diagnostic. This is a fixed diagnostic configuration; it should not be read as an explanation of every separately tuned combination model.

## Interpretation and limits

Repeated runs strengthen the evidence for a useful SLIMElastic, FISM, and NeuMF contribution to the weighted blend, while the activity analysis suggests gains are concentrated among more active users. The coefficient results, validation ablations, and architecture diagnostics remain conditional on the selected candidates, fixed hyperparameters, and validation procedures. Five seeds are a small sample, and these summaries do not include a test-set significance analysis. See the Task 2.3 summaries in [`results/task2/multiseed/aggregate/task23/`](../results/task2/multiseed/aggregate/task23/) for per-seed values and complete tables.
