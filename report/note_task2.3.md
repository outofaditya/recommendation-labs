# Task 2.3 — Hybrid Contribution Analysis

## Objective

Task 2.3 asks which individual recommenders contribute to hybrid predictions and how their contributions explain the observed hybrid results. Task 1.5 selected configurations; this analysis interprets those configurations. We organise the analysis into three questions:

1. **Global contribution:** which members receive weight, and which members have positive marginal value in one overall blend?
2. **Conditional contribution:** does a member's marginal value change with user activity, and does a global blend hide this specialization?
3. **Architecture contribution:** how do Weighted, Mixed, Switching, Cascade, and Feature Combination use or discard member signals?

The primary metric is validation NDCG@10. Weighted coefficient selection, global ablations, group-specific ablations, and the Feature Combination ablations use five user folds (seed 2020). Mixed member removal and the Switching and Cascade diagnostics use the fixed validation split directly. Test labels are not used to fit or select any Task 2.3 variant.

The run selects `C = 0.0001` for Weighted (five-fold NDCG@10 = 0.2641). EASE and NeuMF have the largest fitted coefficients, while removing SLIMElastic, NeuMF, and FISM causes the largest Weighted ablation drops. Removing EASE instead improves the ablated score, demonstrating that a large coefficient can coexist with little or negative marginal value. In the other designs, Mixed is sensitive to weak voters, Switching assigns LightGCN to the sparsest group and SLIMElastic to the other groups, Cascade benefits from using BPR for retrieval and SLIMElastic for reranking, and the Feature Combination tree relies mainly on movie features. The saved tuned-hybrid test NDCG@10 values are 0.3375 for Weighted, 0.3360 for Mixed, 0.3282 for Cascade, 0.3209 for Switching, and 0.3399 for Feature Combination.

Run from the repository root:

```bash
uv run python -m scripts.task23.analyze_contributions
```

The script writes machine-readable results to `results/task23/`. It uses the saved tuned individual score exports and saved hybrid configuration metadata. The Feature Combination feature-block ablation retrains the shallow gradient-boosted model within each validation fold.

The run also writes a score-correlation matrix and heatmap, a fixed-weight ablation for comparison with the refitted ablation, and a four-group user-activity analysis. The activity extension now additionally fits a separate Weighted policy within each activity group using the same five held-out user folds. Multi-seed repetition is left as an optional future run because it is computationally expensive; the same analysis can later be repeated with a separate results root using the existing `--results` argument.

## 1. Global contribution: coefficients, redundancy, and ablation

Each member score is standardized per user outside the training interactions. Logistic regression is fitted on the union of each member's top 100 candidates. Coefficients are therefore comparable as contributions to this standardized linear score, conditional on the other included members. The magnitude share below is `abs(coefficient) / sum(abs(coefficients))`; it is descriptive and is not a probability or an independent causal share.

| Member | Full-fit coefficient | Absolute magnitude share | Five-fold mean coefficient | Fold SD |
|---|---:|---:|---:|---:|
| EASE | 0.1401 | 23.85% | 0.1304 | 0.0021 |
| NeuMF | 0.1397 | 23.79% | 0.1262 | 0.0032 |
| SLIMElastic | 0.1113 | 18.96% | 0.1087 | 0.0025 |
| FISM | 0.0798 | 13.58% | 0.0765 | 0.0032 |
| Pop | 0.0344 | 5.86% | 0.0292 | 0.0043 |
| BPR | 0.0300 | 5.10% | 0.0310 | 0.0016 |
| NGCF | 0.0242 | 4.12% | 0.0248 | 0.0021 |
| ItemKNN | −0.0193 | 3.29% | −0.0108 | 0.0026 |
| UserKNN | −0.0052 | 0.88% | 0.0060 | 0.0022 |
| LightGCN | 0.0033 | 0.56% | 0.0076 | 0.0015 |

EASE and NeuMF have the largest fitted coefficients, followed by SLIMElastic and FISM. The leading four account for about 80% of the absolute coefficient magnitude. Fold variation is small relative to the leading positive coefficients, so their ordering is stable across these validation folds. UserKNN's full-fit coefficient is slightly negative even though its fold-average coefficient is positive; this is a reminder that correlated members can exchange weight when they are fitted together. A small or negative coefficient describes the conditional fit, not the standalone quality of that recommender.

| C | Five-fold validation NDCG@10 |
|---:|---:|
| 1 | 0.2527 |
| 0.1 | 0.2524 |
| 0.01 | 0.2558 |
| 0.001 | 0.2594 |
| **0.0001** | **0.2641** |
| 0.00001 | 0.2631 |
| 0.000001 | 0.2627 |
| 0.0000001 | 0.2627 |

The best value is `C = 0.0001`, matching the tuned Weighted setting. Strong regularization improves validation ranking substantially over weak regularization: NDCG@10 rises from 0.2527 at `C = 1` to 0.2641. Shrinkage reduces unstable competition between correlated recommenders and gives a more balanced blend. The improvement does not continue at the strongest tested values, so the result supports moderate shrinkage rather than unlimited regularization.

### 1.1 Weighted leave-one-model-out ablation

For each row, one member is removed, the remaining coefficients are refitted using the other four user folds, and the held-out fold is scored. The candidate union is rebuilt from the remaining members. Positive `full − ablated` means the full model performs better.

| Removed member | Ablated CV NDCG@10 | Full − ablated |
|---|---:|---:|
| SLIMElastic | 0.2620 | +0.0021 |
| NeuMF | 0.2628 | +0.0013 |
| FISM | 0.2630 | +0.0011 |
| BPR | 0.2634 | +0.0007 |
| ItemKNN | 0.2636 | +0.0005 |
| NGCF | 0.2638 | +0.0003 |
| LightGCN | 0.2639 | +0.0002 |
| Pop | 0.2641 | 0.0000 |
| UserKNN | 0.2651 | −0.0010 |
| EASE | 0.2656 | −0.0015 |

The ablation changes the interpretation of the coefficient table. Removing SLIMElastic, NeuMF, or FISM hurts validation NDCG most, which supports a useful marginal contribution from these members. Removing EASE improves it by 0.0015, despite EASE having the largest fitted coefficient. EASE may overlap with other strong item-based signals, so its positive fitted weight does not imply a positive marginal gain in this particular cross-validated mixture. Removing UserKNN also gives a small improvement. Pop, LightGCN, and NGCF have near-zero effects at this precision. This is why coefficient size and leave-one-out change should be read together.

### Fixed-weight ablation

The refitted ablation above allows the remaining members to compensate after one member is removed. As a control, a second ablation keeps each fold's full-model coefficients and sets only the removed coefficient to zero, while preserving the full candidate pool.

| Removed member | Fixed-weight CV NDCG@10 | Full − ablated |
|---|---:|---:|
| SLIMElastic | 0.2608 | +0.0033 |
| NeuMF | 0.2625 | +0.0016 |
| FISM | 0.2632 | +0.0009 |
| ItemKNN | 0.2635 | +0.0006 |
| NGCF | 0.2635 | +0.0006 |
| BPR | 0.2637 | +0.0004 |
| Pop | 0.2643 | −0.0002 |
| LightGCN | 0.2642 | −0.0001 |
| UserKNN | 0.2654 | −0.0013 |
| EASE | 0.2665 | −0.0024 |

The two ablations agree on the main result: SLIMElastic, NeuMF and FISM are the most useful contributors, while EASE and UserKNN do not provide positive marginal value in this particular blend. The fixed-weight drops for SLIMElastic and NeuMF are larger than their refitted counterparts because the remaining models cannot compensate. This confirms that the result is not caused only by re-estimating the other coefficients.

## 2. Conditional contribution: user activity and marginal value

The member score correlation was computed over 214,583 rows in the union of the members' top-100 candidate pools. The strongest correlations are between NGCF and LightGCN (0.958), EASE and SLIMElastic (0.912), and the graph/embedding members more generally. Pop has weak or negative correlation with most personalized scores. These patterns support the coefficient and ablation results: correlated models can exchange coefficient mass, whereas a model with a different score pattern can add information even with a smaller coefficient.

The correlation matrix is descriptive and does not establish that correlation causes a small marginal contribution. It is supporting evidence alongside the coefficient and ablation analyses.

Weighted performance was also evaluated by four equal-sized user groups ordered by the number of training interactions:

| Train interactions | Weighted NDCG@10 | Best individual | Best individual NDCG@10 | Weighted − best |
|---:|---:|---|---:|---:|
| 16–27 | 0.2295 | LightGCN | 0.2428 | −0.0133 |
| 27–53 | 0.2218 | SLIMElastic | 0.2195 | +0.0023 |
| 53–120 | 0.2449 | SLIMElastic | 0.2346 | +0.0103 |
| 120–591 | 0.3586 | SLIMElastic | 0.3520 | +0.0066 |

The hybrid gives its largest improvement over the best individual model for medium and highly active users. For the sparsest users, LightGCN is stronger than the global Weighted blend, so one global coefficient vector is not optimal for every activity group. This motivates activity-aware hybrids, while the overall Switching result shows that selecting one model per group can still lose useful complementary signals.

### 2.1 Group-specific Weighted coefficients

To test whether the global coefficient vector hides user-state dependence, we refit the Weighted model separately inside each of the four activity groups. For every held-out user fold, only the other users from the same activity group provide fitting rows; the held-out group's validation labels are used only for scoring. We keep `C = 0.0001`, the global choice, and use the same union-of-top-100 candidate construction. No test labels are used.

| Train interactions | Group-specific NDCG@10 | Global Weighted NDCG@10 | Group-specific − global |
|---:|---:|---:|---:|
| 16–27 | 0.2288 | 0.2295 | −0.0007 |
| 27–53 | 0.2235 | 0.2218 | **+0.0017** |
| 53–120 | 0.2441 | 0.2449 | −0.0008 |
| 120–591 | **0.3601** | 0.3586 | **+0.0015** |

The coefficients show the expected specialization, but the performance evidence is mixed. LightGCN receives a larger group-specific coefficient for sparse users (0.0207 versus 0.0076 in the global five-fold fit), while SLIMElastic receives its largest coefficient for active users (0.1105). EASE and NeuMF remain important across groups, and Pop is near zero or negative in the two most active groups. However, group-specific fitting improves only groups 2 and 4; it slightly harms groups 1 and 3. Therefore, the result supports *conditional contribution* but not yet a universal activity-aware production policy. The small differences are also consistent with estimation noise from fitting on only about 236 users per group.

This is a useful negative result: selecting a group-specific blend is not automatically better than one regularized global blend. A production system should keep the global Weighted model as a fallback and require repeated-split evidence before enabling activity-aware routing. The result motivates Task 2.4 experiments with a held-out gate or a candidate-stage policy, rather than directly deploying four independently fitted coefficient vectors.

### 2.2 Per-group marginal contribution

Group-specific coefficients alone do not prove that a model adds value: a large coefficient can represent a correlated signal, and a small coefficient can still correct errors in a particular segment. We therefore repeat the leave-one-model-out experiment separately within each activity group. For each removed member, the remaining coefficients are refitted using only the other users in that group and evaluated on held-out users from the same group. The reported marginal contribution is:

\[
\Delta_{m,g} = NDCG_{\text{full},g} - NDCG_{\text{without }m,g}.
\]

Positive values mean that retaining member \(m\) helps that group; negative values mean that the group-specific model improves when that member is removed.

| Activity group | History range | Largest positive contributions | Negative or redundant signals |
|---|---:|---|---|
| 1 | 16–27 | ItemKNN +0.0023; BPR +0.0013; EASE +0.0009 | SLIMElastic −0.0058; FISM −0.0024; UserKNN −0.0017 |
| 2 | 27–53 | SLIMElastic +0.0072; UserKNN +0.0025; BPR +0.0014 | FISM −0.0016; EASE −0.0003; ItemKNN −0.0003 |
| 3 | 53–120 | SLIMElastic +0.0076; FISM +0.0066 | UserKNN −0.0020; NGCF −0.0008; EASE −0.0005 |
| 4 | 120–591 | NeuMF +0.0044; SLIMElastic +0.0033; BPR +0.0026 | FISM −0.0001; LightGCN −0.0000 |

This is stronger evidence of conditional contribution than the coefficient table alone. The sparse group does not benefit from retaining every item-oriented signal: SLIMElastic is strongly redundant or harmful in the fitted sparse-group blend, while ItemKNN and BPR provide positive marginal corrections. In groups 2 and 3, SLIMElastic is the clearest contributor, and FISM is useful in group 3. In the most active group, NeuMF and SLIMElastic contribute most, while graph and FISM signals add little after the other members are present. The pattern is therefore not simply “graph models for sparse users and item models for active users”; it is a segment-specific complementarity pattern.

The differences are descriptive rather than causal. Some values are small, and refitting after removal lets correlated members compensate. Nevertheless, this analysis distinguishes three cases that should not be conflated: a model can have a large coefficient but little marginal value, a model can have a modest coefficient but add a useful correction, and a model can be useful only for one user segment.

## 3. Architecture contribution: how hybrid structures use the signals

### Mixed

The tuned Mixed model fuses all ten members with reciprocal-rank fusion at `k = 10`. Its validation NDCG@10 is 0.2658. Removing members one at a time while keeping the selected fusion constant gives these largest changes:

| Removed member | Change in NDCG@10 (`full − ablated`) |
|---|---:|
| NeuMF | +0.0012 |
| EASE | −0.0010 |
| Pop | −0.0006 |
| BPR | −0.0005 |
| ItemKNN | −0.0023 |

Removing NeuMF lowers validation NDCG, while removing ItemKNN, BPR, EASE, or Pop raises it slightly. Thus equal participation in rank fusion does not mean equal usefulness. Rank fusion is robust to score-scale differences, but weak or overlapping voters can dilute the strongest ranks. These are small differences from a single validation set; they should be treated as descriptive.

### Switching

The selected member for each activity group matches the best standalone member measured within that same group on validation:

| User train interactions | Selected recommender | Group validation NDCG@10 |
|---:|---|---:|
| 16–27 | LightGCN | 0.2455 |
| 27–53 | SLIMElastic | 0.2192 |
| 53–120 | SLIMElastic | 0.2366 |
| 120–591 | SLIMElastic | 0.3496 |

This supports a limited specialization pattern: LightGCN wins for the sparsest group, while SLIMElastic wins in the other three groups. It does not show that Switching is better overall. On the same saved split, the tuned Switching test NDCG@10 is 0.3209, below the tuned Weighted (0.3375), Mixed (0.3360), and Feature Combination (0.3399) scores. Choosing one model per user group discards the complementary signals available from the other members.

### Cascade

The tuned Cascade uses BPR to propose 20 items and SLIMElastic to order that shortlist. On validation, BPR alone over the full catalogue reaches NDCG@10 = 0.2535; the BPR shortlist followed by SLIMElastic reaches 0.2624; SLIMElastic alone over the full catalogue reaches 0.2589. The BPR shortlist contains 26.7% of validation relevant interactions. In this configuration, BPR's contribution is candidate retrieval, while SLIMElastic adds value by reranking the restricted set. The gain over either single stage is consistent with the stages playing different roles, although the shortlist still excludes most relevant items. The saved Cascade test NDCG@10 is 0.3282.

### Feature Combination

The Feature Combination ablation uses the tuned shallow tree (depth 2, learning rate 0.1) and the same five user folds. Candidate scores are evaluated with the hybrid's candidate pool.

| Feature set | Five-fold validation NDCG@10 | Fold SD |
|---|---:|---:|
| Member scores + user and movie features | **0.2636** | 0.0049 |
| Member scores + movie features | 0.2634 | 0.0054 |
| Member scores + user features | 0.2370 | 0.0101 |
| Member scores only | 0.2322 | 0.0094 |

Movie metadata and popularity features account for nearly all of the gain in this tree-based ablation; user demographics and activity features add little beyond the movie features. The tree using only member scores performs much worse than the full feature set. This comparison is specific to the tuned tree and does not contradict the Weighted result: the linear Weighted model and the tree learn different functions. The saved Feature Combination test NDCG@10 is 0.3399, the highest among the tuned hybrids on this local split.

## Overall interpretation

The strongest evidence of useful marginal contribution comes from SLIMElastic, NeuMF, and FISM in the Weighted leave-one-out analysis. The coefficient fit also assigns large positive weights to EASE and NeuMF, but EASE's removal improves cross-validated NDCG, showing that its signal overlaps with other members in the fitted blend. Strong shrinkage helps manage this overlap. Contributions are conditional on user activity: graph signal receives relatively more weight for sparse users and item-oriented signals remain prominent for active users, but the first group-specific experiment does not consistently outperform the global blend. Other hybrid structures expose different roles: Mixed can be diluted by weak voters, Switching specializes by user activity but loses the benefit of combining signals, Cascade separates retrieval from reranking, and Feature Combination relies heavily on movie-side features under the tested tree configuration.

## Limitations

- Coefficients are conditional on the included members, per-user score standardization, candidate construction, and regularization. They do not establish causality or standalone model quality.
- The tuned Mixed member list and Switching configuration were selected using validation data; their validation contribution results are descriptive and may be optimistic.
- Leave-one-out differences are small and come from one fixed split with five user folds. No significance claim is made.
- The activity-group analysis uses four equal-sized groups and the selected global `C = 0.0001`; the group-specific policy and per-group ablations are not separately tuned, and each group has only about 236 users.
- Per-group ablation differences use refitted coefficients and rebuilt candidate pools, so they measure marginal value under a changed model rather than a pure fixed-weight intervention.
- Test results are reported only as the final outcomes of the already selected Task 1 configurations; no Task 2.3 decisions use test labels.
- The local artifacts and corrected split should be used consistently. These values describe the checked-in local result files, and may differ from numbers in earlier Task 1 notes.
- Multi-seed coefficient and ablation uncertainty is not executed in the current run because of compute limits. To repeat it later, run the same module once per seed-specific results root, for example `uv run python -m scripts.task23.analyze_contributions --results results_seed_2021 --output results/task23_seed_2021`, and aggregate the resulting CSV files.

## Output files

- `results/task23/weighted_coefficients.csv`
- `results/task23/weighted_coefficients_by_C.csv`
- `results/task23/weighted_regularization.csv`
- `results/task23/weighted_leave_one_out.csv`
- `results/task23/weighted_fixed_weight_ablation.csv`
- `results/task23/member_score_correlations.csv`
- `results/task23/member_score_correlations.png`
- `results/task23/weighted_by_activity.csv`
- `results/task23/weighted_by_activity_policy.csv`
- `results/task23/weighted_activity_coefficients.csv`
- `results/task23/weighted_leave_one_out_by_activity.csv`
- `results/task23/switching_by_user_group.csv`
- `results/task23/cascade_stages.csv`
- `results/task23/combination_feature_ablation.csv`
- `results/task23/summary.json`
