# Recommendation Systems — G12

DSAIT4335 Final Project. Hybrid Recommenders with Evaluation and Re-Rankers on MovieLens 100K using RecBole.

```
.github/
  workflows/       CI Conventions · Lint · BPR Export
data/
  ml-100k/         MovieLens 100K Atomic Files
docs/              Contributing · Submission Checklist
parameters/        RecBole Model Configurations
  search/          Tuning Search Spaces
  tuned/           Tuned Configurations
report/            Working Notes and LaTeX Report
results/           Generated Outputs · Ignored
scripts/           Runnable Entry Points per Task
source/
  analysis/        Coefficient and Group Analysis
  hybrids/         Task 1 Hybrid Recommenders
  metrics/         Task 2 Accuracy and Beyond-Accuracy Metrics
  reranking/       Task 3 Re-Rankers
```

## Simulate

Run from the repository root in this order. Every step writes to `results/` and shares one seeded split except seed inference which draws a new split per seed. Steps use all cores by default and `--workers` caps them. RecBole uses a GPU when one is present which matters for FISM and NGCF whose tuning takes hours on CPU.

```bash
uv sync
```

| Task                                     | Command                                   | Outcome                                                                                                                 |
| ---------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Task 1 — Hybrid Recommender**          |                                           |                                                                                                                         |
| 1.1 – Individual Models                  | `uv run python scripts/models.py`         | Score tables in `results/scores/` and test metrics in `results/metrics/` for 11 models in a few minutes                 |
| 1.2 – Individual Tuning                  | `uv run python scripts/tune.py`           | Every trial in `results/tuning/` and the winners in `parameters/tuned/` with `--workers 1` for an exact rerun           |
| 1.2 – Run Tuned Models                   | `uv run python scripts/models.py --tuned` | Score tables and test metrics of the tuned models in `results/tuned/`                                                   |
| 1.2 – Seed Inference                     | `uv run python scripts/seeds.py`          | Test metrics per seed from 2020 to 2024 in `results/seeds/` with mean and std in `summary.csv`                          |
| 1.1 and 1.2 – Content-Based             | `uv run python scripts/content.py`        | Base and tuned score tables and test metrics in `results/` and `results/tuned/` with every grid point in `results/tuning/` |
| 1.3 – Weighted Hybrid                    | `uv run python scripts/weighted.py`       | Weights cross-validation and test metrics in `results/hybrid/metrics/` and candidate scores in `results/hybrid/scores/` |
| 1.4 – Other Hybrids                      | `uv run python scripts/hybrids.py`        | Scores and test metrics of the six hybrids in `results/hybrid/`                                                         |
| 1.5 – Hybrid Tuning                      | `uv run python scripts/tune_hybrids.py`   | Scores and test metrics of the seven tuned hybrids in `results/hybrid/`                                                 |
| **Task 2 — Evaluation of Effectiveness** |                                           |                                                                                                                         |
| 2.2 – Regenerate RecBole Split           | `uv run python -m scripts.task22.save_split --config parameters/tuned/BPR.yaml --output results/split` | Recreate train/validation/test TSVs with the course RecBole implementation and the configured seed; no model training |
| 2.2 – Core Evaluation (Experiments 1–5) | `uv run python -m scripts.task22.task22_pipeline --ks 10` | Run the complete fixed-export evaluation in one reproducible command; outputs are written to `results/task2/experiment1/` through `experiment5/` |
| 2.2 – K Sweep and Uncertainty             | `uv run python -m scripts.task22.task22_bootstrap --reps 10000` | Evaluate K = 1, 3, 5, 10, 20 and paired user-bootstrap 95% confidence intervals for additive metrics; point-only coverage for all 18 saved exports in `results/task2/repeated/` |
| 2.2–2.4 – Prepare Repeated Seeds | `uv run python -m scripts.task2.multiseed --phase prepare --output results/task2/multiseed_v2` | Freeze seeds 2020–2024, source/data hashes and the seven selected `results/hybrid/metrics/Tuned*.json` settings from Task 1; optionally add `--training-config PATH --require-gpu` |
| 2.2–2.4 – Train Repeated Seeds | `for seed in 2020 2021 2022 2023 2024; do uv run python -m scripts.task2.multiseed --phase train --seed "$seed" --output results/task2/multiseed_v2; done` | Train 11 individual/baseline exports per seed using the frozen settings and save the corresponding splits and embeddings |
| 2.2–2.4 – Analyze Repeated Seeds | `for seed in 2020 2021 2022 2023 2024; do uv run python -m scripts.task2.multiseed --phase analyze --seed "$seed" --output results/task2/multiseed_v2; done` | Refit seven hybrids per seed and run Tasks 2.2–2.4 at K=1,3,5,10,20 |
| 2.2–2.4 – Aggregate Repeated Seeds | `uv run python -m scripts.task2.multiseed --phase aggregate --output results/task2/multiseed_v2` | Aggregate seed-level means, SDs and 95% t intervals in `results/task2/multiseed_v2/aggregate/` |
| 2.2–2.4 – Check Repeated Seeds | `uv run python -m scripts.task2.multiseed --phase status --output results/task2/multiseed_v2` | Validate all seed stages and aggregate checksums without running experiments |
| 2.2–2.4 – CPU Analysis of Existing Exports | `uv run python -m scripts.task2.analyze --source-root results/delftblue_import/verified_w4lfecaj/results/task2/multiseed --output results/task2/analysis_cpu_v2 --threads 2 --bootstrap-reps 10000` | Reuse verified five-seed exports, rerun evaluation and contribution diagnostics on CPU, run the paired seed-2020 bootstrap and aggregate automatically; replace the source path with your extracted archive and use a fresh output directory |
| 2.2–2.4 – Check Completed CPU Analysis | `uv run python -m scripts.task2.analyze --output results/task2/reanalysis_cpu --status` | Validate the completed review run and its preserved source snapshot without retraining or recomputing metrics |
| Task 2 – Validate and Refresh Notes | `uv run python -m scripts.task2.report --results results/task2/reanalysis_cpu` | Validate completion hashes and seed statistics, then update Task 2 in `report/notes.md` and its figures |
| 2.3 – Hybrid Contribution Analysis       | `uv run python -m scripts.task23.analyze_contributions` | Analyze TunedWeighted coefficients and regularization, run leave-one-member-out ablations, and interpret Mixed, Switching, Cascade, and Feature Combination; outputs go to `results/task23/` |
| 2.4 – Hybrid Behaviour and Long Tail | `uv run python -m scripts.task24.task24_pipeline` | User-level routing, candidate-depth comparisons, popularity exposure, specialized tail retrieval, and tail-interest segments in `results/task24/` |
| Task 2 – Code Regression Tests | `uv run python -m unittest discover -s scripts/task22/tests -t .` | Run synthetic fixtures and mocked training for evaluation, analysis and reporting; no experiments |
| **Task 3 — Societal Aspects**            |                                           |                                                                                                                         |
| 3.1 – Re-Rankers                         | `uv run python scripts/rerank.py`         | Four re-rankers on three models for every lambda on validation and the chosen one on test in `results/rerank/`          |
| 3.2 – Re-Ranked Evaluation               | `uv run python scripts/tradeoff.py`       | Accuracy and beyond-accuracy per lambda in `results/rerank/curves.csv`, before and after on test in `test.csv` and what each top 100 holds in `evidence.csv` |
| 3.3 – Re-Ranked Hybrid                   | `uv run python scripts/rerank_hybrid.py`  | Tuned Mixed with its members re-ranked before fusion or its list re-ranked after, curves, test and evidence in `results/rerank/hybrid/` |
| 3.4 – Re-Ranker Impact per Group         | `uv run python scripts/rerank_groups.py`  | Test change per user group and movie group with 95% bootstrap intervals for every chosen re-ranking in `results/rerank/groups.csv` |
| 3.2 to 3.4 – Figures                     | `uv run python scripts/rerank_figures.py` | Trade-off, hybrid order and group impact figures as PNG in `report/figures/` |

