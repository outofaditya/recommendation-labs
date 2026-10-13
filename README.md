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
  task23/           Hybrid Coefficient, Ablation, and Contribution Analysis
source/
  analysis/        Coefficient and Group Analysis
  hybrids/         Task 1 Hybrid Recommenders
  metrics/         Task 2 Ranking, Rating-Prediction, and Beyond-Accuracy Metrics
  reranking/       Task 3 Re-Rankers
```

## Simulate

Run from the repository root in this order. Every step writes to `results/`; the main experiments share one seeded split, while Task 1 seed inference and the Task 2.2 repeated-seed suite draw a new split per seed. Steps use all cores by default and `--workers` caps them. RecBole uses a GPU when one is present which matters for FISM and NGCF whose tuning takes hours on CPU.

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
| 1.3 – Weighted Hybrid                    | `uv run python scripts/weighted.py`       | Weights cross-validation and test metrics in `results/hybrid/metrics/` and candidate scores in `results/hybrid/scores/` |
| 1.4 – Other Hybrids                      | `uv run python scripts/hybrids.py`        | Scores and test metrics of the six hybrids in `results/hybrid/`                                                         |
| 1.5 – Hybrid Tuning                      | `uv run python scripts/tune_hybrids.py`   | Scores and test metrics of the seven tuned hybrids in `results/hybrid/`                                                 |
| **Task 2 — Evaluation of Effectiveness** |                                           |                                                                                                                         |
| 2.2 – Regenerate RecBole Split           | `uv run python -m scripts.task22.save_split --config parameters/tuned/BPR.yaml --output results/split` | Recreate train/validation/test TSVs with the course RecBole implementation and the configured seed; no model training |
| 2.2 – Core Evaluation (Experiments 1–5) | `uv run python -m scripts.task22.task22_pipeline --ks 10` | Run the complete fixed-export evaluation in one reproducible command; outputs are written to `results/task2/experiment1/` through `experiment5/` |
| 2.2 – K Sweep and Uncertainty             | `uv run python -m scripts.task22.task22_bootstrap --reps 10000` | Evaluate K = 1, 3, 5, 10, 20 and user-bootstrap 95% confidence intervals for all 18 saved exports in `results/task2/repeated/` |
| 2.2 – Optional Repeated-Seed Experiments    | `caffeinate -dimsu uv run python -m scripts.task22.task22_seeds --workers 4` | Optional retraining over seeds 2020–2024 for run-to-run uncertainty; not required for the current reported results |
| 2.3 – Hybrid Contribution Analysis       | `uv run python -m scripts.task23.analyze_contributions` | Analyze Weighted coefficients and regularization, run leave-one-member-out ablations, and interpret Mixed, Switching, Cascade, and Feature Combination; outputs go to `results/task23/` |
| **Task 3 — Societal Aspects**            |                                           |                                                                                           