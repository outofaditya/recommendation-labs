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
| 1.3 – Weighted Hybrid                    | `uv run python scripts/weighted.py`       | Weights cross-validation and test metrics in `results/hybrid/metrics/` and candidate scores in `results/hybrid/scores/` |
| 1.4 – Other Hybrids                      | `uv run python scripts/hybrids.py`        | Scores and test metrics of the six hybrids in `results/hybrid/`                                                         |
| 1.5 – Hybrid Tuning                      | `uv run python scripts/tune_hybrids.py`   | Scores and test metrics of the seven tuned hybrids in `results/hybrid/`                                                 |
| **Task 2 — Evaluation of Effectiveness** |                                           |                                                                                                                         |
| **Task 3 — Societal Aspects**            |                                           |                                                                                                                         |
| 3.1 – Re-Rankers                         | `uv run python scripts/rerank.py`         | Four re-rankers on three models for every lambda on validation and the chosen one on test in `results/rerank/`          |
| 3.2 – Re-Ranked Evaluation               | `uv run python scripts/tradeoff.py`       | Accuracy and beyond-accuracy per lambda in `results/rerank/curves.csv` and before and after on test in `test.csv`       |
