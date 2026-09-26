# Recommendation Systems — G12

DSAIT4335 Final Project. Hybrid Recommenders with Evaluation and Re-Rankers on MovieLens 100K using RecBole.

```
.github/
  workflows/       CI Branch and Commit Checks
.venv/             Local Environment · Ignored
data/
  ml-100k/         MovieLens 100K Atomic Files
docs/              Contributing · Submission Checklist
notebooks/         Exploration Only
parameters/        RecBole Model Configurations
report/            LaTeX Report · Overleaf
results/           Generated Outputs · Ignored
scripts/           Runnable Entry Points per Task
source/
  analysis/        Coefficient and Group Analysis
  hybrids/         Task 1 Hybrid Recommenders
  metrics/         Task 2 Accuracy and Beyond-Accuracy Metrics
  reranking/       Task 3 Re-Rankers
```

## Simulate
```bash
uv sync
uv run python scripts/<script>.py
```
