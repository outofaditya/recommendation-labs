# Recommendation Systems — G12

DSAIT4335 Final Project. Hybrid Recommenders with Evaluation and Re-Rankers on MovieLens 100K using RecBole.

```
.github/
  workflows/       CI Conventions · Lint · BPR Export
.venv/             Local Environment · Ignored
data/
  ml-100k/         MovieLens 100K Atomic Files
docs/              Contributing · Submission Checklist
notebooks/         Exploration Only
parameters/        RecBole Model Configurations
  search/          Tuning Search Spaces
  tuned/           Tuned Configurations
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

Run from the repository root in this order. Every step writes to `results/` and uses the same seeded split.

```bash
uv sync
```

| Step | Command | Outcome |
| --- | --- | --- |
| 1.1 Individual Models | `uv run python scripts/models.py` | Full score tables for 11 models in `results/scores/` and test metrics in `results/metrics/` within a few minutes |
| 1.2 Individual Tuning | `uv run python scripts/tune.py` | Validation scores of every trial in `results/tuning/` and the winning configs in `parameters/tuned/` |
| 1.2 Tuned Models | `uv run python scripts/models.py --tuned` | Score tables and test metrics of the tuned models replacing those of step 1.1 |
