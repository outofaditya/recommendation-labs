# Contributing

## Hooks
`uv run pre-commit install` once per clone. Ruff formats and lints on every commit. When it changes files stage them and commit again.

## Scopes
| Scope | Covers |
| --- | --- |
| `t1.1` – `t1.5` | Hybrid Recommender |
| `t2.1` – `t2.6` | Evaluation of Effectiveness |
| `t3.1` – `t3.4` | Societal Aspects |
| `report` `setup` `docs` | Report · Tooling · Documentation |

## Branches
`<name>/<scope>-<topic>` eg. `aditya/t1.3-weighted`. Name first so several people can share a scope.

## Commits
Conventional Commits with a required scope eg. `feat(t1.3): learn hybrid weights with regression`.

Types: `feat` `fix` `docs` `refactor` `test` `chore`.

## Pull Requests
- CI checks the branch name · commit messages · lock file · lint · a BPR score export.
- One approval. Merge commits only so every author keeps their history on `main`.
- Sync with `git merge main`. Never rebase a pushed branch.
