# Contributing

## Setup
```bash
uv sync
```

## Scopes
Every branch and commit names one scope.

| Scope | Meaning |
| --- | --- |
| `t1.1` to `t1.5` | Task 1 Hybrid Recommender |
| `t2.1` to `t2.6` | Task 2 Evaluation Of Effectiveness |
| `t3.1` to `t3.4` | Task 3 Societal Aspects |
| `report` | LaTeX report |
| `setup` | Tooling, CI, dependencies |
| `docs` | Documentation |

## Branches
`<name>/<scope>-<topic>` in lower case.

- `aditya/t1.3-weighted`
- `wout/t2.1-novelty`
- `jiayue/report-intro`

Your name goes first so several people can work on one scope. One branch per piece of work. Never commit to `main`.

## Commits
[Conventional Commits](https://www.conventionalcommits.org) with a required scope.

`<type>(<scope>): <summary in lower case>`

- `feat(t1.3): learn hybrid weights with linear regression`
- `fix(t2.1): exclude train items from novelty`
- `docs(report): add task 2 discussion`

Types: `feat` `fix` `docs` `refactor` `test` `chore`.

## Pull Requests
- Open a PR into `main`. CI checks the branch name and every commit message.
- One teammate approves before merging.
- Merge with a merge commit. Squash and rebase are disabled so every author keeps their commits on `main`.
- Update your branch with `git merge main`, never rebase a pushed branch.

```bash
gh pr create --base main
```
