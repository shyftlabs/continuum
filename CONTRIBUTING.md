# Contributing to Continuum

Thanks for your interest in contributing! This document covers the workflow, conventions, and IP terms for landing changes in Continuum.

## Code of Conduct

This project adheres to the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By participating, you agree to uphold it. Report unacceptable behavior to **continuum@shyftlabs.io**.

## Licensing

Continuum is licensed under the [Apache License 2.0](LICENSE). By submitting a contribution, you agree that your contribution is licensed under the same Apache 2.0 terms (Section 5 of the license).

### Developer Certificate of Origin (DCO)

All commits must be signed off under the [Developer Certificate of Origin](https://developercertificate.org). This is a lightweight alternative to a CLA — by signing off, you certify that you wrote the code or otherwise have the right to submit it under Apache 2.0.

Add `Signed-off-by: Your Name <your@email>` to every commit. Git can do this automatically:

```bash
git commit -s -m "feat: add structured output schema"
```

PRs without signed-off commits will be blocked by the DCO check.

## Branch model

```
main          ← stable, tagged releases (what users `pip install`)
└── dev       ← integration branch — all PRs land here first
    ├── feat/<short-slug>       ← new features
    ├── fix/<short-slug>        ← bug fixes
    └── chore/<short-slug>      ← refactors, deps, docs
```

- **Always branch from `dev`**, never from `main`.
- **PRs target `dev`**, never `main`. `dev` → `main` happens via a release PR cut by a maintainer.
- **Squash-merge into `dev`** (one PR = one commit).
- **Long-lived branches are deleted after merge.**

### Branch protection (maintainers)

Both `main` and `dev` are protected. Configure under **Settings → Branches → Branch protection rules** (or `gh api`):

| Setting | `main` | `dev` |
|---|---|---|
| Require a pull request before merging | ✅ | ✅ |
| Required approving reviews | 1 (from a maintainer) | 1 |
| Dismiss stale approvals on new commits | ✅ | ✅ |
| Require status checks to pass | ✅ | ✅ |
| Required checks | `Lint & type-check`, `Tests (3.13)`, `Secret scan (gitleaks)` | same |
| Require branches up to date before merging | ✅ | ✅ |
| Require signed-off commits (DCO) | ✅ | ✅ |
| Require linear history | ✅ | ✅ |
| Restrict who can push | maintainers only | maintainers only |
| Allow force pushes / deletions | ❌ | ❌ |

`pip-audit` and `bandit` run as **advisory** checks (non-blocking) — review their output, but they will not gate a merge. Direct pushes to `main`/`dev` are disabled; all changes land through PRs.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org):

| Prefix | When to use |
|---|---|
| `feat:` | New user-facing feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `chore:` | Refactor, deps, tooling |
| `test:` | Test-only changes |
| `perf:` | Performance improvement |
| `ci:` | CI / build config |

Conventional Commit history powers automated changelogs and version bumps via [`release-please`](https://github.com/googleapis/release-please).

## Pull request checklist

Before opening a PR:

1. **Tests pass locally** — `pytest`
2. **Lint passes** — `ruff check .`
3. **Type-check passes** — `mypy src/orchestrator`
4. **Docs updated** if user-facing
5. **`CHANGELOG.md` updated** under `## [Unreleased]` if user-facing
6. **Every commit is signed off** (`git commit -s`)
7. **PR title follows Conventional Commits**

Fill in the PR template — *what / why / how tested*.

## Review process

- First maintainer response: within **3 working days**.
- Approval decision: within **7 working days**.
- 1 approving review from a maintainer + green CI required to merge.

Maintainers may request changes, suggest alternatives, or decline a PR. Decisions come with a reason — see [`MAINTAINERS.md`](MAINTAINERS.md) for tone and escalation.

## Local setup

```bash
git clone https://github.com/shyftlabs/continuum.git
cd continuum

python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pre-commit install                          # ruff lint+format on commit
pre-commit install --hook-type pre-push     # mypy on push

cp .env.template .env       # add your provider keys
docker compose up -d        # Redis · Milvus · Langfuse

pytest                      # run the test suite
```

### Code quality

`ruff` (lint + format) and `mypy` run automatically via [pre-commit](https://pre-commit.com) once installed — `ruff` on every commit, `mypy` on push. They read their config from `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`). The same checks run in CI on every PR. Run them manually anytime:

```bash
ruff check .            # lint
ruff format .           # format
mypy src/orchestrator   # type-check
pre-commit run --all-files   # everything, across the whole repo
```

## Reporting bugs

Use [`.github/ISSUE_TEMPLATE/bug.yml`](.github/ISSUE_TEMPLATE/bug.yml) — include version, Python version, reproduction steps, expected vs actual behavior. Issues missing a reproduction are auto-closed after 14 days.

## Requesting features

Use [`.github/ISSUE_TEMPLATE/feature.yml`](.github/ISSUE_TEMPLATE/feature.yml) — describe the problem first, then the proposed solution. For larger proposals, open a [GitHub Discussion](https://github.com/shyftlabs/continuum/discussions) before writing code.

## Issue labels

Issues are organized along three axes. The issue templates auto-apply a **type** label plus `needs-triage`; maintainers add **area** and **status** labels during triage.

| Axis | Labels | Meaning |
|---|---|---|
| **Type** | `bug`, `feature`, `question`, `documentation`, `research` | What kind of issue it is. `research` = investigation / design exploration, not a concrete bug or feature yet. |
| **Area** | `area:agents`, `area:workflows`, `area:memory`, `area:session`, `area:tools-mcp`, `area:llm`, `area:temporal`, `area:observability`, `area:streaming`, `area:evaluation`, `area:cli` | Which part of Continuum is affected — mirrors the "Affected surface" field in the feature template. An issue may carry more than one. |
| **Status** | `needs-triage`, `needs-repro`, `blocked`, `good first issue`, `help wanted`, `duplicate`, `invalid`, `wontfix` | Where the issue stands. Newly filed issues start at `needs-triage`. |

Triage convention (maintainers): replace `needs-triage` with the right `area:*` label(s) and, if applicable, a status label. A bug without a reproduction gets `needs-repro` and is auto-closed after 14 days if none is provided. **There is no public `security` label** — security reports go through the private advisory channel below, never a public issue.

## Security issues

**Do not file security issues in public.** See [`SECURITY.md`](SECURITY.md) for the private disclosure channel.

## Questions

Use [GitHub Discussions](https://github.com/shyftlabs/continuum/discussions) for usage questions. For commercial / enterprise inquiries: **continuum@shyftlabs.io**.
