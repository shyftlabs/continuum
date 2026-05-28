<!--
Thanks for the PR! Please fill in the sections below.
PR title must follow Conventional Commits: feat: | fix: | docs: | chore: | test: | perf: | ci:
PRs target `dev` (not `main`). See CONTRIBUTING.md for the branch model.
-->

## What

<!-- One-line summary of the change. -->

## Why

<!-- The reason for the change — link the issue if there is one (e.g. Closes #123). -->

## How tested

<!-- unit / integration / manual. Paste output or a screenshot if helpful. -->

## Type of change

- [ ] `feat` — new user-facing feature
- [ ] `fix` — bug fix
- [ ] `docs` — documentation only
- [ ] `chore` — refactor / deps / tooling
- [ ] `test` — test-only change
- [ ] `perf` — performance improvement
- [ ] `ci` — CI / build config

## Checklist

- [ ] PR targets `dev` (not `main`)
- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Every commit is signed off (`git commit -s`) per the [DCO](https://developercertificate.org)
- [ ] `pytest` passes locally
- [ ] `ruff check .` passes
- [ ] `mypy src/orchestrator` clean for the code you changed (not yet a CI gate — repo-wide type debt tracked separately)
- [ ] Tests added or updated for the change
- [ ] Docs updated if user-facing
- [ ] `CHANGELOG.md` entry added under `## [Unreleased]` if user-facing
- [ ] No secrets, credentials, or `.env` files committed
