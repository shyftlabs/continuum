# Changelog

All notable changes to Continuum are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and Continuum adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub issue templates (`bug.yml`, `feature.yml`, `question.yml`) and a chooser config that disables blank issues and routes security reports to a private advisory.
- Pull request template with Conventional-Commits typing, DCO checkbox, and lint/type/test gates.
- `CODEOWNERS` for automatic review routing.
- Dependabot configured for security-only updates (`pip`, `github-actions`); routine version-bump PRs are disabled.
- `MAINTAINERS.md` with the current maintainer list, tone rules, and escalation path.
- `SECURITY.md` with the private disclosure channel and severity SLAs.
- CI workflow (`ruff`, `mypy`, `pytest`) and security scan workflow (`gitleaks`, `pip-audit`, `bandit`).
- `release-please` workflow for automated version bumps and changelog updates.

### Changed
- _Nothing yet._

### Deprecated
- _Nothing yet._

### Removed
- _Nothing yet._

### Fixed
- _Nothing yet._

### Security
- _Nothing yet._

---

## [0.2.0] — 2026-05

Initial public release. See the repository history for details prior to this changelog being introduced.

[Unreleased]: https://github.com/shyftlabs/continuum/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shyftlabs/continuum/releases/tag/v0.2.0
