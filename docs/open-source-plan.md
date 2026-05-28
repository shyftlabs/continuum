# Open-Sourcing Continuum — Plan & Governance

Single source of truth for releasing Continuum as an OSS project owned by **ShyftLabs Inc.**

---

## 1 · License

ShyftLabs wants three properties at once:

1. Anyone can read, fork, learn from, and build on Continuum.
2. **No one can host Continuum as a paid service or embed it in a commercial product they sell.**
3. **Production commercial use by a for-profit organisation requires a Commercial Agreement with ShyftLabs.**

That rules out pure-OSS licenses (MIT, Apache 2.0, BSD, GPL) — they all explicitly permit hosting and resale. The right category is **source-available**.

### Options that fit

| License | What it blocks | What it allows | Notable users |
|---|---|---|---|
| **Business Source License (BSL) 1.1** | Whatever you write into the *Additional Use Grant* (e.g. no SaaS, no resale, no production by for-profits above $X revenue) | Reading, modifying, building, internal eval; auto-converts to Apache 2.0 after a Change Date you pick (typically 3–4 yr) | HashiCorp · MariaDB · CockroachDB · Sentry (historically) |
| **Functional Source License (FSL)** | "Competing use" — hosting, reselling, rebranding — for 2 years | Everything else; auto-converts to MIT or Apache after 2 yr | Sentry (current) · Keygen |
| **Elastic License v2 (ELv2)** | Offering as a hosted/managed service to third parties; removing license keys; bypassing notices | Internal production use, modification, redistribution as long as you don't host-for-others | Elasticsearch · Kibana · Logstash |
| **PolyForm Strict** | All commercial production use without a separate agreement | Personal, educational, evaluation, internal dev/test only | Smaller projects |

### Recommendation: **Business Source License 1.1**

Most flexible, most enterprise-recognised, and you write the exact restriction. Use these parameters:

| BSL parameter | Value |
|---|---|
| **Licensor** | ShyftLabs Inc. |
| **Licensed Work** | Continuum |
| **Change Date** | 2030-01-01 *(every release becomes Apache 2.0 four years after its publication date)* |
| **Change License** | Apache License, Version 2.0 |
| **Additional Use Grant** | (see below) |

**Additional Use Grant** — paste into the `LICENSE` file verbatim:

> You may use the Licensed Work for any non-production purpose, including evaluation, development, testing, internal demos, and contributions back to the Licensed Work, without limitation.
>
> You may also use the Licensed Work in production, **except** that you may not:
> (a) offer the Licensed Work or any derivative of it as a hosted, managed, or "as-a-service" offering to third parties;
> (b) embed the Licensed Work into a commercial product that you sell, sublicense, or otherwise distribute for value; or
> (c) use the Licensed Work in a production environment within a for-profit organisation with more than ten (10) employees or annual revenue above five hundred thousand US dollars (USD 500,000), whichever is reached first.
>
> For any use that falls under (a), (b), or (c), you must obtain a Commercial Agreement from ShyftLabs Inc. at **enterprise@shyftlabs.io**.

Tune the thresholds in clause (c) to your sales target. Common settings: 10 employees / $500k ARR (early-stage friendly) up to 250 employees / $10M ARR (enterprise-only restriction). Smaller numbers = more sales conversations; larger = more permissive.

### What this gives you

- **Adoption** — individuals, students, OSS projects, and small startups can use Continuum without ever talking to you.
- **Moat** — AWS, Azure, GCP, and competitors cannot offer "Continuum-as-a-service" without a contract.
- **Sales pipeline** — every for-profit team above the threshold has a legal reason to email enterprise@shyftlabs.io.
- **Community trust** — every version eventually becomes Apache 2.0 on its Change Date. You're not pulling the rug; you're holding the moat for four years per release.

### What this *cannot* do

No license can force someone to email you before they self-host internally. The restriction triggers when their use crosses one of clauses (a)–(c). The "must contact us" outcome is enforced by:

1. The license making them legally ineligible to use it that way without a Commercial Agreement.
2. ShyftLabs offering tangible enterprise value they want — SLAs, security audits, indemnification, priority fixes, custom features, training — bundled into that Commercial Agreement.

### Files to update at the flip

- Replace `LICENSE` with the BSL 1.1 template from <https://mariadb.com/bsl11/>, filling in the parameters above.
- Add a `NOTICE` file:
  > Continuum © 2025–2026 ShyftLabs Inc. Licensed under the Business Source License 1.1. After the Change Date for each release, that release is also available under the Apache License, Version 2.0.
- Add to `README.md`:
  > Continuum is source-available under the **Business Source License 1.1** — free for non-production use and for small teams in production. For production use at scale, hosted offerings, or commercial embedding, contact **enterprise@shyftlabs.io**.

---

## 2 · Contributor License Agreement (CLA)

ShyftLabs needs unambiguous ownership of contributed code (so you can relicense, sell, or change the model later). Two practical options:

| Option | When to pick |
|---|---|
| **CLA Assistant** ([cla-assistant.io](https://cla-assistant.io)) — bot-enforced sign-off on each PR | If you want frictionless contributor onboarding (recommended) |
| **DCO** (Developer Certificate of Origin) — `Signed-off-by:` line | Lighter-weight, used by Linux kernel; weaker IP guarantee than a CLA |

Use the **Apache ICLA template** (Individual) and **Apache CCLA template** (Corporate) verbatim — they are battle-tested.

---

## 3 · Branch model

```
main          ← stable, tagged releases, what users `pip install`
└── dev       ← integration, all PRs land here first
    ├── feat/<short-slug>       ← feature work, squash-merged into dev
    ├── fix/<short-slug>        ← bug fixes
    └── chore/<short-slug>      ← refactors, deps, docs
```

| Rule | Why |
|---|---|
| `main` is protected: required PR review, required CI green, no force-push | Stability for users |
| PRs always target `dev`, never `main` | Single integration line |
| `dev` → `main` only via a release PR cut by a maintainer | Forces an intentional release moment |
| Tags `v0.x.y` on `main` only | Semver discipline |
| Long-lived branches deleted after merge | Repo hygiene |

Squash-merge into `dev` (one PR = one commit). Merge-commit `dev` → `main` so the release shows the full history.

---

## 4 · Issue triage

| Label | Meaning | SLA — first response |
|---|---|---|
| `bug` | Confirmed defect | 2 working days |
| `question` | Usage help | 3 working days, or close with link to docs |
| `feature` | New capability | 5 working days, may get `needs-design` |
| `security` | Vuln — see §6 | **Do not file as issue** (private channel) |
| `good-first-issue` | New-contributor friendly | Maintainer reserves these |
| `needs-repro` | Missing reproduction | Auto-close after 14 days idle |

Templates: ship `.github/ISSUE_TEMPLATE/bug.yml`, `feature.yml`, `question.yml`.

---

## 5 · Pull-request workflow

Required by branch protection on `dev`:

1. CI green: lint (`ruff`), type-check (`mypy`), tests (`pytest`), security scan (see §6).
2. 1 approving review from a maintainer.
3. PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`).
4. CLA signed.
5. Description fills the template: *what / why / how tested*.

PR template (`.github/pull_request_template.md`):

```markdown
## What
<one-line summary>

## Why
<reason / linked issue>

## How tested
<unit / integration / manual>

- [ ] I have signed the CLA
- [ ] I added/updated tests
- [ ] I updated docs/CHANGELOG.md if user-facing
```

Maintainer response targets: first reply within **3 working days**, decision within **7**.

---

## 6 · Security disclosure

Add `SECURITY.md` at the repo root with this contract:

| Channel | Who sees it |
|---|---|
| GitHub Security Advisory (private) — preferred | Maintainers only |
| Email `security@shyftlabs.io` (PGP key in `SECURITY.md`) | Security lead only |

| Severity | Acknowledge | Fix target |
|---|---|---|
| Critical (RCE, secret exfil) | 24 h | 7 days |
| High (auth bypass, data leak) | 48 h | 14 days |
| Medium / Low | 5 working days | next minor |

Never disclose details publicly until a patched version is shipped and users have a 30-day window to upgrade. Credit the reporter in the release notes unless they decline.

CI security gates (block merge if any trips):
- `gitleaks` — secrets scan on every PR
- `pip-audit` — Python dep vulns
- `bandit` — Python AST security lint
- Dependabot for monthly dep PRs

---

## 7 · Community response — tone & cadence

| Surface | Cadence | Owner |
|---|---|---|
| Issues | Triage daily, respond ≤ 3 d | Rotating maintainer |
| Pull requests | First review ≤ 3 d | Code owner of touched area |
| Discussions / Q&A | Weekly sweep | Anyone with commit access |
| Security email | ≤ 24 h | Security lead |

**Tone rules** (write these into a one-page `MAINTAINERS.md`):

- Lead with thanks. "Thanks for the PR / report" — always.
- Be concrete in critique: link to the line + propose a fix, never "this is wrong."
- Say "no" with a reason and a pointer ("we'd accept a version that does X because Y").
- Never personalise. Critique the patch, not the person.
- If a thread heats up, mute notifications, take 24 h, respond from a draft.
- Public conversations only — refuse to move technical decisions to DMs. (Exception: security.)

Escalation path: contributor → maintainer → **steering committee** (3 ShyftLabs staff, rotating quarterly). Steering committee resolves disputes that maintainers can't.

---

## 8 · Files to add to the repo

Checklist before the public flip:

- [ ] `LICENSE` — BSL 1.1 template (replace existing), with Additional Use Grant from §1
- [ ] `NOTICE` — ShyftLabs copyright + BSL summary + post-Change-Date Apache attribution
- [ ] `CONTRIBUTING.md` — branch model + PR template summary, links here
- [ ] `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1 verbatim
- [ ] `SECURITY.md` — disclosure channel + severity SLAs
- [ ] `MAINTAINERS.md` — current maintainer list + tone rules
- [ ] `CHANGELOG.md` — `Keep a Changelog` format, `Unreleased` section at top
- [ ] `.github/ISSUE_TEMPLATE/{bug,feature,question}.yml`
- [ ] `.github/pull_request_template.md`
- [ ] `.github/workflows/` — CI (lint + type + test), security scan, release-please
- [ ] `.github/CODEOWNERS` — maintainer → directory mapping
- [ ] CLA bot configured ([cla-assistant.io](https://cla-assistant.io))
- [ ] Branch protection on `main` and `dev` (require review + green CI, no force-push)
- [ ] Dependabot config (`.github/dependabot.yml`)
- [ ] `README.md` — add Apache badge + enterprise contact line

---

## 9 · Release cadence

- **Patch** (`v0.x.Y`) — when fixes accumulate, ad-hoc
- **Minor** (`v0.X.0`) — every 4–6 weeks, with CHANGELOG entries
- **Major** — only for breaking API changes; pre-announce on Discussions ≥ 30 days

Use [`release-please`](https://github.com/googleapis/release-please) to automate version bumps and CHANGELOG from Conventional Commits.

---

## 10 · Pre-flip cleanup

Before flipping the repo to public:

1. **Audit history for secrets** — `git secrets --scan-history` + Trufflehog. If anything leaks, rewrite history before going public (we already removed `Continuum_Agent_Framework_Documentation.docx` from `shyftlabs/main` history — same playbook).
2. **Strip internal references** — Slack channels, internal URLs, customer names in comments.
3. **Validate `.env.template`** — placeholders only, no real keys.
4. **Run the full test suite green** on a fresh clone.
5. **Tag `v0.2.0`** as the public launch version and write the release notes.

---

*Owner: ShyftLabs Inc. · Maintained by Bhavik Ardeshna · enterprise inquiries: enterprise@shyftlabs.io*
