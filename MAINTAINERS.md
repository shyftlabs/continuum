# Maintainers

Continuum is maintained by **ShyftLabs Inc.** This document lists the current maintainer team, the areas they own, the response cadence the community can expect, and the tone rules every maintainer is expected to follow.

If you are looking for **how to contribute**, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
If you are reporting a **security issue**, see [`SECURITY.md`](SECURITY.md) — do **not** open a public issue.

---

## Current maintainers

| Name | GitHub | Role | Areas |
|---|---|---|---|
| Bhavik Ardeshna | [@bhavik-shyftlabs](https://github.com/bhavik-shyftlabs) | Lead maintainer | All areas — core agents, runner, memory, workflows, releases |

> Additional maintainers will be listed here as they are added. To propose a new maintainer, open a discussion in [Discussions → Governance](https://github.com/shyftlabs/continuum/discussions).

## Steering committee

A 3-member committee of ShyftLabs staff resolves disputes that individual maintainers cannot, decides on major architectural changes, and approves new maintainers. The committee rotates quarterly.

Current committee — **2026 Q2**:

- Bhavik Ardeshna (chair)
- _seat 2 — open_
- _seat 3 — open_

Escalation path: contributor → maintainer → steering committee.

## Response cadence

These are *targets*, not contractual SLAs. They define what the community should expect under normal load.

| Surface | First response | Decision / merge target | Owner |
|---|---|---|---|
| Bug reports (`bug`) | 2 working days | Triaged within 5 working days | On-call maintainer |
| Feature requests (`feature`) | 5 working days | May get `needs-design` and move to Discussions | On-call maintainer |
| Questions (`question`) | 3 working days, or close with a docs link | — | Any maintainer |
| Pull requests | 3 working days first review | Approve / reject within 7 working days | Code owner of touched area |
| Discussions | Weekly sweep | — | Rotating |
| Security email | ≤ 24 hours | Per severity in [`SECURITY.md`](SECURITY.md) | Security lead |

If you have not received a response within the targets above, ping the thread once and tag `@bhavik-shyftlabs`. Do not open duplicate issues.

## Tone rules

Every maintainer reply on the public tracker is held to these rules. They exist because friendly, specific feedback is what keeps contributors coming back.

1. **Lead with thanks.** "Thanks for the PR / report" — always, even on PRs you will decline.
2. **Be concrete in critique.** Link the line, propose the fix. Never "this is wrong" without a pointer to what `right` looks like.
3. **Say no with a reason and a pointer.** "We would accept a version that does X because Y" beats a flat rejection.
4. **Critique the patch, not the person.** No "you should have known," no "obviously." Surface the issue, not a judgement of the contributor.
5. **Cool off before flame.** If a thread heats up, mute notifications, take 24 hours, respond from a draft.
6. **Decisions happen in public.** Refuse to move technical discussion to DMs. The only exceptions are security disclosures and commercial / enterprise inquiries.
7. **Hold the bar; do not move it.** If a PR is 80% there, say what the remaining 20% looks like. Don't merge to be nice, and don't reject because of taste alone — both signal an unstable bar.

## Adding a new maintainer

A contributor may be invited to become a maintainer after:

- Sustained, high-quality contributions over at least 8 weeks across multiple PRs.
- At least one feature or sub-system shipped end-to-end (design → code → docs → release).
- Demonstrated alignment with the tone rules in code review.

Process:

1. Existing maintainer nominates in a private steering-committee thread with a summary of contributions.
2. Steering committee approves by simple majority.
3. Nominee is added to this file, the GitHub team, `CODEOWNERS`, and the relevant branch-protection rule lists in the same PR.

## Stepping down

Maintainers may step down at any time by removing themselves from this file and `CODEOWNERS` in a single PR. Emeritus maintainers are credited in the release notes of the next minor version.

---

*Last updated: 2026-05. Steering composition is reviewed quarterly.*
