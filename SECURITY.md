# Security Policy

ShyftLabs Inc. takes the security of Continuum seriously. This document describes how to report a vulnerability, what to expect after you do, and which versions receive fixes.

## Supported versions

We patch security issues on the **latest minor release** of Continuum. Older minor releases receive a fix only if the underlying issue is critical and an upgrade path is non-trivial.

| Version | Supported |
|---|---|
| `0.2.x` (latest) | ✅ |
| `< 0.2.0` | ❌ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security reports.** Public disclosure before a patch is available puts other users at risk.

Use one of the two private channels below:

| Channel | When to use | Who sees it |
|---|---|---|
| [GitHub Security Advisory](https://github.com/shyftlabs/continuum/security/advisories/new) **(preferred)** | Default for all reports | Repository maintainers only |
| Email `continuum@shyftlabs.io` | If you cannot use GitHub, or you want PGP encryption | Security lead only |

Include in your report:

- A clear description of the vulnerability and the affected component.
- A minimal reproduction or proof-of-concept (code, request, configuration).
- The Continuum version, Python version, and any relevant dependency versions.
- The impact you observed and any impact you suspect but did not confirm.
- Whether you intend to disclose the issue publicly, and on what timeline.

If you would like to encrypt your email, request our PGP key in your first message and we will send it back over a side channel.

## What to expect

| Stage | Target |
|---|---|
| Acknowledgement that we received the report | within **24 hours** for critical and high severity, **5 working days** otherwise |
| Initial assessment and severity classification | within **3 working days** of acknowledgement |
| Patch development and review | per severity, see below |
| Coordinated public disclosure | after a patched release is available and users have had **30 days** to upgrade |

### Severity-driven fix targets

| Severity | Examples | Fix target |
|---|---|---|
| **Critical** | Remote code execution, secret exfiltration, full auth bypass | **7 days** to patched release |
| **High** | Privilege escalation, sensitive data leak, persistent XSS in shipped UIs | **14 days** to patched release |
| **Medium** | DoS that requires unusual conditions, info-disclosure with limited impact | next minor release |
| **Low** | Hardening improvements, defense-in-depth | next minor release |

Targets are measured from the moment severity is confirmed, not from initial submission. If a fix will slip a target, the reporter is notified in writing with a revised date and reason.

## Disclosure and credit

- We never disclose vulnerability details publicly until a patched version is shipped **and** users have had a 30-day upgrade window, unless the issue is already public.
- The reporter is credited by name (or handle) in the release notes and the GitHub Security Advisory, unless they explicitly decline.
- We do not currently operate a paid bug-bounty programme. We will, where appropriate, send a thank-you and Continuum swag.

## Scope

In-scope:

- Code in this repository under `src/`, `scripts/`, `examples/`, and the published Python package `shyftlabs-continuum`.
- Default configurations shipped in `.env.template`, `docker-compose.yml`, and `pyproject.toml`.
- Documentation under `docs/` that recommends an insecure pattern or configuration.

Out of scope (please do **not** report):

- Vulnerabilities in dependencies that have already been disclosed upstream — open a normal issue if the version pin needs to change.
- Issues that require an attacker to already have local code execution or root on the host.
- Social-engineering attacks against maintainers or contributors.
- Self-XSS, missing security headers on non-authenticated marketing pages, or theoretical issues without a working reproduction.

## CI security gates

The following checks run on every pull request and block merge if they trip:

- `gitleaks` — secret scanning on the diff and full history of the PR branch.
- `pip-audit` — Python dependency vulnerabilities against the resolved environment.
- `bandit` — Python AST security lint on `src/`.

Dependabot opens monthly PRs for `pip`, `github-actions`, and `docker` ecosystems against the `dev` branch.

## Hardening recommendations for operators

If you are running Continuum in production, please also:

- Pin your Continuum version (`shyftlabs-continuum==X.Y.Z`) and review the `CHANGELOG.md` and Security Advisories before upgrading.
- Subscribe to repository Security Advisories (Watch → Custom → Security alerts).
- Run the agent process with the minimum privileges it needs — never as root, never with broader cloud-IAM scopes than the deployed agents require.
- Treat LLM outputs as untrusted input when feeding them into tools, shells, or database queries.
- Keep `mem0`, `Milvus`, `Qdrant`, `Redis`, `Temporal`, and `Langfuse` reachable only from the application network — never expose them to the public internet without an authenticated proxy.

## Questions

Non-vulnerability security questions (e.g. "how do I configure X safely") belong in [GitHub Discussions](https://github.com/shyftlabs/continuum/discussions). Commercial / enterprise security inquiries (SOC 2, indemnification, custom hardening) go to **continuum@shyftlabs.io**.

---

*Last updated: 2026-05.*
