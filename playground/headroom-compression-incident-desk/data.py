"""Deterministic synthetic incident data for the Incident Desk rig.

Everything is seeded, so the needles and aggregates are EXACT — e2e_test.py
asserts against GROUND_TRUTH, computed from the generated data itself (the
tests can never desync from the generators).

Payload sizes are tuned to the Headroom sidecar's thresholds:

  * logs         ~4,000 plain-text lines  → LogCompressor (lossy) + CCR marker
  * orders       43 uniform JSON rows     → SmartCrusher lossless reformat
                                            (below the sidecar's
                                            max_items_after_crush=50, so every
                                            row survives)
  * runbooks     20 search results        → SearchCompressor (lossy) + CCR
  * service.yaml one page of config       → via the `read` tool: on Headroom's
                                            DEFAULT_EXCLUDE_TOOLS, never touched
  * postmortem   ~6,000 words of prose    → Kompress candidate (scenario 9);
                                            kept well under the ~13k words where
                                            the spike hit Kompress's 20s deadline
"""

from __future__ import annotations

import random

# --- needles ---------------------------------------------------------------
# Buried mid-log as plain INFO lines (NOT errors — error lines tend to survive
# lossy log compression; an unremarkable audit line gets crushed away, which is
# what forces the model to call continuum_headroom_retrieve).
INCIDENT_TOKENS: dict[str, str] = {
    "checkout-api": "AMBER-FALCON-4413",
    "payments-svc": "CRIMSON-OTTER-9021",
}

POSTMORTEM_IMPACT_FACT = "4,127 checkout attempts failed"


# --- logs (LogCompressor + CCR) --------------------------------------------

_NOISE_TEMPLATES = [
    "INFO [http] {m} GET /api/v1/orders/{rid} 200 in {ms}ms",
    "INFO [http] {m} POST /api/v1/checkout 201 in {ms}ms",
    "INFO [cache] {m} hit key=cart:{rid} ttl={ms}s",
    "INFO [cache] {m} miss key=price:{rid} — fetched upstream in {ms}ms",
    "INFO [auth] {m} token validated for session s-{rid}",
    "INFO [worker] {m} job refund-{rid} dequeued (lag {ms}ms)",
    "INFO [metrics] {m} flushed 120 datapoints in {ms}ms",
    "INFO [gc] {m} minor collection freed {ms}KB",
    "DEBUG [http] {m} request headers parsed for req-{rid}",
    "INFO [pool] {m} connection checked out (active={act}/20)",
]


def _ts(seconds_offset: float) -> str:
    """Timestamp within the incident window, ~14:00Z + offset."""
    base_min = 14 * 60  # 14:00
    total = base_min * 60 + seconds_offset
    h, rem = divmod(int(total), 3600)
    m, s = divmod(rem, 60)
    ms = int((seconds_offset % 1) * 1000)
    return f"2026-07-08T{h:02d}:{m:02d}:{s:02d}.{ms:03d}Z"


def generate_logs(service: str = "checkout-api", lines: int = 900) -> str:
    """~900 lines of app logs with the incident token buried ~65% through.

    Sized so the RETRIEVED ORIGINAL fits the model's context: at ~0.42
    tokens/char these logs are ~31k tokens each, so even two retrieved
    originals in one run stay under the SDK's compression threshold
    (0.92 × 96k). At 4,000 lines (first cut) a retrieve worked but the SDK's
    own context manager then correctly truncated the 138k-token original away
    — you cannot retrieve more than the window holds.
    """
    token = INCIDENT_TOKENS[service]
    rng = random.Random(f"logs:{service}")
    needle_at = int(lines * 0.65)
    error_cluster_at = int(lines * 0.60)
    out: list[str] = [f"=== {service} application log — window 14:00Z–14:35Z ==="]

    for i in range(lines):
        t = _ts(i * 0.5)
        if i == needle_at:
            out.append(
                f"{t} INFO [audit] incident reference token recorded: {token} "
                f"(assigned by incident-bot, sev2)"
            )
            continue
        if error_cluster_at <= i < error_cluster_at + 12:
            out.append(
                f"{t} ERROR [pool] connection pool exhausted: max_size=20 "
                f"waiters={100 + i - error_cluster_at} timeout after 5000ms "
                f"(suspect: refund worker leaking connections)"
            )
            continue
        tpl = rng.choice(_NOISE_TEMPLATES)
        out.append(
            t
            + " "
            + tpl.format(
                m=f"req-{rng.randint(10_000, 99_999)}",
                rid=rng.randint(1_000, 9_999),
                ms=rng.randint(2, 480),
                act=rng.randint(1, 20),
            )
        )
    return "\n".join(out)


# --- orders (SmartCrusher, lossless) ----------------------------------------

_FAIL_REASONS = [
    "db_pool_timeout",
    "payment_gateway_5xx",
    "inventory_lock_timeout",
    "db_pool_timeout",
    "db_pool_timeout",
]


def generate_failed_orders() -> list[dict]:
    """43 failed orders. One (planted) has a uniquely large refund so the
    'largest refund' question is pure extraction, never arithmetic."""
    rng = random.Random("orders:failed")
    orders: list[dict] = []
    for i in range(43):
        amount = round(rng.uniform(18.0, 940.0), 2)
        orders.append(
            {
                "order_id": f"ORD-9{100 + i}",
                "status": "failed",
                "amount_usd": amount,
                "refund_amount_usd": amount,
                "failure_reason": rng.choice(_FAIL_REASONS),
                "customer_id": f"C-{rng.randint(10_000, 99_999)}",
                "created_at": _ts(rng.uniform(0, 2100)),
                "region": rng.choice(["us-east", "us-west", "eu-central"]),
                "gateway_txn": f"txn_{rng.getrandbits(64):016x}",
                "payment_method": rng.choice(["card", "wallet", "bank_transfer"]),
                "items_count": rng.randint(1, 8),
                "retry_attempts": rng.randint(0, 3),
            }
        )
    # the planted maximum — larger than the 18–940 range can produce
    orders[17]["amount_usd"] = 2499.00
    orders[17]["refund_amount_usd"] = 2499.00
    return orders


# --- runbooks (SearchCompressor) --------------------------------------------

_RUNBOOK_TOPICS = [
    ("RB-101", "High p99 latency on checkout-api", "latency"),
    ("RB-104", "Kafka consumer lag in order-events", "kafka"),
    ("RB-107", "Redis eviction storm under memory pressure", "redis"),
    ("RB-110", "TLS certificate rotation for edge proxies", "tls"),
    ("RB-113", "Stuck deployments: rollback procedure", "deploy"),
    ("RB-118", "Database connection pool exhaustion (checkout-api)", "pool"),
    ("RB-121", "Payment gateway 5xx storm: circuit-breaker playbook", "gateway"),
    ("RB-124", "Disk pressure on log volumes", "disk"),
    ("RB-127", "DNS resolution failures inside the mesh", "dns"),
    ("RB-130", "Rate-limiter misconfiguration triage", "ratelimit"),
    ("RB-133", "Slow queries after index bloat", "index"),
    ("RB-136", "S3 upload failures in receipt renderer", "s3"),
    ("RB-139", "Feature-flag rollout gone wrong", "flags"),
    ("RB-142", "OOMKilled pods in the checkout namespace", "oom"),
    ("RB-145", "Clock skew breaking token validation", "clock"),
    ("RB-148", "Webhook retry storms from partners", "webhook"),
    ("RB-151", "Split-brain in the session store", "session"),
    ("RB-154", "CDN cache poisoning response", "cdn"),
    ("RB-157", "Schema migration locked the orders table", "migration"),
    ("RB-160", "Elevated 401s after IdP maintenance", "idp"),
]

ANSWER_RUNBOOK_ID = "RB-118"


def generate_runbook_results(query: str) -> list[dict]:
    """20 search results, vector-search shaped. RB-118 is the right answer for
    pool-exhaustion queries; its snippet carries the distinctive remediation."""
    rng = random.Random("runbooks")
    results = []
    for rid, title, tag in _RUNBOOK_TOPICS:
        if rid == ANSWER_RUNBOOK_ID:
            snippet = (
                "Symptoms: waiters piling up on the connection pool, 5000ms "
                "checkout timeouts, ERROR [pool] connection pool exhausted. "
                "Remediation: (1) raise pool_max_size in service.yaml, "
                "(2) restart the refund worker (known connection leak), "
                "(3) enable connection reaping, (4) verify with pool_active gauge."
            )
            score = 0.94
        else:
            snippet = (
                f"Covers {tag} incidents: symptoms, detection queries, paging "
                f"policy, and step-by-step remediation. Last reviewed 2026-Q2. "
                + " ".join(
                    f"Step {n}: {rng.choice(['check', 'restart', 'scale', 'verify'])} "
                    f"the {tag} {rng.choice(['dashboard', 'service', 'config', 'alert'])}, "
                    f"then confirm the {tag} {rng.choice(['gauge', 'error rate', 'p99', 'queue depth'])} "
                    f"returns to baseline within {rng.randint(2, 15)} minutes before proceeding."
                    for n in range(1, 9)
                )
            )
            score = round(rng.uniform(0.35, 0.82), 2)
        # body_excerpt bulks each hit to realistic RAG-chunk size — without it
        # the whole result set stays under the sidecar's compression floor and
        # the search path is never actually exercised (observed: 0.0% saved).
        body = " ".join(
            f"{rng.choice(['Ensure', 'Confirm', 'Record', 'Escalate if'])} the "
            f"{tag} {rng.choice(['runbook step', 'alert threshold', 'oncall handoff', 'rollback window'])} "
            f"{rng.choice(['is acknowledged', 'has an owner', 'is linked in the incident channel', 'matches the SLO doc'])}."
            for _ in range(18)
        )
        results.append(
            {
                "id": rid,
                "title": title,
                "url": f"https://runbooks.internal/{rid.lower()}",
                "score": score,
                "snippet": snippet,
                "body_excerpt": body,
            }
        )
    results.sort(key=lambda r: -r["score"])
    return results


def format_runbook_results(query: str) -> str:
    """Search results in grep `file:line:content` format — deliberately.

    Probed against the sidecar (v0.29.0): Headroom's SEARCH detection is
    STRICTLY grep-shaped (`^[^\\s:]+:\\d+:`, ≥30% of lines). Text-heavy JSON
    arrays of {title, url, snippet} pass through 0%-compressed (SmartCrusher
    declines them; JSON detection also requires a top-level `[` array), and
    freeform text blocks fall to the Kompress path (passthrough unless the ML
    model is warm). If you want RAG/search payloads compressed by Headroom,
    serve them grep-style."""
    out = [f"query: {query!r} — 20 runbooks matched (grep format: path:line:text)"]
    for r in generate_runbook_results(query):
        path = f"runbooks/{r['id'].lower()}.md"
        n = 1
        out.append(f"{path}:{n}:# {r['id']}: {r['title']} (relevance {r['score']:.2f})")
        for chunk in (r["url"], r["snippet"], r["body_excerpt"]):
            n += 1
            out.append(f"{path}:{n}:{chunk}")
    return "\n".join(out)


# --- service.yaml (the `read` exclusion) -------------------------------------

SERVICE_YAML = """\
# checkout-api service configuration (rendered)
service:
  name: checkout-api
  version: 3.14.2
  environment: production
http:
  port: 8443
  read_timeout_ms: 5000
  write_timeout_ms: 5000
  max_body_kb: 512
database:
  driver: postgres
  host: orders-db.internal
  port: 5432
  pool_max_size: 20
  pool_min_idle: 4
  pool_acquire_timeout_ms: 5000
  statement_timeout_ms: 8000
cache:
  backend: redis
  host: cache.internal
  ttl_seconds: 300
workers:
  refund_worker_concurrency: 7
  refund_worker_batch: 25
  reconciliation_cron: "*/15 * * * *"
observability:
  tracing: enabled
  sample_rate: 0.2
  metrics_port: 9102
flags:
  new_pricing_engine: false
  async_receipts: true
"""


# --- postmortem prose (Kompress candidate, scenario 9) -----------------------

# NOTE: the sidecar's log detector matches ERROR/FAIL/WARN/INFO anywhere in a
# line, needing only 10% of lines — a paragraph-per-line document about an
# outage trips it easily (observed live: router:log:0.31, and the prose never
# reached the Kompress path). Hence: incident vocabulary that avoids those
# exact words, and generate_postmortem() wraps lines at ~90 chars so the one
# deliberate "failed" (the impact fact) stays under the ratio.
_PM_SENTENCES = [
    "The on-call engineer was paged after the checkout reliability budget began burning at roughly {n} times the normal rate.",
    "Dashboards showed the connection pool for the orders database pinned at its configured maximum while waiter counts climbed steadily.",
    "Initial triage focused on the payment gateway, which had shown intermittent 5xx responses earlier in the week, but gateway health was quickly ruled out.",
    "A rolling restart of the API tier produced a brief recovery followed by an identical regression within {n} minutes, which pointed away from transient state.",
    "Reviewing recent deploys surfaced a refund-worker change that had shipped {n} hours before the incident window opened.",
    "The refund worker acquired database connections inside a retry loop and, on one specific code path, returned from the function without releasing them.",
    "Each retry burst therefore leaked a small number of connections, and under the afternoon traffic peak the leak outpaced the pool's idle reclamation.",
    "Mitigation consisted of scaling the worker to zero, which released its held connections and restored checkout success rates within {n} minutes.",
    "A follow-up change added a context-managed acquisition pattern so that every code path releases its connection deterministically.",
    "Alerting gaps were also identified: pool saturation had no dedicated alert and was only visible as a secondary symptom of elevated checkout latency.",
    "The incident review agreed to add a pool_waiters alert with a {n}-minute burn window and to include pool gauges on the primary service dashboard.",
    "Load testing in staging had not exercised the problematic retry path because the injected fault profile never triggered the specific gateway timeout involved.",
    "Customer support saw a corresponding spike in contacts about checkouts that did not go through, concentrated in the {r} region where the traffic peak was strongest.",
    "The team also confirmed that no orders were double-charged, since the payment capture step is idempotent and keyed on the order identifier.",
    "Longer term, the service owners committed to connection-reaping in the pool configuration and a lint rule for unmanaged acquisitions.",
]


def generate_postmortem() -> str:
    """~6,000 words of prose about incident INC-2417. The impact figure
    (POSTMORTEM_IMPACT_FACT) is the extraction target for scenario 9.
    Paragraphs are wrapped at ~90 chars — see the note above _PM_SENTENCES."""
    import textwrap

    rng = random.Random("postmortem")
    wrap = lambda p: textwrap.fill(p, width=90)  # noqa: E731
    sections = [
        "# Postmortem INC-2417 — checkout-api connection pool exhaustion",
        "",
        "## Impact",
        wrap(
            "Over a twenty-six minute window on the afternoon of the incident, "
            f"a total of {POSTMORTEM_IMPACT_FACT} with the db_pool_timeout "
            "signature, corresponding to an estimated three hundred ten "
            "thousand dollars in delayed (not lost) order volume. There was no "
            "data loss and no double-charging. Incident reference token: "
            f"{INCIDENT_TOKENS['checkout-api']}."
        ),
        "",
        "## Narrative",
    ]
    word_target = 6000
    words = 0
    para: list[str] = []
    while words < word_target:
        s = rng.choice(_PM_SENTENCES).format(
            n=rng.randint(3, 40),
            r=rng.choice(["us-east", "us-west", "eu-central"]),
        )
        para.append(s)
        words += len(s.split())
        if len(para) >= rng.randint(4, 7):
            sections.append(wrap(" ".join(para)))
            sections.append("")
            para = []
    if para:
        sections.append(wrap(" ".join(para)))
    return "\n".join(sections)


# --- ground truth (asserted by e2e_test.py) ----------------------------------


def _orders_ground_truth() -> dict:
    orders = generate_failed_orders()
    largest = max(orders, key=lambda o: o["refund_amount_usd"])
    return {
        "failed_count": len(orders),
        "largest_refund_order": largest["order_id"],
        "largest_refund_amount": f"{largest['refund_amount_usd']:.2f}",
    }


GROUND_TRUTH: dict = {
    "orders": _orders_ground_truth(),
    "tokens": dict(INCIDENT_TOKENS),
    "runbook_id": ANSWER_RUNBOOK_ID,
    "config": {"pool_max_size": "20", "refund_worker_concurrency": "7"},
    "postmortem_fact": POSTMORTEM_IMPACT_FACT,
}


if __name__ == "__main__":
    logs = generate_logs()
    pm = generate_postmortem()
    print(f"logs: {len(logs.splitlines())} lines, {len(logs)} chars")
    print(f"orders: {len(generate_failed_orders())} rows")
    print(f"runbooks: {len(generate_runbook_results('pool'))} results")
    print(f"postmortem: {len(pm.split())} words")
    print(f"ground truth: {GROUND_TRUTH}")
