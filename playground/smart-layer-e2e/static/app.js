/**
 * Routing playground — chat, transparency, stats, golden benchmarks.
 */

const SESSION_KEY = "routing_playground_session_id";
const CLIENT_STATS_KEY = "routing_playground_client_stats";

function ensureSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function sessionHeaders() {
  return {
    "Content-Type": "application/json",
    "X-Routing-Session-Id": ensureSessionId(),
  };
}

function loadClientStats() {
  try {
    return JSON.parse(localStorage.getItem(CLIENT_STATS_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

function saveClientStats(obj) {
  localStorage.setItem(CLIENT_STATS_KEY, JSON.stringify(obj));
}

function bumpClientTier(tier) {
  const st = loadClientStats();
  st.turns = (st.turns || 0) + 1;
  st.tier_counts = st.tier_counts || {};
  st.tier_counts[tier] = (st.tier_counts[tier] || 0) + 1;
  const light = new Set(["nano", "fast"]);
  const heavy = new Set(["balanced", "specialist", "frontier"]);
  st.light_turns = (st.light_turns || 0) + (light.has(tier) ? 1 : 0);
  st.heavy_turns = (st.heavy_turns || 0) + (heavy.has(tier) ? 1 : 0);
  saveClientStats(st);
  return st;
}

function resetClientStats() {
  localStorage.removeItem(CLIENT_STATS_KEY);
}

function modeLabel(info, id) {
  const m = (info.classifier_modes || []).find((x) => x.id === id);
  return m ? m.label : id;
}

function tierBadge(tier) {
  const t = (tier || "").toLowerCase();
  return `<span class="badge ${t}">${t || "?"}</span>`;
}

function renderRoutingStory(info, routing, errorMsg) {
  const el = document.getElementById("routing-story");
  if (!routing || Object.keys(routing).length === 0) {
    el.innerHTML = errorMsg
      ? `<p class="err">Error: ${escapeHtml(errorMsg)}</p><p class="muted">No routing context was available before the failure.</p>`
      : "<p class=\"muted\">Send a message to see routing details.</p>";
    return;
  }

  const tier = routing.tier;
  const modeId = routing.tier_classifier;
  const modeTitle = modeLabel(info, modeId);
  const skipReason = routing.classifier_skip_reason || null;
  const shortcut = routing.skipped_classifier;
  const tc = routing.timings_ms || {};
  const cls = tc.classify ?? "—";
  const llm = tc.llm ?? "—";
  const total =
    typeof tc.classify === "number" && typeof tc.llm === "number"
      ? (tc.classify + tc.llm).toFixed(2)
      : "—";

  const routed = routing.routed_model || "—";
  const exec = routing.execution_model || "—";
  const modelNote =
    routed !== exec
      ? `<p><strong>Models:</strong> intended slot model <code>${escapeHtml(routed)}</code> → actual completion <code>${escapeHtml(exec)}</code> (deployment forces a single completion model).</p>`
      : `<p><strong>Model:</strong> <code>${escapeHtml(exec)}</code> (same as tier slot).</p>`;

  let pathTxt;
  if (skipReason === "heuristic_shortcut") {
    pathTxt =
      "This tier came from a <strong>fast keyword / length shortcut</strong> (no classifier LLM call). Uncheck “Fast keyword…” in the sidebar to disable.";
  } else if (skipReason === "forced_tier") {
    pathTxt = "Tier was <strong>forced</strong> by your request; the classifier did not run.";
  } else if (skipReason === "light_only" || skipReason === "heavy_only") {
    pathTxt = `Classifier mode is fixed (<strong>${escapeHtml(skipReason)}</strong>); the tier classifier LLM was not used.`;
  } else if (shortcut) {
    pathTxt = "The classifier step was skipped (see routing payload <code>classifier_skip_reason</code>).";
  } else {
    pathTxt = "The tier was chosen by the <strong>classifier LLM</strong>.";
  }

  el.innerHTML = `
    <div class="routing-story">
      <p><strong>Product tier</strong> ${tierBadge(tier)} <span class="muted">(nano … frontier)</span></p>
      <p><strong>Routing mode</strong> ${escapeHtml(modeTitle)} <span class="muted">(${escapeHtml(modeId)})</span></p>
      <p>${pathTxt}</p>
      ${modelNote}
      <p><strong>Time story</strong></p>
      <dl>
        <dt>Choosing tier</dt><dd>${cls} ms</dd>
        <dt>Generating answer</dt><dd>${llm} ms</dd>
        <dt>Total this turn</dt><dd>${total} ms</dd>
      </dl>
      ${errorMsg ? `<p class="err">Turn failed after routing: ${escapeHtml(errorMsg)}</p>` : ""}
    </div>
  `;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function applyEnvDefaultsToForm(info) {
  const ef = info.effective_from_env;
  if (!ef) return;

  const sel = document.getElementById("tier-classifier");
  if (ef.tier_classifier && sel.querySelector(`option[value="${ef.tier_classifier}"]`)) {
    sel.value = ef.tier_classifier;
  }
  if (ef.tier_classifier_llm_model) {
    document.getElementById("classifier-llm-model").value = ef.tier_classifier_llm_model;
  }
  if (ef.tier_router_api_base) {
    document.getElementById("tier-router-api-base").value = ef.tier_router_api_base;
  }
  if (ef.tier_local_router_api_base) {
    document.getElementById("tier-local-router-api-base").value = ef.tier_local_router_api_base;
  }
  if (ef.tier_force_completion_model) {
    document.getElementById("force-completion-model").value = ef.tier_force_completion_model;
  }
  if (typeof ef.tier_classifier_heuristic_shortcut === "boolean") {
    document.getElementById("tier-heuristic-shortcut").checked = ef.tier_classifier_heuristic_shortcut;
  }
}

async function loadInfo() {
  const r = await fetch("/info");
  const info = await r.json();
  window.__serverInfo = info;

  document.getElementById("hero-title").textContent = info.product_name || "Routing playground";
  document.getElementById("hero-desc").textContent =
    info.tagline ||
    "Try tier routing with transparency (tier, mode, models, timings).";

  let banner = "";
  if (info.provider_hint) {
    banner += `<p class="err" style="margin:0.35rem 0 0">${escapeHtml(info.provider_hint)}</p>`;
  }
  if (!info.openai_api_key_configured) {
    banner +=
      '<p class="err" style="margin:0.35rem 0 0">OPENAI_API_KEY is not set — chat and benchmarks will fail until it is configured.</p>';
  }
  if (banner) {
    document.getElementById("hero-desc").insertAdjacentHTML("afterend", banner);
  }

  const sel = document.getElementById("tier-classifier");
  sel.innerHTML = "";
  const serverDef = info.effective_from_env?.tier_classifier || info.server_default_classifier || "gpt_4o_mini";
  for (const m of info.classifier_modes || []) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.label}`;
    if (m.id === serverDef) opt.selected = true;
    sel.appendChild(opt);
  }

  applyEnvDefaultsToForm(info);

  const ek = info.effective_from_env?.tier_router_api_key_configured ? "yes (from env)" : "no";
  const heur =
    info.effective_from_env?.tier_classifier_heuristic_shortcut !== false ? "on" : "off (LLM always)";
  document.getElementById("server-default-hint").textContent =
    `Effective classifier after LLM_ROUTE_* / env: ${serverDef}. Keyword shortcuts: ${heur}. External classifier API key loaded: ${ek}. Plain RouterConfig default classifier was ${info.server_default_classifier}.`;

  const reg = await fetch("/classifiers/registry").then((x) => x.json());
  const locSel = document.getElementById("local-classifier");
  locSel.innerHTML = '<option value="">— None —</option>';
  for (const item of reg.items || []) {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = item.label;
    locSel.appendChild(opt);
  }
  if (!(reg.items || []).length) {
    document.getElementById("local-classifier-hint").textContent =
      "No entries in SMART_LAYER_LOCAL_CLASSIFIERS — optional local registry is empty.";
  }

  await refreshStats();
}

async function refreshStats() {
  const r = await fetch("/routing/stats", { headers: { "X-Routing-Session-Id": ensureSessionId() } });
  const srv = await r.json();
  const cli = loadClientStats();

  document.getElementById("stats-server").innerHTML = `
    <div class="stats-grid">
      <div><span>Requests (server)</span>${srv.total_requests ?? 0}</div>
      <div><span>Light tiers</span>${srv.light_turns ?? 0}</div>
      <div><span>Heavy tiers</span>${srv.heavy_turns ?? 0}</div>
      <div><span>Avg classify*</span>${srv.averages_ms?.classifier_when_llm_ran ?? "—"} ms</div>
      <div><span>Avg completion</span>${srv.averages_ms?.completion ?? "—"} ms</div>
    </div>
    <p class="server-sync">*Classifier average excludes shortcut-only turns (matches “real classifier” latency).</p>
    <p class="server-sync">Tier counts: ${JSON.stringify(srv.tier_counts || {})}</p>
  `;

  document.getElementById("stats-client").innerHTML = `
    <div class="stats-grid">
      <div><span>This browser session</span>${cli.turns || 0} turns</div>
      <div><span>Light</span>${cli.light_turns || 0}</div>
      <div><span>Heavy</span>${cli.heavy_turns || 0}</div>
    </div>
    <p class="server-sync">Client counts update on each completed stream; use them as a quick cost/capability mix hint.</p>
  `;
}

async function clearStats() {
  await fetch("/routing/stats/clear", {
    method: "POST",
    headers: { "X-Routing-Session-Id": ensureSessionId() },
  });
  resetClientStats();
  await refreshStats();
}

async function newSession() {
  const old = localStorage.getItem(SESSION_KEY);
  if (old) {
    await fetch("/routing/stats/clear", {
      method: "POST",
      headers: { "X-Routing-Session-Id": old },
    }).catch(() => {});
  }
  localStorage.removeItem(SESSION_KEY);
  resetClientStats();
  ensureSessionId();
  refreshStats();
  document.getElementById("routing-story").innerHTML =
    '<p class="muted">New session ID generated; server counters are fresh for this ID.</p>';
}

function collectControls() {
  const tier_classifier = document.getElementById("tier-classifier").value || null;
  const forced = document.getElementById("forced-tier").value || null;
  const forceModel = document.getElementById("force-completion-model").value.trim() || null;
  const clsModel = document.getElementById("classifier-llm-model").value.trim() || null;
  const apiBase = document.getElementById("tier-router-api-base").value.trim() || null;
  const apiKey = document.getElementById("tier-router-api-key").value.trim() || null;
  const localBase = document.getElementById("tier-local-router-api-base").value.trim() || null;
  const localKey = document.getElementById("tier-local-router-api-key").value.trim() || null;
  const heuristicShortcut = document.getElementById("tier-heuristic-shortcut").checked;

  return {
    tier_classifier,
    forced_product_tier: forced,
    tier_force_completion_model: forceModel,
    tier_classifier_llm_model: clsModel,
    tier_router_api_base: apiBase,
    tier_router_api_key: apiKey,
    tier_local_router_api_base: localBase,
    tier_local_router_api_key: localKey,
    tier_classifier_heuristic_shortcut: heuristicShortcut,
  };
}

async function sendChat() {
  const msg = document.getElementById("message-input").value.trim();
  if (!msg) return;

  const info = window.__serverInfo || {};
  const body = { message: msg, ...collectControls() };

  const log = document.getElementById("chat-log");
  log.innerHTML = `<div class="meta">Sending…</div>`;

  document.getElementById("send-btn").disabled = true;

  let assistant = "";
  let lastRouting = {};
  let streamError = null;

  try {
    const res = await fetch("/chat/stream", {
      method: "POST",
      headers: sessionHeaders(),
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";

      for (const block of parts) {
        const line = block.trim();
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        let ev;
        try {
          ev = JSON.parse(raw);
        } catch {
          continue;
        }

        if (ev.type === "routing") {
          lastRouting = { ...ev };
          renderRoutingStory(info, lastRouting, null);
        }
        if (ev.type === "content" && ev.text) {
          assistant += ev.text;
          log.innerHTML = `<div class="meta">Assistant (streaming)</div>${escapeHtml(assistant)}`;
        }
        if (ev.type === "done" && ev.routing) {
          lastRouting = ev.routing;
          bumpClientTier(ev.routing.tier);
          renderRoutingStory(info, lastRouting, null);
          await refreshStats();
        }
        if (ev.type === "error") {
          streamError = ev.error || "Unknown error";
          renderRoutingStory(info, ev.routing || lastRouting, streamError);
        }
      }
    }

    log.innerHTML = `<div class="meta">Assistant</div>${escapeHtml(assistant || "(empty)")}`;
    if (streamError) {
      log.innerHTML += `<p class="err">${escapeHtml(streamError)}</p>`;
    }
  } catch (e) {
    log.innerHTML = `<p class="err">${escapeHtml(String(e))}</p>`;
    renderRoutingStory(info, lastRouting, String(e));
  } finally {
    document.getElementById("send-btn").disabled = false;
  }
}

async function loadGoldenMeta() {
  const el = document.getElementById("golden-meta");
  try {
    const m = await fetch("/golden/metadata").then((r) => r.json());
    el.textContent = `${m.total_examples} labeled examples — breakdown: ${JSON.stringify(m.expected_tier_breakdown)}`;
  } catch (e) {
    el.textContent = `Could not load golden metadata: ${e}`;
  }
}

async function parseHttpError(res) {
  const raw = await res.text();
  try {
    const j = JSON.parse(raw);
    let d = j.detail ?? raw;
    if (Array.isArray(d)) d = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
    return String(d);
  } catch {
    return raw || res.statusText;
  }
}

async function runGoldenRouteOnly() {
  const box = document.getElementById("golden-results");
  box.innerHTML = '<p class="busy">Running route-only benchmark…</p>';
  const maxRows = parseInt(document.getElementById("golden-max-rows").value || "50", 10);
  try {
    const res = await fetch("/golden/benchmark/route-only", {
      method: "POST",
      headers: sessionHeaders(),
      body: JSON.stringify({ ...collectControls(), max_rows: maxRows }),
    });
    if (!res.ok) throw new Error(await parseHttpError(res));
    const result = await res.json();

    const rows = result.per_row || [];
    box.innerHTML = `
      <p><strong>Top-1 accuracy</strong> ${result.top1_accuracy} ·
      <strong>Escalation (heavy tiers)</strong> ${result.escalation_rate_heavy_tiers} ·
      <strong>Classifier ms mean (when LLM ran)</strong> ${result.classifier_latency_ms?.mean_when_classifier_ran ?? "—"}</p>
      <p><strong>Cost proxy</strong> vs all-frontier spend ratio: ${result.cost_proxy?.relative_spend_vs_all_frontier}</p>
      <div class="table-wrap"><table><thead><tr><th>Match</th><th>Expected</th><th>Predicted</th><th>Shortcut</th><th>Cls ms</th><th>Query</th></tr></thead><tbody>
      ${rows
        .map(
          (row) =>
            `<tr><td>${row.match ? "✓" : "✗"}</td><td>${row.expected_tier}</td><td>${row.predicted_tier}</td><td>${row.skipped_classifier}</td><td>${row.classify_ms}</td><td>${escapeHtml(row.query_preview)}</td></tr>`
        )
        .join("")}
      </tbody></table></div>
    `;
  } catch (e) {
    box.innerHTML = `<p class="err">${escapeHtml(String(e))}</p>`;
  }
}

async function runGoldenE2E() {
  const box = document.getElementById("golden-results");
  box.innerHTML = '<p class="busy">Running e2e + judge (slow, uses many completions)…</p>';
  const maxRows = parseInt(document.getElementById("golden-e2e-rows").value || "5", 10);
  try {
    const res = await fetch("/golden/benchmark/e2e-judge", {
      method: "POST",
      headers: sessionHeaders(),
      body: JSON.stringify({ ...collectControls(), max_rows: maxRows }),
    });
    if (!res.ok) throw new Error(await parseHttpError(res));
    const result = await res.json();

    const rows = result.per_row || [];
    box.innerHTML = `
      <p><strong>Mean judge (routed)</strong> ${result.mean_judge_routed} ·
      <strong>Mean PGR</strong> ${result.mean_pgr} <span class="muted">(${result.pgr_formula})</span></p>
      <div class="table-wrap"><table><thead><tr><th>Exp</th><th>Routed</th><th>Judge R/N/F</th><th>PGR</th><th>Cls/Llm ms</th><th>Query</th></tr></thead><tbody>
      ${rows
        .map(
          (row) =>
            `<tr><td>${row.expected_tier}</td><td>${row.routed_tier}</td><td>${row.judge_routed}/${row.judge_nano}/${row.judge_frontier}</td><td>${row.pgr}</td><td>${JSON.stringify(row.timings_ms || {})}</td><td>${escapeHtml(row.query_preview)}</td></tr>`
        )
        .join("")}
      </tbody></table></div>
    `;
  } catch (e) {
    box.innerHTML = `<p class="err">${escapeHtml(String(e))}</p>`;
  }
}

document.getElementById("send-btn").addEventListener("click", sendChat);
document.getElementById("refresh-stats").addEventListener("click", refreshStats);
document.getElementById("clear-stats").addEventListener("click", clearStats);
document.getElementById("new-session").addEventListener("click", newSession);
document.getElementById("golden-route-btn").addEventListener("click", runGoldenRouteOnly);
document.getElementById("golden-e2e-btn").addEventListener("click", runGoldenE2E);

ensureSessionId();
loadInfo();
loadGoldenMeta();
