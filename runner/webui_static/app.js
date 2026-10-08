"use strict";

const ui = {
  state: null,
  filter: "all",
  search: "",
  sort: { key: "priority", direction: 1 },
  loading: false,
  openTask: null,
  timer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
}[c]));
const num = (value, digits = 2) => Number(value || 0).toFixed(digits);
const signed = (value, digits = 2) => `${Number(value) >= 0 ? "+" : ""}${num(value, digits)}`;

function duration(seconds) {
  if (seconds == null) return "—";
  seconds = Math.max(0, Number(seconds));
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${minutes}m`;
}

function relative(timestamp) {
  if (!timestamp) return "never";
  const seconds = Math.max(0, (Date.now() - new Date(timestamp).getTime()) / 1000);
  if (seconds < 5) return "just now";
  return `${duration(seconds)} ago`;
}

function modelClass(model) {
  const value = String(model || "unknown").toLowerCase();

  return "codex";
}

function deltaClass(value) {
  if (value == null) return "delta-zero";
  return Number(value) > .0001 ? "delta-positive" : Number(value) < -.0001 ? "delta-negative" : "delta-zero";
}

function setConnection(status, message) {
  const pill = $("#connection-pill");
  pill.className = `connection-pill ${status}`;
  $("span", pill).textContent = message || status.toUpperCase();
}

function renderSummary(data) {
  const s = data.summary;
  $("#metric-delta").textContent = signed(s.banked_delta);
  $("#metric-workers").textContent = s.workers;
  $("#metric-worker-cap").textContent = s.worker_target;
  $("#metric-deciders").textContent = s.deciders;
  $("#metric-decider-wait").textContent = s.decider_wait;
  $("#metric-queue").textContent = s.queue;
  $("#cooling-count").textContent = `${s.cooling} COOLING`;
  $("#run-count").textContent = `${s.live_runs} RUN${s.live_runs === 1 ? "" : "S"}`;
  $("#helper-count").textContent = `${s.analysts + s.price_searches} HELPERS`;
  const utilization = s.worker_target ? Math.round(s.workers / s.worker_target * 100) : 0;
  $("#worker-capacity").style.width = `${Math.min(100, utilization)}%`;
  $("#worker-utilization").textContent = s.worker_target ? `${utilization}% utilized` : "fleet idle";
  $("#pin-sha").textContent = data.pin_sha || "unknown";
  $("#pin-short").textContent = `PIN ${data.pin_sha || "unknown"}`;
  $("#era-name").textContent = data.era?.era || "unknown";
  $("#last-sync").textContent = `SYNC ${relative(data.generated_at).toUpperCase()}`;
  setConnection(data.status, data.status === "live" ? "TELEMETRY LIVE" : data.status.toUpperCase());
  $("#pipe-build").textContent = s.workers;
  $("#pipe-analyst").textContent = s.analysts;
  $("#pipe-price").textContent = s.price_searches;
  $("#pipe-decide").textContent = s.deciders + s.decider_wait;
  $("#pipe-ready").textContent = s.queue;
}

function workerDelta(worker) {
  const signal = worker.delta_signal;
  if (!signal) return `<div class="worker-delta pending">AWAITING PRICE <small>vs latest pin</small></div>`;
  const cls = Number(signal.delta) >= 0 ? "positive" : "negative";
  return `<div class="worker-delta ${cls}">${signed(signal.delta, 3)} Δ <small>${esc(signal.qualification)}</small></div>`;
}

function renderWorkers(workers) {
  const grid = $("#worker-grid");
  if (!workers.length) {
    grid.innerHTML = `<div class="empty-state">
      <div class="empty-glyph"><i></i><i></i><i></i><i></i></div>
      <strong>No build workers active</strong>
      <span>The console remains live; start or resume a wave to populate ownership.</span>
    </div>`;
    return;
  }
  grid.innerHTML = workers.map(worker => {
    const model = worker.model_id || worker.model || "unknown";
    const cls = modelClass(model);
    const elapsed = Number(worker.elapsed_s || 0) + Math.max(0, (Date.now() / 1000) - Number(ui.state.generated_ts || 0));
    const progress = worker.timeout_s ? Math.min(100, elapsed / worker.timeout_s * 100) : 0;
    const strikes = worker.watch_strikes || [];
    const watch = strikes.length ? `<span class="watch-flag">${esc(strikes.at(-1))}</span>` : `<span>watch ${worker.watchdog ? "checked" : "quiet"}</span>`;
    return `<article class="worker-card ${cls}" data-task="${esc(worker.task)}" tabindex="0">
      <div class="worker-top"><span class="model-name">${esc(model)}${worker.benchmark_score ? ` · B${esc(worker.benchmark_score)}` : ""}</span><span class="run-tag">${esc(worker.run_id)}</span></div>
      <div class="worker-task"><strong>${esc(worker.task)}</strong><span>D${esc(worker.difficulty_score ?? "—")} → B${esc(worker.required_capability ?? "—")} · ATT ${esc(worker.attempt ?? "—")}</span></div>
      ${workerDelta(worker)}
      <div class="timebar"><i style="width:${progress.toFixed(1)}%"></i></div>
      <div class="worker-stats"><span>${duration(elapsed)} / ${duration(worker.timeout_s)}</span>${watch}</div>
      <div class="worker-tail">${esc(worker.log_tail || "Agent reasoning in progress…")}</div>
    </article>`;
  }).join("");
}

function renderAllocation(allocation) {
  const workers = allocation?.workers || { shares: {}, active: {}, spawns: {} };
  const deciders = allocation?.deciders || { shares: {}, active: {}, spawns: {}, bands: [] };
  const deciderLive = Object.values(deciders.active || {}).reduce((a, b) => a + Number(b || 0), 0);
  const deciderRouted = Object.values(deciders.spawns || {}).reduce((a, b) => a + Number(b || 0), 0);
  const labels = { codex: "GPT" };
  const mixRows = (group, colors) => Object.entries(group.shares || {}).map(([provider, share]) => {
    const active = group.active?.[provider] || 0;
    const spawns = group.spawns?.[provider] || 0;
    return `<div class="mix-row ${esc(colors[provider] || provider)}">
      <span>${esc(labels[provider] || provider)}</span><div class="mix-track"><i style="width:${Number(share) * 100}%"></i></div>
      <b>${Math.round(Number(share) * 100)}%</b><em>${active} live · ${spawns} routed</em>
    </div>`;
  }).join("");
  const buildBands = (workers.bands || []).map(band => `<span class="build-band"><i>${esc(band.range)}</i><b>${esc(band.gpt)}</b><em>B${esc(band.target)}</em></span>`).join("");
  const bands = (deciders.bands || []).map(band => `<div class="reasoning-band">
    <span>${esc(band.range)} → B${esc(band.target)}</span><b>${esc(band.gpt)}</b>
  </div>`).join("");
  $("#allocation-deck").innerHTML = `
    <section class="allocation-block">
      <div class="allocation-head"><span>BUILD PROVIDER MIX</span><b>weighted fair</b></div>
      ${mixRows(workers, { codex: "codex" })}
      <div class="build-bands">${buildBands}</div>
    </section>
    <section class="allocation-block decider-policy">
      <div class="allocation-head"><span>DECIDER REASONING</span><b>GPT · ${deciderLive} live / ${deciderRouted} routed</b></div>
      <div class="reasoning-head"><span>SCORE</span><b>GPT ROUTE</b></div>${bands}
    </section>`;
}

function renderRuns(runs) {
  const strip = $("#run-strip");
  if (!runs.length) {
    strip.innerHTML = `<div class="run-card"><i></i><div><strong>NO RUN RECORD</strong><span>waiting for heartbeat</span></div><b>—</b></div>`;
    return;
  }
  strip.innerHTML = runs.map(run => {
    const counts = run.counts || {};
    const workers = (run.workers || []).length || counts.running || counts.workers || 0;
    return `<div class="run-card ${esc(run.health)}"><i></i><div><strong>${esc(run.run_id.toUpperCase())}</strong><span>${esc(run.source)} · ${workers} workers · ${counts.queued || 0} queued</span></div><b>${run.alive ? duration(run.uptime_s) : relative(new Date((run.heartbeat_at || 0) * 1000).toISOString())}</b></div>`;
  }).join("");
}

function renderHealth(checks) {
  const healthy = checks.filter(check => check.status === "ok").length;
  const score = checks.length ? Math.round(healthy / checks.length * 100) : 0;
  const scoreEl = $("#health-score");
  scoreEl.textContent = `${score}%`;
  const worst = checks.some(c => c.status === "error") ? "var(--red)" : checks.some(c => ["warn", "paused"].includes(c.status)) ? "var(--yellow)" : "var(--green)";
  scoreEl.style.borderColor = worst; scoreEl.style.color = worst;
  $("#health-grid").innerHTML = checks.map(check => `<div class="health-check ${esc(check.status)}" title="${esc(check.detail)}">
    <div class="check-top"><i></i><b>${esc(check.label)}</b></div><span>${esc(check.detail)}</span>
  </div>`).join("");
}

function renderAttention(items) {
  $("#attention-count").textContent = items.length;
  $("#attention-list").innerHTML = items.length ? items.map(item => `<div class="attention-item ${esc(item.severity)}" ${item.task ? `data-task="${esc(item.task)}"` : ""}>
    <strong>${esc(item.title)}</strong><span>${esc(item.detail)}</span>
  </div>`).join("") : `<div class="attention-clear">No operator intervention needed.</div>`;
}

function taskDelta(task) {
  if (!task.delta) return { value: null, html: `<span class="delta-zero">—</span>` };
  const value = Number(task.delta.delta);
  const qualifier = task.delta.qualification || task.delta.verdict || task.delta.source;
  return { value, html: `<span class="${deltaClass(value)}">${signed(value, 3)}</span><small>${esc(qualifier)}</small>` };
}

function filteredTasks() {
  let tasks = [...(ui.state?.tasks || [])];
  const query = ui.search.trim().toLowerCase();
  if (query) tasks = tasks.filter(task => [task.task, task.arc_id, task.holder, task.model_id, task.family, task.lane].some(value => String(value || "").toLowerCase().includes(query)));
  if (ui.filter === "active") tasks = tasks.filter(task => ["building", "analyst", "price-search", "decider", "decider-queued"].includes(task.state));
  if (ui.filter === "positive") tasks = tasks.filter(task => task.delta && Number(task.delta.delta) > 0);
  if (ui.filter === "attention") tasks = tasks.filter(task => task.state === "cooling" || (task.delta && (Number(task.delta.delta) < 0 || ["wrong", "rejected", "PHANTOM"].includes(task.delta.qualification || task.delta.verdict))));
  const { key, direction } = ui.sort;
  const getters = {
    priority: t => t.priority ?? 9999,
    task: t => t.task,
    state: t => t.state,
    holder: t => t.holder || "zzz",
    delta: t => t.delta?.delta ?? -999,
    pin: t => t.pin_points ?? -999,
    attempts: t => t.clean_attempts,
    difficulty: t => t.difficulty_score ?? -1,
  };
  tasks.sort((a, b) => {
    const av = getters[key](a), bv = getters[key](b);
    return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * direction;
  });
  return tasks;
}

function renderTasks() {
  const tasks = filteredTasks();
  $("#task-count").textContent = `${tasks.length} task${tasks.length === 1 ? "" : "s"}`;
  $("#task-table").innerHTML = tasks.map(task => {
    const delta = taskDelta(task);
    const holder = task.model_id || task.holder;
    return `<tr data-task="${esc(task.task)}">
      <td class="priority-cell numeric">#${esc(task.priority)}</td>
      <td class="task-cell"><strong>${esc(task.task)}</strong><span>${esc(task.arc_id || "no ARC mapping")} · lane ${esc(task.lane)}</span></td>
      <td><span class="state-pill state-${esc(task.state)}">${esc(task.state.replaceAll("-", " "))}</span></td>
      <td class="holder-cell">${holder ? `<b>${esc(holder)}</b>` : `<span>unassigned</span>`}</td>
      <td class="numeric difficulty-cell"><b>${esc(task.difficulty_score ?? "—")}</b><small>${esc(task.difficulty || "")}</small></td>
      <td class="numeric delta-cell">${delta.html}</td>
      <td class="numeric">${task.pin_points == null ? "—" : num(task.pin_points, 3)}</td>
      <td class="numeric attempt-count">${task.clean_attempts}<small> / ${task.historical_attempts}</small></td>
    </tr>`;
  }).join("") || `<tr><td colspan="8"><div class="attention-clear">No tasks match this view.</div></td></tr>`;
}

function renderActivity(events) {
  $("#activity-list").innerHTML = events.length ? events.map(event => {
    const message = esc(event.message).replace(/task\d{3}/g, match => `<span class="activity-task">${match}</span>`);
    return `<div class="activity-item ${esc(event.severity)}" ${event.task ? `data-task="${esc(event.task)}"` : ""}>
      <i class="activity-dot"></i><div class="activity-time">${relative(event.time).toUpperCase()} · ${esc(event.run)}</div><div class="activity-message">${message}</div>
    </div>`;
  }).join("") : `<div class="attention-clear">No clean-era activity recorded yet.</div>`;
}

function render(data) {
  ui.state = data;
  renderSummary(data);
  renderWorkers(data.workers);
  renderAllocation(data.allocation);
  renderRuns(data.runs);
  renderHealth(data.health);
  renderAttention(data.attention);
  renderTasks();
  renderActivity(data.activity);
  const linkedTask = location.hash.match(/^#(task\d{3})$/i)?.[1]?.toLowerCase();
  if (linkedTask && ui.openTask !== linkedTask) openTask(linkedTask, false);
}

function toast(message) {
  const el = $("#toast");
  el.textContent = message; el.classList.add("show");
  clearTimeout(el._timer); el._timer = setTimeout(() => el.classList.remove("show"), 2200);
}

async function refresh(manual = false) {
  if (ui.loading) return;
  ui.loading = true;
  const button = $("#refresh-button");
  if (manual) button.classList.add("spinning");
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    if (manual) toast("Telemetry refreshed");
  } catch (error) {
    setConnection("offline", "CONNECTION LOST");
    $("#last-sync").textContent = `ERROR · ${error.message}`;
  } finally {
    ui.loading = false;
    setTimeout(() => button.classList.remove("spinning"), 350);
  }
}

function attemptRows(attempts) {
  if (!attempts.length) return `<div class="evidence-note">No attempts exist after the scorer-fix cutoff. Older attempts remain archived but are intentionally excluded from this operational view.</div>`;
  return attempts.slice().reverse().map(attempt => `<div class="attempt-row">
    <b>att ${esc(attempt.att ?? attempt.ordinal)}</b><span>${esc(attempt.model)} · ${esc(attempt.outcome)}</span><span>${esc(attempt.fail_reason || "verified completion")}</span><em>${attempt.delta == null ? "—" : signed(attempt.delta, 3)}</em>
  </div>`).join("");
}

async function openTask(task, updateLocation = true) {
  ui.openTask = task;
  if (updateLocation) history.replaceState(null, "", `#${task}`);
  const drawer = $("#task-drawer"), backdrop = $("#drawer-backdrop");
  drawer.classList.add("open"); backdrop.classList.add("open"); drawer.setAttribute("aria-hidden", "false");
  $("#drawer-title").textContent = task;
  $("#drawer-arc").textContent = "Loading evidence…";
  $("#drawer-body").innerHTML = `<div class="attention-clear">Reading task evidence…</div>`;
  try {
    const response = await fetch(`/api/task/${encodeURIComponent(task)}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const t = data.summary;
    $("#drawer-title").textContent = t.task;
    $("#drawer-arc").textContent = `ARC ${t.arc_id || "unknown"} · LANE ${t.lane} · PRIORITY #${t.priority}`;
    const d = t.delta;
    const deltaText = d ? signed(d.delta, 3) : "—";
    const deltaCls = d ? deltaClass(d.delta) : "delta-zero";
    const active = data.worker ? `<div class="evidence-note"><strong>${esc(data.worker.model_id || data.worker.model)}</strong> owns this task now · attempt ${esc(data.worker.attempt)} · ${duration(data.worker.elapsed_s)} elapsed${data.worker.watch_strikes?.length ? ` · watchdog ${esc(data.worker.watch_strikes.at(-1))}` : ""}.</div>` : "";
    const reasons = (t.difficulty_reasons || []).map(reason => `<li>${esc(reason)}</li>`).join("");
    const routes = t.decider_ladder || {};
    const latest = data.latest_log;
    const decision = data.decision;
    $("#drawer-body").innerHTML = `
      <div class="detail-grid">
        <div class="detail-stat"><span>Delta vs pin</span><strong class="${deltaCls}">${deltaText}</strong></div>
        <div class="detail-stat"><span>Pin points</span><strong>${t.pin_points == null ? "—" : num(t.pin_points, 4)}</strong></div>
        <div class="detail-stat"><span>Pin cost</span><strong>${t.pin_cost ?? "—"}</strong></div>
        <div class="detail-stat"><span>State</span><strong>${esc(t.state)}</strong></div>
        <div class="detail-stat"><span>Clean attempts</span><strong>${t.clean_attempts}</strong></div>
        <div class="detail-stat"><span>Difficulty</span><strong>${esc(t.difficulty_score ?? "—")} · ${esc(t.difficulty || "—")}</strong></div>
      </div>
      ${active}
      <section class="detail-section"><h3>Difficulty evidence // clean era</h3>
        <div class="difficulty-explain"><ul>${reasons || "<li>No dynamic hardness signals yet.</li>"}</ul>
          <div class="route-options"><div><span>Required capability B${esc(routes.required_capability || "—")} · GPT 80%</span><b>${esc(routes.gpt?.model || "—")} / ${esc(routes.gpt?.effort || "—")} · B${esc(routes.gpt?.benchmark_score || "—")}</b></div>
</div>
        </div>
      </section>
      <section class="detail-section"><h3>Clean-era attempts <span>(${t.clean_attempts} / ${data.historical_attempts} all-time)</span></h3>${attemptRows(data.attempts)}</section>
      ${decision ? `<section class="detail-section"><h3>Current decider brief</h3><div class="path-label">${esc(decision.path)}</div><pre class="code-block">${esc(decision.text)}</pre></section>` : ""}
      ${latest ? `<section class="detail-section"><h3>Latest worker transcript</h3><div class="path-label">${esc(latest.path)}</div><pre class="code-block">${esc(latest.text || "No worker output captured.")}</pre></section>` : ""}
    `;
  } catch (error) {
    $("#drawer-body").innerHTML = `<div class="evidence-note">Could not load task detail: ${esc(error.message)}</div>`;
  }
}

function closeDrawer() {
  $("#task-drawer").classList.remove("open"); $("#drawer-backdrop").classList.remove("open");
  $("#task-drawer").setAttribute("aria-hidden", "true");
  ui.openTask = null;
  if (location.hash) history.replaceState(null, "", location.pathname + location.search);
}

document.addEventListener("click", event => {
  const taskNode = event.target.closest("[data-task]");
  if (taskNode?.dataset.task) openTask(taskNode.dataset.task);
});
document.addEventListener("keydown", event => {
  if (event.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault(); $("#task-search").focus();
  } else if (event.key.toLowerCase() === "r" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    refresh(true);
  } else if (event.key === "Escape") closeDrawer();
  if ((event.key === "Enter" || event.key === " ") && event.target.classList.contains("worker-card")) openTask(event.target.dataset.task);
});
$("#refresh-button").addEventListener("click", () => refresh(true));
$("#drawer-close").addEventListener("click", closeDrawer);
$("#drawer-backdrop").addEventListener("click", closeDrawer);
$("#task-search").addEventListener("input", event => { ui.search = event.target.value; renderTasks(); });
$("#task-filter").addEventListener("click", event => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  ui.filter = button.dataset.filter;
  $$("button", $("#task-filter")).forEach(b => b.classList.toggle("active", b === button));
  renderTasks();
});
$("thead").addEventListener("click", event => {
  const th = event.target.closest("th[data-sort]");
  if (!th) return;
  if (ui.sort.key === th.dataset.sort) ui.sort.direction *= -1;
  else ui.sort = { key: th.dataset.sort, direction: 1 };
  renderTasks();
});
document.addEventListener("visibilitychange", () => {
  clearInterval(ui.timer);
  ui.timer = setInterval(refresh, document.hidden ? 15000 : 4000);
  if (!document.hidden) refresh();
});

refresh();
ui.timer = setInterval(refresh, 4000);
setInterval(() => {
  if (ui.state) $("#last-sync").textContent = `SYNC ${relative(ui.state.generated_at).toUpperCase()}`;
}, 1000);
