/* =========================================================================
   CineAgent front-end (plain JavaScript, no framework)

   Part 1 - NL2SQL view:        runNL2SQL()  -> POST /api/nl2sql
   Part 2 - Agent & Router view: runAgent()   -> POST /api/ask
   Shared: tables, guard checks, schema, prompt editor
   ========================================================================= */

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = (v) => (typeof v === "string" ? v : JSON.stringify(v, null, 2));

const GUARD_PRESETS = {
  "✓ valid join": "SELECT m.name, d.first_name || ' ' || d.last_name AS director FROM movies m JOIN movies_directors md ON m.id = md.movie_id JOIN directors d ON d.id = md.director_id LIMIT 10",
  "✗ DROP TABLE": "DROP TABLE movies",
  "✗ two statements": "SELECT * FROM movies; DELETE FROM movies",
  "✗ hidden PRAGMA": "SELECT * FROM pragma_table_info('movies')",
  "✗ unknown column": "SELECT title FROM movies",
  "✗ UPDATE": "UPDATE movies SET rank = 10",
};

const PROMPT_INFO = {
  router_prompt: ["Coordinator", "Agent mode · chooses database vs document"],
  sql_prompt: ["NL → SQL", "{schema} is filled automatically"],
  sql_repair_prompt: ["SQL repair", "{question} {sql} {error} {schema}"],
  db_answer_prompt: ["Database answer", "{sql} {result} · can call calculator in Agent mode"],
  doc_answer_prompt: ["Document answer", "{context} · can call calculator"],
};

const AGENT_COLOR = { Coordinator: "badge-indigo", "Database Agent": "badge-cyan", "Document Agent": "badge-emerald" };

/* ---------------- Helpers ---------------- */
async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  return res.json();
}

const setSpent = (usd) => ($("spent-badge").textContent = `💰 spent $${Number(usd || 0).toFixed(4)}`);

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.classList.toggle("loading", busy);
  button.querySelector(".btn-text").textContent = busy ? "Working..." : label;
  document.querySelector(".system-status").classList.toggle("busy", busy);
  $("status-label").textContent = busy ? "Running" : "Ready";
}

function tableHTML(columns, rows) {
  if (!rows || !rows.length) return `<p class="muted">Query returned no rows.</p>`;
  return `<div class="table-wrap"><table>
    <thead><tr>${columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((row) => `<tr>${row.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

function checksHTML(checks) {
  return `<div class="checks">${(checks || []).map((c) =>
    `<div class="check ${c.ok ? "" : "bad"}">${c.ok ? "✅" : "⛔"}<b>${esc(c.name)}</b><span class="detail">${esc(c.detail)}</span></div>`
  ).join("")}</div>`;
}

// Generated SQL -> guard checks, one block per attempt (a repair shows as attempt 2)
function sqlAttemptsHTML(trace) {
  let html = "", attempt = 0;
  trace.filter((s) => s.agent === "Database Agent").forEach((s) => {
    if (s.kind === "llm" && s.title.includes("SQL")) {
      attempt++;
      html += `<div class="sub">Attempt ${attempt} · ${esc(s.title)}</div><p class="explain">${esc(s.output.explanation)}</p>`;
    } else if (s.kind === "sql") {
      html += `<div class="attempt ${s.ok ? "" : "bad"}"><pre class="code">${esc(s.input)}</pre>${checksHTML(s.output.checks)}
               ${s.output.error ? `<div class="check bad"><b>Error</b><span class="detail">${esc(s.output.error)}</span></div>` : ""}</div>`;
    } else if (s.title.startsWith("Refused")) {
      html += `<div class="check bad">⛔<b>Refused</b><span class="detail">${esc(s.output)}</span></div>`;
    }
  });
  return html || `<p class="muted">No SQL was generated.</p>`;
}

function chipsHTML(items) {
  return items.map((c) => `<span class="chip">${c}</span>`).join("");
}

// Light up flow-diagram nodes inside one view
function lightUp(view, names) {
  view.querySelectorAll(".flow-node").forEach((n) => n.classList.remove("on", "fail"));
  names.forEach(([name, ok]) => view.querySelector(`[data-node="${name}"]`)?.classList.add(ok ? "on" : "fail"));
}

/* =========================================================================
   PART 1 - NL2SQL
   ========================================================================= */
async function runNL2SQL() {
  const question = $("sql-question").value.trim();
  if (!question) return $("sql-question").focus();
  setBusy($("sql-run"), true, "Generate & run SQL");
  try {
    const r = await api("/api/nl2sql", { question });
    if (r.error) return alert(r.error);
    renderNL2SQL(r);
    setSpent(r.spent_usd);
    showTab("sql-result");
  } catch (e) {
    alert("Request failed: " + e);
  } finally {
    setBusy($("sql-run"), false, "Generate & run SQL");
  }
}

function renderNL2SQL(r) {
  $("sql-empty").classList.add("hidden");
  $("sql-box").classList.remove("hidden");
  $("sql-q").textContent = r.question;
  $("sql-answer").textContent = r.answer;

  const attempts = r.trace.filter((s) => s.kind === "sql").length;
  $("sql-chips").innerHTML = chipsHTML([
    `🧠 ${attempts > 1 ? "SQL repaired once" : "SQL on first try"}`,
    `⏱ ${(r.summary.ms / 1000).toFixed(1)} s`,
    `💰 $${r.summary.cost.toFixed(5)}`,
  ]);
  $("sql-attempts").innerHTML = sqlAttemptsHTML(r.trace);

  const ok = r.sql && !r.sql.error;
  $("sql-rowcount").textContent = ok ? `${r.sql.row_count} rows` : "no result";
  $("sql-table").innerHTML = ok ? tableHTML(r.sql.columns, r.sql.rows) : `<p class="muted">Nothing was executed.</p>`;

  const sqlSteps = r.trace.filter((s) => s.kind === "sql");
  lightUp($("view-sql"), [
    ["question", true],
    ["generate", r.trace.some((s) => s.title === "Generate SQL")],
    ...sqlSteps.slice(0, 1).map((s) => ["guard", s.ok]),
    ...(sqlSteps.length > 1 ? [["repair", sqlSteps[1].ok]] : []),
    ...(ok ? [["result", true]] : []),
  ]);
}

/* ---------------- Guard lab (no LLM) ---------------- */
async function runGuard() {
  const r = await api("/api/sql/check", { sql: $("guard-sql").value });
  $("guard-out").innerHTML = checksHTML(r.checks) +
    (r.ok ? `<div class="sub">Result · ${r.row_count} rows</div>` + tableHTML(r.columns, r.rows)
          : `<div class="check bad"><b>Blocked</b><span class="detail">${esc(r.error)}</span></div>`);
}

/* =========================================================================
   PART 2 - AGENT & ROUTER
   ========================================================================= */
let agentMode = "auto";

async function runAgent() {
  const question = $("agent-question").value.trim();
  if (!question) return $("agent-question").focus();
  setBusy($("agent-run"), true, "Run Agent");
  try {
    const r = await api("/api/ask", { question, agent: agentMode });
    if (r.error) return alert(r.error);
    renderAgentAnswer(r);
    renderAgentFlow(r.trace);
    renderRouter(r.route);
    renderAgentSQL(r);
    renderChunks(r.chunks);
    renderTools(r.trace);
    renderTrace(r);
    setSpent(r.spent_usd);
    showTab("agent-answer");
  } catch (e) {
    alert("Request failed: " + e);
  } finally {
    setBusy($("agent-run"), false, "Run Agent");
  }
}

function renderAgentAnswer(r) {
  $("agent-empty").classList.add("hidden");
  $("agent-box").classList.remove("hidden");
  $("agent-q").textContent = r.question;
  $("agent-answer").textContent = r.answer;
  $("agent-who").textContent = r.route.agent === "database" ? "🗄 Database Agent" : "📄 Document Agent";

  const tools = r.summary.tools.length ? r.summary.tools.map((t) => "🧮 " + t).join(", ") : "no tools";
  $("agent-chips").innerHTML = chipsHTML([
    `🧭 routed to <b>${esc(r.route.agent)}</b>`, tools, `${r.trace.length} steps`,
    `⏱ ${(r.summary.ms / 1000).toFixed(1)} s`, `💰 $${r.summary.cost.toFixed(5)}`,
  ]);
  // Compact list: who ran what
  $("agent-mini-trace").innerHTML = r.trace.map((s) =>
    `<div class="mini-step ${s.ok ? "" : "bad"}"><span class="step-n">${s.n}</span>
      <span class="badge ${AGENT_COLOR[s.agent] || "badge-amber"}">${esc(s.agent)}</span>
      <span class="${s.kind === "tool" ? "tool-name" : ""}">${esc(s.title)}${s.kind === "tool" ? ` (${esc(s.input)} = ${esc(s.output)})` : ""}</span></div>`
  ).join("");
}

function renderAgentFlow(trace) {
  const nodes = [["question", true]];
  trace.forEach((s) => {
    if (s.kind === "router") nodes.push(["router", true]);
    if (s.agent === "Database Agent") nodes.push(["database", true]);
    if (s.agent === "Document Agent") nodes.push(["document", true]);
    if (s.kind === "sql") nodes.push(["sql", s.ok]);
    if (s.kind === "retrieval") nodes.push(["retrieval", true]);
    if (s.kind === "tool") nodes.push(["tool", s.ok]);
    if (s.kind === "answer") nodes.push(["answer", s.ok]);
  });
  lightUp($("view-agent"), nodes);
}

function renderRouter(d) {
  const yesNo = (v) => (v === null || v === undefined ? "—" : v ? "yes" : "no");
  $("router-out").innerHTML = `
    <dl class="kv">
      <dt>Selected sub-agent</dt><dd><span class="badge badge-emerald">${esc(d.agent)}</span></dd>
      <dt>Needs calculation</dt><dd>${yesNo(d.needs_calculation)}</dd>
      <dt>Write request</dt><dd>${yesNo(d.is_write_request)}</dd>
      <dt>Reason</dt><dd>${esc(d.reason)}</dd>
    </dl>
    <div class="sub">Raw router output</div>
    <pre class="code">${esc(pretty(d))}</pre>`;
}

function renderAgentSQL(r) {
  if (!r.trace.some((s) => s.agent === "Database Agent")) {
    $("agent-sql-out").innerHTML = `<span class="muted">The Database Agent did not run in the last request.</span>`;
    return;
  }
  const ok = r.sql && !r.sql.error;
  $("agent-sql-out").innerHTML = sqlAttemptsHTML(r.trace) +
    (ok ? `<div class="sub">Result · ${r.sql.row_count} rows</div>` + tableHTML(r.sql.columns, r.sql.rows) : "");
}

function renderChunks(chunks) {
  if (!chunks) { $("chunks-out").innerHTML = `<span class="muted">The Document Agent did not run in the last request.</span>`; return; }
  $("chunks-out").innerHTML = chunks.map((c) => `
    <div class="chunk">
      <div class="chunk-head">
        <span class="badge badge-emerald">page ${c.page}</span>
        <span class="badge badge-indigo">${esc(c.id)}</span>
        <div class="score-bar"><div style="width:${Math.max(0, c.score) * 100}%"></div></div>
        <span class="mono">cos ${c.score.toFixed(3)}</span>
      </div>
      <div class="chunk-text">${esc(c.text)}</div>
    </div>`).join("");
}

function renderTools(trace) {
  const calls = trace.filter((s) => s.kind === "tool");
  $("tools-out").innerHTML = calls.length
    ? calls.map((s) => `<div class="tool-call"><span class="badge badge-amber">${esc(s.agent)}</span>
         calculator(<b>${esc(s.input)}</b>) → <span class="res">${esc(s.output)}</span></div>`).join("")
    : `<span class="muted">No tool was called in the last request.</span>`;
}

function renderTrace(r) {
  $("trace-sum").textContent = `${r.trace.length} steps · ${r.summary.agents.join(" + ")}`;
  $("trace-out").innerHTML = r.trace.map((s) => `
    <div class="step ${s.ok ? "" : "bad"}">
      <div class="step-head" onclick="this.parentElement.classList.toggle('open')">
        <span class="step-n">${s.n}</span>
        <span class="badge ${AGENT_COLOR[s.agent] || "badge-amber"}">${esc(s.agent)}</span>
        <span class="badge ${s.kind === "tool" ? "badge-amber" : "badge-indigo"}">${esc(s.kind)}</span>
        <span class="step-title">${s.ok ? "" : "✗ "}${esc(s.title)}</span>
        <span class="step-meta">${s.ms} ms · $${s.cost.toFixed(5)}</span>
      </div>
      <div class="step-body">
        <div><div class="sub">Input</div><pre class="code">${esc(pretty(s.input ?? "—"))}</pre></div>
        <div><div class="sub">Output</div><pre class="code">${esc(pretty(s.output ?? "—"))}</pre></div>
      </div>
    </div>`).join("");
}

/* =========================================================================
   Shared data: status, schema, documents, logs
   ========================================================================= */
async function loadStatus() {
  const s = await api("/api/status");
  $("model-tags").innerHTML =
    `<span class="badge badge-indigo">LLM: ${esc(s.models.llm)}</span>` +
    `<span class="badge badge-cyan">Emb: ${esc(s.models.embedding)}</span>`;
  setSpent(s.spent_usd);
}

async function loadSchema() {
  const s = await api("/api/schema");
  $("sql-tables-mini").innerHTML = s.tables.map((t) =>
    `<div class="mini-table"><span class="mono">${t.name}</span><span class="muted">${t.rows} rows</span></div>`).join("");
  $("schema-tables").innerHTML = s.tables.map((t) => `
    <div class="tbl">
      <div class="tbl-name"><span>${t.name}</span><span class="badge badge-indigo">${t.rows} rows</span></div>
      <div class="tbl-about">${esc(t.about)}</div>
      ${t.columns.map((c) => `<div class="tbl-col"><span>${c.name}</span><span>${esc(c.about)}</span></div>`).join("")}
    </div>`).join("");
  $("schema-context").textContent = s.context;
}

async function loadDocs() {
  const chunks = await api("/api/chunks");
  $("docs-count").textContent = `${chunks.length} chunks`;
  $("docs-out").innerHTML = chunks.map((c) => `
    <div class="chunk"><div class="chunk-head"><span class="badge badge-emerald">page ${c.page}</span>
    <span class="badge badge-indigo">${c.id}</span></div><div class="chunk-text">${esc(c.text)}</div></div>`).join("");
}

async function loadLogs() {
  const logs = await api("/api/logs");
  $("logs-out").innerHTML = logs.length ? `<div class="table-wrap"><table>
    <thead><tr><th>Time</th><th>Question</th><th>Route</th><th>Sub-agents</th><th>Tools</th><th>Steps</th><th>Cost</th></tr></thead>
    <tbody>${logs.map((l) => `<tr>
      <td class="mono">${esc(l.time.replace("T", " "))}</td><td>${esc(l.question)}</td>
      <td><span class="badge badge-emerald">${esc(l.route?.agent)}</span></td>
      <td>${esc(l.agents.join(", "))}</td><td>${esc(l.tools.join(", ") || "—")}</td>
      <td>${l.steps.length}</td><td class="mono">$${l.cost.toFixed(5)}</td></tr>`).join("")}</tbody>
  </table></div>` : `<p class="muted">No requests yet.</p>`;
}

/* ---------------- Prompt editor ---------------- */
let prompts = {}, activePrompt = "sql_prompt";

async function openPrompts() {
  prompts = (await api("/api/prompts")).prompts;
  $("prompt-text").value = "";
  $("prompt-list").innerHTML = Object.keys(prompts).map((k) =>
    `<button class="prompt-item" data-key="${k}">${PROMPT_INFO[k]?.[0] || k}<small>${k}</small></button>`).join("");
  document.querySelectorAll(".prompt-item").forEach((b) => (b.onclick = () => selectPrompt(b.dataset.key)));
  selectPrompt(activePrompt);
  $("modal").classList.remove("hidden");
}

function selectPrompt(key) {
  if ($("prompt-text").value) prompts[activePrompt] = $("prompt-text").value;   // keep unsaved edits
  activePrompt = key;
  document.querySelectorAll(".prompt-item").forEach((b) => b.classList.toggle("active", b.dataset.key === key));
  $("prompt-text").value = prompts[key];
  $("prompt-hint").innerHTML = `<b>${PROMPT_INFO[key]?.[0] || key}</b> · <code>${esc(PROMPT_INFO[key]?.[1] || "")}</code>`;
}

async function savePrompts() {
  prompts[activePrompt] = $("prompt-text").value;
  await api("/api/prompts", { prompts });
  $("modal").classList.add("hidden");
}

async function resetPrompts() {
  prompts = (await api("/api/prompts/reset", {})).prompts;
  $("prompt-text").value = prompts[activePrompt];
}

/* ---------------- Navigation ---------------- */
function showMode(mode) {
  document.body.className = "mode-" + mode;
  document.querySelectorAll(".mode-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + mode));
}

function showTab(name) {
  const view = document.querySelector(`[data-tab="${name}"]`).closest(".view");
  view.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  view.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + name));
  if (name === "agent-logs") loadLogs();
}

function init() {
  document.querySelectorAll(".mode-btn").forEach((b) => (b.onclick = () => showMode(b.dataset.mode)));
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));

  // Part 1
  $("sql-run").onclick = runNL2SQL;
  $("guard-run").onclick = runGuard;
  $("guard-presets").innerHTML = Object.keys(GUARD_PRESETS).map((k) => `<button class="btn btn-sm">${k}</button>`).join("");
  $("guard-presets").querySelectorAll("button").forEach((b) => (b.onclick = () => { $("guard-sql").value = GUARD_PRESETS[b.textContent]; runGuard(); }));

  // Part 2
  $("agent-run").onclick = runAgent;
  $("logs-refresh").onclick = loadLogs;
  document.querySelectorAll("#agent-mode button").forEach((b) => (b.onclick = () => {
    agentMode = b.dataset.agent;
    document.querySelectorAll("#agent-mode button").forEach((x) => x.classList.toggle("active", x === b));
  }));

  // Enter = run, Shift+Enter = new line
  [["sql-question", runNL2SQL], ["agent-question", runAgent]].forEach(([id, fn]) =>
    $(id).addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); fn(); } }));

  // Prompts
  $("btn-prompts").onclick = openPrompts;
  $("modal-close").onclick = () => $("modal").classList.add("hidden");
  $("prompts-save").onclick = savePrompts;
  $("prompts-reset").onclick = resetPrompts;

  loadStatus();
  loadSchema();
  loadDocs();
}

init();
