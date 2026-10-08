const $ = (s) => document.querySelector(s);
let CFG = null;
let selected = null;      // run_id shown in the detail pane
let timer = null;

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status} ${r.statusText}`);
  return body;
}

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (err ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), err ? 12000 : 4000);
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("ja-JP", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

// ---- 1. checks ---------------------------------------------------------------
async function loadChecks() {
  const tb = $("#checks tbody");
  tb.innerHTML = `<tr><td class="muted">確認中…（初回は CLI のダウンロードで1分ほどかかります）</td></tr>`;
  try {
    const rows = await api("/api/checks");
    tb.innerHTML = rows.map((c) =>
      `<tr><td>${c.ok ? "✅" : "⚠️"}</td><td class="name">${escapeHtml(c.name)}</td><td>${escapeHtml(c.detail)}</td></tr>`).join("");
  } catch (e) {
    tb.innerHTML = `<tr><td>❌</td><td colspan="2">${escapeHtml(e.message)}</td></tr>`;
  }
}

// ---- 2/3. job buttons ----------------------------------------------------------
function renderButtons() {
  const prep = CFG.jobs.filter((j) => j.nodes === 0);
  const train = CFG.jobs.filter((j) => j.nodes > 0);
  const btn = (j) => `<button data-kind="${j.kind}" ${j.nodes === 0 ? 'class="secondary"' : ""}>${j.label}` +
    (j.nodes ? `<span class="tag">${j.nodes === 1 ? "8×H100" : "8×H100 ×2"}</span>` : "") + `</button>`;
  $("#prep-buttons").innerHTML = prep.map(btn).join("");
  $("#train-buttons").innerHTML = train.map(btn).join("");
  document.querySelectorAll("button[data-kind]").forEach((b) => b.addEventListener("click", () => submit(b)));
}

async function submit(button) {
  const job = CFG.jobs.find((j) => j.kind === button.dataset.kind);
  if (job.nodes > 0 && !confirm(`「${job.label}」を投入します（${job.nodes === 1 ? "8×H100 1ノード" : "8×H100 2ノード"}）。よろしいですか？`)) return;
  button.disabled = true;
  const orig = button.innerHTML;
  button.textContent = "投入中…";
  try {
    const r = await api(`/api/runs/${job.kind}`, { method: "POST" });
    toast(`投入しました: ${job.label}（run ${r.run_id}）`);
    selected = r.run_id;
    await loadRuns();
  } catch (e) {
    toast(`投入に失敗しました: ${e.message}`, true);
  } finally {
    button.disabled = false;
    button.innerHTML = orig;
  }
}

// ---- 4. runs + detail ------------------------------------------------------------
async function loadRuns() {
  try {
    const runs = await api("/api/runs");
    $("#runs-updated").textContent = "最終更新 " + new Date().toLocaleTimeString("ja-JP");
    const tb = $("#runs tbody");
    if (!runs.length) { tb.innerHTML = `<tr><td colspan="5" class="muted">まだジョブはありません</td></tr>`; return; }
    tb.innerHTML = runs.map((r) => `
      <tr data-id="${escapeHtml(r.run_id)}" class="${r.run_id === selected ? "sel" : ""}">
        <td>${escapeHtml(r.label || r.run_name)}<div class="muted">run ${r.run_id}${r.submitted_by ? " · " + escapeHtml(r.submitted_by) : ""}</div></td>
        <td><span class="badge ${escapeHtml(r.status)}">${escapeHtml(r.status)}</span></td>
        <td>${fmtTime(r.started_at)}</td>
        <td><a href="${escapeHtml(CFG.host)}/jobs/runs/${escapeHtml(r.run_id)}" target="_blank">ジョブ</a></td>
        <td>${r.terminal ? "" : `<button class="danger" data-cancel="${escapeHtml(r.run_id)}">停止</button>`}</td>
      </tr>`).join("");
    tb.querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", (ev) => {
      if (ev.target.closest("a,button")) return;
      selected = tr.dataset.id; loadRuns(); loadDetail();
    }));
    tb.querySelectorAll("button[data-cancel]").forEach((b) => b.addEventListener("click", () => cancel(b.dataset.cancel)));
    if (selected) loadDetail();
  } catch (e) {
    toast(`ジョブ一覧の取得に失敗しました: ${e.message}`, true);
  }
}

async function loadDetail() {
  if (!selected) return;
  $("#detail").classList.remove("hidden");
  try {
    const d = await api(`/api/runs/${selected}`);
    $("#detail-title").textContent = `${d.experiment_name || ""}（run ${selected}）`;
    const st = $("#detail-status");
    st.textContent = d.status; st.className = "badge " + d.status;
    const links = [];
    if (d.dashboard_url) links.push(`<a href="${escapeHtml(d.dashboard_url)}" target="_blank">ジョブ実行画面 ↗</a>`);
    if (d.mlflow_url) links.push(`<a href="${escapeHtml(d.mlflow_url)}" target="_blank">MLflow ↗</a>`);
    if (d.duration_seconds != null) links.push(`<span class="muted">経過 ${Math.floor(d.duration_seconds / 60)}分${d.duration_seconds % 60}秒</span>`);
    $("#detail-links").innerHTML = links.join("");
    const lg = await api(`/api/runs/${selected}/logs?n=300`);
    const pre = $("#logs");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    pre.textContent = lg.text || "（まだログがありません。GPU ノードの起動を待っています）";
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  } catch (e) {
    $("#logs").textContent = `取得に失敗しました: ${e.message}`;
  }
}

async function cancel(runId) {
  if (!confirm(`run ${runId} を停止します。よろしいですか？`)) return;
  try {
    await api(`/api/runs/${runId}/cancel`, { method: "POST" });
    toast(`停止を依頼しました（run ${runId}）`);
    loadRuns();
  } catch (e) { toast(`停止に失敗しました: ${e.message}`, true); }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function main() {
  CFG = await api("/api/config");
  $("#who").innerHTML = `ログイン中: ${escapeHtml(CFG.user || "不明")}<br>実行 ID（SP）: ${escapeHtml(CFG.service_principal)}`;
  $("#cfg").textContent = JSON.stringify({ image: CFG.image, volume: CFG.volume, mlflow: CFG.experiment_dir }, null, 2);
  renderButtons();
  $("#btn-check").addEventListener("click", loadChecks);
  $("#btn-refresh").addEventListener("click", loadRuns);
  $("#btn-volume").addEventListener("click", async () => {
    try { await api("/api/volume", { method: "POST" }); toast("Volume を作成しました"); loadChecks(); }
    catch (e) { toast(`Volume の作成に失敗しました: ${e.message}`, true); }
  });
  loadChecks();
  // Poll sequentially (log tail takes a few seconds), so requests never pile up.
  const loop = async () => {
    if ($("#auto").checked) await loadRuns();
    timer = setTimeout(loop, 5000);
  };
  loop();
}
main();
