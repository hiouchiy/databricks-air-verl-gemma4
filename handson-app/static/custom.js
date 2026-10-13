// 「自分のデータで学習」: one path, one main button per step.
//   1 which data -> 2 check & convert -> 3 how to score (+ confirmation test) -> 4 train -> 5 watch
// Uses api / toast / escapeHtml / loadRuns / CFG from app.js.
const W = { path: null, cols: [], hasImages: false, name: null, rewardFrom: null, runId: null, experiment: null };
let customInit = false, watchTimer = null;
const DEFAULT_INSTR = "Think step by step, then give the final answer in \\boxed{} (for example \\boxed{42}).";

function initCustom() {
  if (customInit) return;
  customInit = true;
  const parts = CFG.volume.split("/");   // /Volumes/<catalog>/<schema>/<volume>
  $("#g-volume-link").href = `${CFG.host}/explore/data/volumes/${parts[2]}/${parts[3]}/${parts[4]}`;
  $("#g-cli").textContent = `databricks fs cp -r ./mydata dbfs:${CFG.volume}/raw/mydata -p <プロファイル>`;
  $("#c-path").value = `${CFG.volume}/raw/mydata/data.jsonl`;
  $("#c-instr").value = DEFAULT_INSTR;
  document.querySelectorAll('input[name="src"]').forEach((r) => r.addEventListener("change", onSource));
  document.querySelectorAll('input[name="rw"]').forEach((r) => r.addEventListener("change", onRewardChoice));
  $("#c-load").addEventListener("click", () => load($("#c-path").value.trim()));
  $("#c-sample").addEventListener("click", makeSample);
  $("#c-use-existing").addEventListener("click", useExisting);
  ["#c-qcol", "#c-acol", "#c-icol", "#c-instr", "#c-imgdir"].forEach((s) => $(s).addEventListener("change", preview));
  $("#c-convert").addEventListener("click", convert);
  $("#c-rw-go").addEventListener("click", runRewardChoice);
  $("#c-agent").addEventListener("click", runAgent);
  $("#c-approve").addEventListener("click", approve);
  $("#c-train").addEventListener("click", train);
  $("#t-cancel").addEventListener("click", () => W.runId && cancel(W.runId));
  $("#r-step").addEventListener("change", () => loadRollouts(W.experiment, $("#r-step").value));
  $("#r-filter").addEventListener("change", () => loadRollouts(W.experiment, $("#r-step").value));
  onSource();
}

// ---- helpers ----------------------------------------------------------------------------
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const status = (id, html, kind) => { $(id).innerHTML = html ? `<span class="${kind || ""}">${html}</span>` : ""; };
function unlock(n) {           // steps 1..n usable, later ones locked
  for (let i = 2; i <= 5; i++) $(`#w${i}`).classList.toggle("locked", i > n);
  $(`#w${n}`).scrollIntoView({ behavior: "smooth", block: "start" });
}
async function waitRun(runId, onTick) {
  for (;;) {
    const r = await api(`/api/runs/${runId}`);
    onTick && onTick(r);
    if (r.terminal) return r;
    await new Promise((res) => setTimeout(res, 5000));
  }
}
async function pollTask(tid) {
  const log = $("#c-agent-log");
  log.classList.remove("hidden");
  for (;;) {
    const t = await api(`/api/custom/task/${tid}`);
    log.textContent = t.log.join("\n") + (t.status === "running" ? "\n…" : "");
    if (t.status === "done") return t.result;
    if (t.status === "error") throw new Error(t.error);
    await new Promise((r) => setTimeout(r, 2500));
  }
}
function options(sel, cols, chosen, allowEmpty) {
  sel.innerHTML = (allowEmpty ? `<option value="">（使わない）</option>` : "") +
    cols.map((c) => `<option ${c === chosen ? "selected" : ""}>${escapeHtml(c)}</option>`).join("");
}

// ---- 1. which data -------------------------------------------------------------------------
function onSource() {
  const v = document.querySelector('input[name="src"]:checked').value;
  ["own", "sample", "existing"].forEach((k) => $(`#src-${k}`).classList.toggle("hidden", k !== v));
  status("#w1-status", "");
  $("#w2-body").classList.remove("hidden");
  status("#w2-status", "");
  if (!document.querySelector('input[name="rw"]:checked')) { $('input[name="rw"][value="numeric"]').checked = true; onRewardChoice(); }
  if (v === "existing") loadExisting();
}

async function makeSample() {
  const kind = document.querySelector('input[name="sk"]:checked').value;
  $("#c-sample").disabled = true;
  try {
    const r = await post("/api/custom/sample-raw", { kind });
    const done = await waitRun(r.run_id, (x) => status("#w1-status", `⏳ サンプルデータを作成中…（${x.status}）`));
    if (done.status !== "SUCCESS") throw new Error(`作成ジョブが ${done.status} で終わりました`);
    $("#c-name").value = `sample-${kind}`;
    await load(r.path);
  } catch (e) { status("#w1-status", `❌ ${escapeHtml(e.message)}`, "fail"); }
  finally { $("#c-sample").disabled = false; }
}

async function load(path) {
  status("#w1-status", "⏳ 読み込み中…");
  try {
    const r = await post("/api/custom/inspect", { path });
    W.path = path; W.cols = r.columns;
    options($("#c-qcol"), r.columns, r.guess.question_col);
    options($("#c-acol"), r.columns, r.guess.answer_col);
    options($("#c-icol"), r.columns, r.guess.image_col, true);
    if (!$("#c-name").value) $("#c-name").value = path.split("/").slice(-2, -1)[0] || "mydata";
    $("#c-imgdir").value = path.split("/").slice(0, -1).join("/");
    const n = r.n_rows != null ? `${r.n_rows} 問` : `${r.n_rows_at_least} 問以上`;
    status("#w1-status", `✅ ${escapeHtml(path)} を読み込みました（${n}）`, "pass");
    $("#c-loaded").textContent = `${n}を読み込みました。項目：${r.columns.join(" / ")}`;
    if (!r.guess.question_col || !r.guess.answer_col) $("#c-mapping").open = true;
    unlock(2);
    preview();
  } catch (e) { status("#w1-status", `❌ 読み込めませんでした：${escapeHtml(e.message)}`, "fail"); }
}

let DATASETS = [];
async function loadExisting() {
  DATASETS = (await api("/api/custom/datasets")).filter((d) => d.converted);
  $("#c-dataset").innerHTML = DATASETS.length
    ? DATASETS.map((d) => `<option value="${escapeHtml(d.name)}">${escapeHtml(d.name)}${d.has_images ? "（画像あり）" : "（テキスト）"}${d.reward_approved ? " — 採点も決定済み" : ""}</option>`).join("")
    : `<option value="">（まだありません）</option>`;
}

async function useExisting() {
  const d = DATASETS.find((x) => x.name === $("#c-dataset").value);
  if (!d) return;
  W.name = d.name; W.hasImages = d.has_images;
  status("#w1-status", `✅ 「${escapeHtml(d.name)}」を使います（変換済み）`, "pass");
  status("#w2-status", "✅ 変換済みのため、この手順は不要です", "pass");
  $("#w2-body").classList.add("hidden");
  unlock(3);
  if (d.reward_approved) {
    const r = await api(`/api/custom/${encodeURIComponent(d.name)}/reward`);
    document.querySelectorAll('input[name="rw"]').forEach((x) => { x.checked = false; });
    $("#agent-box").classList.add("hidden");
    $("#rw-go-row").classList.add("hidden");
    showReview(r.code, null, "以前に決定した採点関数");
    status("#w3-status", "採点の候補：<b>以前に決定した採点関数</b>（このまま「この採点で決定」を押すか、上で方法を選び直してください）");
    await trial();
  }
}

// ---- 2. check & convert --------------------------------------------------------------------------
const mapping = () => ({ path: W.path, question_col: $("#c-qcol").value, answer_col: $("#c-acol").value,
  image_col: $("#c-icol").value || null, name: $("#c-name").value.trim(), instruction: $("#c-instr").value.trim(),
  image_dir: $("#c-imgdir").value.trim() });

async function preview() {
  if (!W.path) return;
  try {
    const r = await post("/api/custom/preview", mapping());
    $("#c-preview-out").innerHTML = `<table class="small"><tr><th>モデルへの入力（${r.has_images ? "画像＋文章" : "文章"}）</th><th>正解</th>${r.has_images ? "<th>画像</th>" : ""}</tr>` +
      r.rows.map((x) => `<tr><td class="answer">${escapeHtml(x.prompt)}</td><td>${escapeHtml(x.ground_truth)}</td>${r.has_images ? `<td>${
        x.images.map((p, i) => `${x.images_found[i] ? "✅" : "❌ 見つかりません"} ${escapeHtml(p.split("/").pop())}`).join("<br>")}</td>` : ""}</tr>`).join("") + "</table>";
  } catch (e) { $("#c-preview-out").textContent = `表示できません：${e.message}`; }
}

async function convert() {
  const m = mapping();
  $("#c-convert").disabled = true;
  try {
    const r = await post("/api/custom/convert", m);
    W.name = r.name; W.hasImages = !!m.image_col;
    const done = await waitRun(r.run_id, (x) => status("#w2-status", `⏳ 変換中…（${x.status}）`));
    if (done.status !== "SUCCESS") throw new Error(`変換ジョブが ${done.status} で終わりました（下の「すべての実行状況」でログを確認してください）`);
    status("#w2-status", `✅ 変換しました：${escapeHtml(CFG.volume)}/custom/${escapeHtml(r.name)}/（train.parquet / test.parquet）`, "pass");
    unlock(3);
  } catch (e) { status("#w2-status", `❌ ${escapeHtml(e.message)}`, "fail"); }
  finally { $("#c-convert").disabled = false; loadRuns(); }
}

// ---- 3. how to score ---------------------------------------------------------------------------------
function onRewardChoice() {
  const v = document.querySelector('input[name="rw"]:checked').value;
  $("#agent-box").classList.toggle("hidden", v !== "agent");
  $("#rw-go-row").classList.toggle("hidden", v === "agent");
}

async function runRewardChoice() {     // presets: load the function, then the confirmation test
  const v = document.querySelector('input[name="rw"]:checked').value;
  $("#c-rw-go").disabled = true;
  try {
    const r = await api(`/api/custom/reward/preset/${v}`);
    showReview(r.code, null, v === "numeric" ? "数値で比べる（標準）" : "文字が完全に同じなら正解");
    await trial();
  } finally { $("#c-rw-go").disabled = false; }
}

async function runAgent() {
  $("#c-agent").disabled = true;
  try {
    const { task_id } = await post(`/api/custom/${encodeURIComponent(W.name)}/reward/agent`, { criteria: $("#c-criteria").value });
    const res = await pollTask(task_id);
    showReview(res.code, res.report, "AI が作った採点関数");
    await trial();
  } catch (e) { status("#w3-status", `❌ ${escapeHtml(e.message)}`, "fail"); }
  finally { $("#c-agent").disabled = false; }
}

function showReview(code, report, from) {
  W.rewardFrom = from;
  $("#c-review").classList.remove("hidden");
  $("#c-code").value = code || "";
  $("#c-trial-out").innerHTML = "";
  $("#c-tests").innerHTML = report ? `<p>AI が作ったテスト：<span class="${report.failed ? "fail" : "pass"}">${report.passed} / ${report.total} 件合格</span>（Sandbox で実行）
    <details><summary class="muted">テストの中身</summary><table class="small"><tr><th></th><th>期待</th><th>実際</th><th>正解</th><th>答え（末尾）</th><th>ねらい</th></tr>${
    report.results.map((r) => `<tr><td class="${r.ok ? "pass" : "fail"}">${r.ok ? "✔" : "✘"}</td><td>${r.expected}</td><td>${r.got ?? escapeHtml(r.error || "")}</td><td>${escapeHtml(r.ground_truth)}</td><td class="answer">${escapeHtml(r.response_tail)}</td><td>${escapeHtml(r.why)}</td></tr>`).join("")}</table></details></p>` : "";
  status("#w3-status", `採点の候補：<b>${escapeHtml(from)}</b>`);
}

async function trial() {
  try {
    const { task_id } = await post(`/api/custom/${encodeURIComponent(W.name)}/reward/trial`, { code: $("#c-code").value, n: 5 });
    const rows = await pollTask(task_id);
    $("#c-trial-out").innerHTML = `<table class="small"><tr><th>点数</th><th>正解</th><th>AI の答え（末尾）</th></tr>${
      rows.map((r) => `<tr><td class="${r.score === 1 ? "pass" : "fail"}">${r.score ?? escapeHtml(r.error || "")}</td><td>${escapeHtml(r.ground_truth)}</td><td class="answer">${escapeHtml((r.response || "").slice(-300))}</td></tr>`).join("")}</table>`;
    $("#c-agent-log").classList.add("hidden");
  } catch (e) { status("#w3-status", `❌ 確認テストに失敗しました：${escapeHtml(e.message)}`, "fail"); }
}

async function approve() {
  try {
    await post(`/api/custom/${encodeURIComponent(W.name)}/reward/approve`, { code: $("#c-code").value });
    status("#w3-status", `✅ 採点を決定しました（${escapeHtml(W.rewardFrom || "")}）`, "pass");
    unlock(4);
  } catch (e) { status("#w3-status", `❌ ${escapeHtml(e.message)}`, "fail"); }
}

// ---- 4. train -----------------------------------------------------------------------------------------------
async function train() {
  const nodes = Number(document.querySelector('input[name="nodes"]:checked').value);
  if (!confirm(`「${W.name}」で学習を開始します（8×H100 × ${nodes}）。GPU の課金が始まります。よろしいですか？`)) return;
  $("#c-train").disabled = true;
  try {
    const r = await post(`/api/custom/${encodeURIComponent(W.name)}/train`, { nodes });
    W.runId = r.run_id; W.experiment = r.experiment;
    status("#w4-status", `✅ 学習を開始しました（run ${escapeHtml(r.run_id)}）`, "pass");
    unlock(5);
    watch();
    loadRuns();
  } catch (e) { status("#w4-status", `❌ ${escapeHtml(e.message)}`, "fail"); }
  finally { $("#c-train").disabled = false; }
}

// ---- 5. watch ----------------------------------------------------------------------------------------------
async function watch() {
  clearTimeout(watchTimer);
  if (!W.runId) return;
  try {
    const d = await api(`/api/runs/${W.runId}`);
    $("#t-status").textContent = d.status; $("#t-status").className = "badge " + d.status;
    $("#t-info").textContent = d.duration_seconds != null ? `経過 ${Math.floor(d.duration_seconds / 60)}分${d.duration_seconds % 60}秒` : "";
    $("#t-links").innerHTML = [d.dashboard_url && `<a href="${escapeHtml(d.dashboard_url)}" target="_blank">ジョブ ↗</a>`,
      d.mlflow_url && `<a href="${escapeHtml(d.mlflow_url)}" target="_blank">MLflow ↗</a>`].filter(Boolean).join(" ");
    $("#t-cancel").classList.toggle("hidden", !!d.terminal);
    const lg = await api(`/api/runs/${W.runId}/logs?n=60`);
    $("#t-logs").textContent = lg.text || "（GPU ノードの起動を待っています）";
    await loadRollouts(W.experiment, $("#r-step").value);
    if (!d.terminal) watchTimer = setTimeout(watch, 10000);
  } catch (e) { watchTimer = setTimeout(watch, 15000); }
}

// called from the run list ("回答と点数") too
async function showRollouts(experiment) {
  document.querySelector('.tab[data-tab="custom"]').click();
  W.experiment = experiment;
  $("#w5").classList.remove("locked");
  await loadRollouts(experiment, "");
  $("#w5").scrollIntoView({ behavior: "smooth" });
}

async function loadRollouts(experiment, step) {
  if (!experiment) return;
  try {
    const r = await api(`/api/custom/rollouts/${encodeURIComponent(experiment)}${step ? `?step=${step}` : ""}`);
    ["#r-step", "#r-filter"].forEach((s) => $(s).classList.toggle("hidden", !r.steps.length));
    if (!r.steps.length) { $("#r-summary").textContent = "最初の学習ステップが終わると表示されます（開始から約20分）"; $("#r-table").innerHTML = ""; return; }
    $("#r-step").innerHTML = r.steps.map((s) => `<option value="${s}" ${s === r.step ? "selected" : ""}>ステップ ${s}</option>`).join("");
    const f = $("#r-filter").value;
    const rows = r.rows.filter((x) => f === "all" || String(Number(x.score)) === f);
    $("#r-summary").textContent = `${r.n} 件 ／ 平均点 ${r.mean?.toFixed(3)}`;
    $("#r-table").innerHTML = `<tr><th>点数</th><th>正解</th><th>回答の長さ</th><th>モデルの回答（末尾）</th></tr>` + rows.map((x) =>
      `<tr><td class="${Number(x.score) === 1 ? "pass" : "fail"}">${x.score}</td><td>${escapeHtml(x.ground_truth)}</td><td>${x.answer_len}</td><td class="answer">${escapeHtml(x.answer_tail)}</td></tr>`).join("");
  } catch (e) { $("#r-summary").textContent = `取得できません：${e.message}`; }
}
