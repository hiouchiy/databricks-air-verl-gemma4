// 「自分のデータで学習」 tab. Uses api / toast / escapeHtml / loadRuns / CFG from app.js.
let customInit = false;
let currentTests = [];       // test cases from the agent (re-run on approval is not needed; shown for review)
let rollExperiment = null;

function initCustom() {
  if (customInit) return;
  customInit = true;
  $("#c-instr").value = "Think step by step, then give the final answer in \\boxed{} (for example \\boxed{42}).";
  $("#c-path").value = `${CFG.volume}/raw/text/questions.csv`;
  $("#c-inspect").addEventListener("click", inspect);
  $("#c-sample").addEventListener("click", makeSample);
  $("#c-preview").addEventListener("click", preview);
  $("#c-convert").addEventListener("click", convert);
  $("#c-ds-refresh").addEventListener("click", loadDatasets);
  $("#c-dataset").addEventListener("change", onDataset);
  $("#c-use-preset").addEventListener("click", usePreset);
  $("#c-agent").addEventListener("click", runAgent);
  $("#c-trial").addEventListener("click", runTrial);
  $("#c-approve").addEventListener("click", approve);
  $("#c-train-1").addEventListener("click", () => train(1));
  $("#c-train-2").addEventListener("click", () => train(2));
  $("#r-step").addEventListener("change", () => loadRollouts(rollExperiment, $("#r-step").value));
  $("#r-filter").addEventListener("change", () => loadRollouts(rollExperiment, $("#r-step").value));
  api("/api/custom/reward/presets").then((ps) => {
    $("#c-preset").innerHTML = ps.map((p) => `<option value="${escapeHtml(p.id)}">${escapeHtml(p.label)}</option>`).join("");
  });
  loadDatasets();
}

const ds = () => $("#c-dataset").value;
function options(sel, cols, chosen, allowEmpty) {
  sel.innerHTML = (allowEmpty ? `<option value="">（なし）</option>` : "") +
    cols.map((c) => `<option ${c === chosen ? "selected" : ""}>${escapeHtml(c)}</option>`).join("");
}

// ---- A: data -------------------------------------------------------------------------
async function makeSample() {
  try {
    const r = await api("/api/custom/sample-raw", { method: "POST" });
    toast(`サンプルの生データを作成中（run ${r.run_id}・約1分）`);
    $("#c-sample-paths").innerHTML = `完了後、次のパスを指定してください：<br>テキスト：${escapeHtml(CFG.volume)}/raw/text/questions.csv<br>画像：${escapeHtml(CFG.volume)}/raw/image/data.jsonl`;
    loadRuns();
  } catch (e) { toast(`作成に失敗しました: ${e.message}`, true); }
}

async function inspect() {
  try {
    const r = await api("/api/custom/inspect", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#c-path").value.trim() }) });
    options($("#c-qcol"), r.columns, r.guess.question_col);
    options($("#c-acol"), r.columns, r.guess.answer_col);
    options($("#c-icol"), r.columns, r.guess.image_col, true);
    if (!$("#c-name").value) $("#c-name").value = $("#c-path").value.split("/").slice(-2, -1)[0] || "dataset";
    $("#c-imgdir").value = $("#c-path").value.split("/").slice(0, -1).join("/");
    $("#c-head").innerHTML = `<tr>${r.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>` +
      r.rows.map((row) => `<tr>${r.columns.map((c) => `<td>${escapeHtml(String(row[c] ?? "").slice(0, 120))}</td>`).join("")}</tr>`).join("");
    $("#c-inspect-out").classList.remove("hidden");
    $("#c-preview-out").innerHTML = "";
  } catch (e) { toast(`読み込めませんでした: ${e.message}`, true); }
}

function mapping() {
  return { path: $("#c-path").value.trim(), question_col: $("#c-qcol").value, answer_col: $("#c-acol").value,
           image_col: $("#c-icol").value || null, name: $("#c-name").value.trim(),
           instruction: $("#c-instr").value.trim(), image_dir: $("#c-imgdir").value.trim() };
}

async function preview() {
  try {
    const r = await api("/api/custom/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(mapping()) });
    $("#c-preview-out").innerHTML = `<h4>変換後のイメージ（先頭3件）</h4><table class="small"><tr><th>モデルへの入力（prompt）</th><th>正解</th><th>画像</th></tr>` +
      r.rows.map((x) => `<tr><td class="answer">${escapeHtml(x.prompt)}</td><td>${escapeHtml(x.ground_truth)}</td><td>${
        x.images.map((p, i) => `${x.images_found[i] ? "✅" : "❌ 見つかりません"} ${escapeHtml(p)}`).join("<br>") || "なし"}</td></tr>`).join("") + "</table>";
  } catch (e) { toast(`プレビューに失敗しました: ${e.message}`, true); }
}

async function convert() {
  try {
    const r = await api("/api/custom/convert", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(mapping()) });
    toast(`変換ジョブを投入しました（${r.name}・run ${r.run_id}・CPU）`);
    loadRuns(); setTimeout(loadDatasets, 3000);
  } catch (e) { toast(`変換の投入に失敗しました: ${e.message}`, true); }
}

// ---- B: datasets ------------------------------------------------------------------------
let DATASETS = [];
async function loadDatasets() {
  try {
    DATASETS = await api("/api/custom/datasets");
    const prev = ds();
    $("#c-dataset").innerHTML = DATASETS.length ? DATASETS.map((d) => `<option value="${escapeHtml(d.name)}" ${d.name === prev ? "selected" : ""}>${escapeHtml(d.name)}${
      d.has_images ? "（画像あり）" : "（テキスト）"}${d.converted ? "" : " — 変換中/未変換"}${d.reward_approved ? " — 採点関数 承認済み" : ""}</option>`).join("")
      : `<option value="">（まだありません。A で変換してください）</option>`;
    onDataset();
  } catch (e) { toast(`データ一覧の取得に失敗しました: ${e.message}`, true); }
}

async function onDataset() {
  const d = DATASETS.find((x) => x.name === ds());
  if (!d) { $("#c-ds-info").textContent = ""; return; }
  $("#c-ds-info").innerHTML = `元データ: ${escapeHtml(d.input)} ／ 質問: ${escapeHtml(d.question_col)} ／ 正解: ${escapeHtml(d.answer_col)} ／ 画像: ${escapeHtml(d.image_col || "なし")}<br>
    変換先: ${escapeHtml(d.out)} ${d.converted ? "✅" : "⏳（変換ジョブの完了を待ってから「更新」）"} ／ 採点関数: ${d.reward_approved ? "✅ 承認済み" : "未承認"}`;
  if (d.reward_approved) {
    const r = await api(`/api/custom/${encodeURIComponent(d.name)}/reward`);
    showCode(r.code, null);
  }
}

// ---- C: reward -------------------------------------------------------------------------------
function showCode(code, tests, report) {
  $("#c-reward-box").classList.remove("hidden");
  $("#c-code").value = code || "";
  currentTests = tests || [];
  if (report) {
    $("#c-tests").innerHTML = `<h4>エージェントが作ったテスト：<span class="${report.failed ? "fail" : "pass"}">${report.passed} / ${report.total} 件合格</span>（Sandbox で実行）</h4>
      <details><summary class="muted">テストケースを見る</summary><table class="small"><tr><th>結果</th><th>期待</th><th>実際</th><th>正解</th><th>回答（末尾）</th><th>ねらい</th></tr>${
      report.results.map((r) => `<tr><td class="${r.ok ? "pass" : "fail"}">${r.ok ? "✔" : "✘"}</td><td>${r.expected}</td><td>${r.got ?? escapeHtml(r.error || "")}</td><td>${escapeHtml(r.ground_truth)}</td><td class="answer">${escapeHtml(r.response_tail)}</td><td>${escapeHtml(r.why)}</td></tr>`).join("")}</table></details>`;
  } else {
    $("#c-tests").innerHTML = "";
  }
  $("#c-trial-out").innerHTML = "";
}

async function usePreset() {
  const r = await api(`/api/custom/reward/preset/${$("#c-preset").value}`);
  showCode(r.code, null);
  toast("型の採点関数を読み込みました。必要なら編集して、試し採点で確かめてください");
}

async function pollTask(tid, onDone) {
  const log = $("#c-agent-log");
  log.classList.remove("hidden");
  for (;;) {
    const t = await api(`/api/custom/task/${tid}`);
    log.textContent = t.log.join("\n") + (t.status === "running" ? "\n…" : "");
    if (t.status === "done") return onDone(t.result);
    if (t.status === "error") { toast(`失敗しました: ${t.error}`, true); log.textContent += `\n❌ ${t.error}`; return; }
    await new Promise((r) => setTimeout(r, 2500));
  }
}

async function runAgent() {
  if (!ds()) return toast("先に B でデータを選んでください", true);
  const b = $("#c-agent"); b.disabled = true;
  try {
    const { task_id } = await api(`/api/custom/${encodeURIComponent(ds())}/reward/agent`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ criteria: $("#c-criteria").value }) });
    await pollTask(task_id, (res) => {
      showCode(res.code, res.tests, res.report);
      toast(res.all_passed ? `採点関数ができました（テスト ${res.report.passed}/${res.report.total} 件合格）` :
        `テストに通らないケースが残りました（${res.report.failed} 件）。内容を確認してください`, !res.all_passed);
    });
  } catch (e) { toast(`エージェントの実行に失敗しました: ${e.message}`, true); }
  finally { b.disabled = false; }
}

async function runTrial() {
  if (!ds()) return toast("先に B でデータを選んでください", true);
  const b = $("#c-trial"); b.disabled = true;
  try {
    const { task_id } = await api(`/api/custom/${encodeURIComponent(ds())}/reward/trial`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code: $("#c-code").value, n: 5 }) });
    await pollTask(task_id, (rows) => {
      $("#c-trial-out").innerHTML = `<h4>試し採点：実際の問題を Claude に解かせ、この採点関数で採点した結果</h4>
        <p class="muted">点数が妥当か（正しい答えが 1、間違い・答えていないものが 0 になっているか）を目で確かめてください。画像のデータでは、画像も一緒に Claude に渡しています。</p>
        <table class="small"><tr><th>点数</th><th>正解</th><th>Claude の回答（末尾）</th></tr>${
        rows.map((r) => `<tr><td class="${r.score === 1 ? "pass" : "fail"}">${r.score ?? escapeHtml(r.error || "")}</td><td>${escapeHtml(r.ground_truth)}</td><td class="answer">${escapeHtml((r.response || "").slice(-300))}</td></tr>`).join("")}</table>`;
    });
  } catch (e) { toast(`試し採点に失敗しました: ${e.message}`, true); }
  finally { b.disabled = false; }
}

async function approve() {
  if (!ds()) return toast("先に B でデータを選んでください", true);
  if (!confirm("この採点関数を承認して、学習に使います。よろしいですか？")) return;
  try {
    await api(`/api/custom/${encodeURIComponent(ds())}/reward/approve`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code: $("#c-code").value }) });
    toast("採点関数を承認しました"); loadDatasets();
  } catch (e) { toast(`承認に失敗しました: ${e.message}`, true); }
}

// ---- D: train ------------------------------------------------------------------------------------
async function train(nodes) {
  if (!ds()) return toast("先に B でデータを選んでください", true);
  if (!confirm(`「${ds()}」で GRPO を投入します（8×H100 × ${nodes}）。よろしいですか？`)) return;
  try {
    const r = await api(`/api/custom/${encodeURIComponent(ds())}/train`, { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nodes }) });
    toast(`投入しました（run ${r.run_id}）`); selected = r.run_id; loadRuns();
  } catch (e) { toast(`投入に失敗しました: ${e.message}`, true); }
}

// ---- rollouts (answers + scores) ---------------------------------------------------------------------
async function showRollouts(experiment) {
  rollExperiment = experiment;
  $("#rollouts").classList.remove("hidden");
  await loadRollouts(experiment, "");
  $("#rollouts").scrollIntoView({ behavior: "smooth" });
}

async function loadRollouts(experiment, step) {
  try {
    const r = await api(`/api/custom/rollouts/${encodeURIComponent(experiment)}${step ? `?step=${step}` : ""}`);
    if (!r.steps.length) { $("#r-summary").textContent = "まだありません（最初の学習ステップが終わると書き出されます）"; $("#r-table").innerHTML = ""; return; }
    $("#r-step").innerHTML = r.steps.map((s) => `<option value="${s}" ${s === r.step ? "selected" : ""}>ステップ ${s}</option>`).join("");
    const f = $("#r-filter").value;
    const rows = r.rows.filter((x) => f === "all" || String(Number(x.score)) === f);
    $("#r-summary").textContent = `${experiment} ／ ${r.n} 件 ／ 平均点 ${r.mean?.toFixed(3)}`;
    $("#r-table").innerHTML = `<tr><th>点数</th><th>正解</th><th>回答の長さ</th><th>モデルの回答（末尾）</th></tr>` + rows.map((x) =>
      `<tr><td class="${Number(x.score) === 1 ? "pass" : "fail"}">${x.score}</td><td>${escapeHtml(x.ground_truth)}</td><td>${x.answer_len}</td><td class="answer">${escapeHtml(x.answer_tail)}</td></tr>`).join("");
  } catch (e) { toast(`回答と点数の取得に失敗しました: ${e.message}`, true); }
}
