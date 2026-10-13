// 「自分のデータで学習」: one path, one main button per step.
//   1 which data -> 2 check & convert -> 3 how to score (+ confirmation test) -> 4 train -> 5 watch
// Uses api / toast / escapeHtml / loadRuns / CFG from app.js.
const W = { path: null, cols: [], hasImages: false, name: null, rewardFrom: null, runId: null, experiment: null,
  cm: null, trialRows: [], tests: [], dirty: false };
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
  $("#c-run").addEventListener("click", runCode);
  initEditor();
  sandboxBadge();
  onSource();
}

// ---- Python editor (CodeMirror 5, bundled under static/vendor) ---------------------------------
function initEditor() {
  W.cm = CodeMirror.fromTextArea($("#c-code"), {
    mode: "python", lineNumbers: true, indentUnit: 4, tabSize: 4, indentWithTabs: false,
    matchBrackets: true, autoCloseBrackets: true, viewportMargin: Infinity,
    extraKeys: { Tab: (cm) => cm.execCommand(cm.somethingSelected() ? "indentMore" : "insertSoftTab"), "Shift-Tab": "indentLess",
                 "Ctrl-Enter": runCode, "Cmd-Enter": runCode },
  });
  W.cm.on("change", (_, ch) => {
    if (ch.origin === "setValue") return;
    W.dirty = true;
    $("#c-dirty").textContent = "コードを直しました。テストしてから「この採点で決定」を押すのがおすすめです（Ctrl / ⌘ + Enter でもテストできます）";
  });
  $("#c-code-box").addEventListener("toggle", () => W.cm.refresh());   // CodeMirror needs a refresh once visible
}
const getCode = () => W.cm ? W.cm.getValue() : $("#c-code").value;
function setCode(code) { W.cm ? W.cm.setValue(code || "") : ($("#c-code").value = code || ""); markLine(null); W.dirty = false; $("#c-dirty").textContent = ""; }
function markLine(line) {
  if (!W.cm) return;
  if (W.errLine != null) W.cm.removeLineClass(W.errLine, "background", "cm-errline");
  W.errLine = null;
  if (line) { W.errLine = line - 1; W.cm.addLineClass(W.errLine, "background", "cm-errline"); W.cm.scrollIntoView({ line: W.errLine, ch: 0 }, 80); }
}

// ---- Sandbox: live progress + status badge ---------------------------------------------------------
// Renders the structured events from reward_agent (stage / state / detail / cmd / stdout ...).
function progressView(el, title) {
  const firstSeen = new Map(), doneAt = new Map();
  let events = [], running = true, ticker = null, t0 = Date.now();
  el.classList.remove("hidden");
  el.innerHTML = `<details open><summary></summary><ul class="stages"></ul><div class="term"></div></details>`;
  const sum = el.querySelector("summary"), ul = el.querySelector(".stages"), term = el.querySelector(".term");
  const fmt = (ms) => ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} 秒`;
  const kb = (b) => b < 1024 ? `${b} B` : `${(b / 1024).toFixed(1)} KB`;
  function render() {
    const stages = new Map();
    events.forEach((e) => {
      if (!e.stage) return;
      if (!firstSeen.has(e.stage)) firstSeen.set(e.stage, e.state === "active" ? Date.now() : null);
      if (e.state !== "active" && !doneAt.has(e.stage)) doneAt.set(e.stage, Date.now());
      stages.set(e.stage, e);
    });
    ul.innerHTML = [...stages.entries()].map(([key, e]) => {
      const st = e.state || "done", sb = /^sb-|^test-/.test(key);
      const t1 = firstSeen.get(key);
      const tm = st === "active" ? fmt(Date.now() - t1) : e.ms != null ? fmt(e.ms) : t1 ? fmt(doneAt.get(key) - t1) : "";
      const bar = e.total ? `<div class="bar"><i style="width:${Math.round(100 * (e.done || 0) / e.total)}%"></i></div>` : "";
      return `<li class="stage ${st}"><span class="ic"></span><span class="lbl">${sb ? `<span class="sb">[Sandbox]</span> ` : ""}${escapeHtml(e.text)}` +
        `${e.detail ? `<span class="det">${escapeHtml(e.detail)}</span>` : ""}${bar}</span><span class="tm">${tm}</span></li>`;
    }).join("");
    const execs = events.filter((e) => e.cmd && e.state !== "active");
    const live = events.filter((e) => e.cmd).slice(-1)[0];
    term.innerHTML = `<span class="ttl">Sandbox のターミナル${live && live.sandbox ? `（${escapeHtml(live.sandbox)}）` : ""}</span>\n` +
      execs.map((e) => `<span class="dim">送ったファイル: ${e.files.map((f) => `${escapeHtml(f.name)}（${kb(f.bytes)}）`).join("、")}</span>\n` +
        `<span class="cmd">$ ${escapeHtml(e.cmd)}</span>\n` +
        (e.stderr || []).map((l) => escapeHtml(l)).join("\n") + ((e.stderr || []).length ? "\n" : "") +
        (e.stdout || []).map((l) => `<span class="dim">${escapeHtml(l)}</span>`).join("\n") + ((e.stdout || []).length ? "\n" : "") +
        `<span class="${e.exit_code ? "err" : "dim"}">終了コード ${e.exit_code ?? "?"} ・ ${fmt(e.ms || 0)}</span>\n`).join("\n") +
      (live && live.state === "active" ? `<span class="cmd">$ ${escapeHtml(live.cmd)}</span>\n<span class="cursor"></span>` :
        running ? `<span class="dim">（待機中）</span> <span class="cursor"></span>` : "");
    term.scrollTop = term.scrollHeight;
    const total = fmt(Date.now() - t0);
    sum.textContent = running ? `⏳ ${title}（${total}）` : `${title} の記録（クリックで開閉）`;
  }
  ticker = setInterval(render, 250);
  setTimeout(sandboxBadge, 1500);
  return {
    add(log, base) { events = (base || []).concat(log); render(); },
    get events() { return events; },
    finish(ok, note) { running = false; clearInterval(ticker); render(); sandboxBadge(); if (note) sum.textContent = note; el.querySelector("details").open = !ok || el.dataset.keepOpen === "1"; },
  };
}

async function pollTask(tid, view, base) {
  for (;;) {
    const t = await api(`/api/custom/task/${tid}`);
    view.add(t.log, base);
    if (t.status === "done") return t.result;
    if (t.status === "error") throw new Error(t.error);
    await new Promise((r) => setTimeout(r, 700));
  }
}

let badgeTimer = null;
async function sandboxBadge() {
  clearTimeout(badgeTimer);
  const el = $("#sb-badge");
  try {
    const s = await api("/api/custom/sandbox");
    if (s.running) {
      const left = Math.max(0, s.idle_limit - s.idle);
      el.className = "sb-badge " + (s.busy ? "busy" : "on");
      el.innerHTML = `<span class="dot"></span>Sandbox ${s.busy ? "実行中" : "起動中"}：${escapeHtml(s.id)} ・ 起動から ${Math.floor(s.age / 60)} 分 ・ 実行 ${s.uses} 回` +
        (s.busy ? "" : ` ・ あと ${Math.ceil(left / 60)} 分使わなければ自動で削除 <button id="sb-del">今すぐ削除</button>`);
      const b = $("#sb-del");
      if (b) b.onclick = async () => { b.disabled = true; await api("/api/custom/sandbox", { method: "DELETE" }); sandboxBadge(); };
    } else {
      el.className = "sb-badge";
      el.innerHTML = `<span class="dot"></span>Sandbox：停止中（テストのときに自動で作ります）`;
    }
  } catch (e) { el.textContent = ""; }
  if (!$("#tab-custom").classList.contains("hidden")) badgeTimer = setTimeout(sandboxBadge, 3000);
  else badgeTimer = setTimeout(sandboxBadge, 15000);
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
    W.tests = [];
    showReview(r.code, null, v === "numeric" ? "数値で比べる（標準）" : "文字が完全に同じなら正解");
    await trial(progressView($("#c-progress"), "確認テストを実行中"));
  } finally { $("#c-rw-go").disabled = false; }
}

async function runAgent() {
  $("#c-agent").disabled = true;
  const view = progressView($("#c-progress"), "AI が採点関数を作っています");
  try {
    const { task_id } = await post(`/api/custom/${encodeURIComponent(W.name)}/reward/agent`, { criteria: $("#c-criteria").value });
    const res = await pollTask(task_id, view);
    W.tests = res.tests || [];
    showReview(res.code, res.report, "AI が作った採点関数");
    await trial(view, view.events);
  } catch (e) { view.finish(false, "❌ 採点関数を作れませんでした"); status("#w3-status", `❌ ${escapeHtml(e.message)}`, "fail"); }
  finally { $("#c-agent").disabled = false; }
}

function showReview(code, report, from) {
  W.rewardFrom = from;
  $("#c-review").classList.remove("hidden");
  setCode(code);
  $("#c-trial-out").innerHTML = "";
  $("#c-run-out").innerHTML = "";
  $("#c-run-progress").classList.add("hidden");
  $("#c-tests").innerHTML = report ? `<p>AI が作ったテスト：<span class="${report.failed ? "fail" : "pass"}">${report.passed} / ${report.total} 件合格</span>（Sandbox で実行）
    <details><summary class="muted">テストの中身</summary><table class="small"><tr><th></th><th>期待</th><th>実際</th><th>正解</th><th>答え（末尾）</th><th>ねらい</th></tr>${
    report.results.map((r) => `<tr><td class="${r.ok ? "pass" : "fail"}">${r.ok ? "✔" : "✘"}</td><td>${r.expected}</td><td>${r.got ?? escapeHtml(r.error || "")}</td><td>${escapeHtml(r.ground_truth)}</td><td class="answer">${escapeHtml(r.response_tail)}</td><td>${escapeHtml(r.why)}</td></tr>`).join("")}</table></details></p>` : "";
  status("#w3-status", `採点の候補：<b>${escapeHtml(from)}</b>`);
}

async function trial(view, base) {
  view = view || progressView($("#c-progress"), "確認テストを実行中");
  try {
    const { task_id } = await post(`/api/custom/${encodeURIComponent(W.name)}/reward/trial`, { code: getCode(), n: 5 });
    const rows = await pollTask(task_id, view, base);
    W.trialRows = rows;
    view.finish(true, `✅ 確認テストが終わりました（${rows.length} 問。Sandbox での実行の記録はクリックで開けます）`);
    $("#c-trial-out").innerHTML = `<table class="small"><tr><th>点数</th><th>正解</th><th>AI の答え（末尾）</th></tr>${
      rows.map((r) => `<tr><td class="${r.score === 1 ? "pass" : "fail"}">${r.score ?? escapeHtml(r.error || "")}</td><td>${escapeHtml(r.ground_truth)}</td><td class="answer">${escapeHtml((r.response || "").slice(-300))}</td></tr>`).join("")}</table>`;
  } catch (e) { view.finish(false, "❌ 確認テストに失敗しました"); status("#w3-status", `❌ 確認テストに失敗しました：${escapeHtml(e.message)}`, "fail"); }
}

// ---- editor's test button: run the edited code in the Sandbox ----------------------------------------
async function runCode() {
  if (!W.name || $("#c-run").disabled) return;
  $("#c-run").disabled = true;
  markLine(null);
  const view = progressView($("#c-run-progress"), "Sandbox で直したコードを実行中");
  const out = $("#c-run-out");
  out.innerHTML = "";
  try {
    const { task_id } = await post(`/api/custom/${encodeURIComponent(W.name)}/reward/run`, {
      code: getCode(), answers: W.trialRows, tests: W.tests,
      case: { response: $("#c-case-resp").value, ground_truth: $("#c-case-gt").value } });
    const r = await pollTask(task_id, view);
    const bad = r.timeout ? "20 秒以内に終わりませんでした（無限ループになっていないか確かめてください）"
      : r.syntax && !r.syntax.ok ? `構文エラー：${r.syntax.line} 行目：${r.syntax.msg}`
      : r.load && !r.load.ok ? `読み込めません${r.load.line ? `（${r.load.line} 行目）` : ""}：${r.load.msg}` : null;
    if (bad) {
      markLine((r.syntax && r.syntax.line) || (r.load && r.load.line));
      view.finish(false, "❌ 実行できませんでした");
      out.innerHTML = `<p class="fail">❌ ${escapeHtml(bad)}</p>`;
      return;
    }
    const errLine = [...(r.answers || []), ...(r.tests || []), r.case || {}].map((x) => x.line).find(Boolean);
    markLine(errLine);
    const changed = (r.answers || []).filter((a, i) => a.score !== W.trialRows[i]?.score).length;
    let html = `<p class="pass">✅ 構文チェック OK ・ Sandbox で実行しました</p>`;
    if ((r.answers || []).length) {
      html += `<h4>確認テストの回答を採点し直した結果（点数が変わったもの：${changed} 件）</h4><table class="small"><tr><th>前の点数</th><th>今の点数</th><th>正解</th><th>AI の答え（末尾）</th></tr>` +
        r.answers.map((a, i) => { const prev = W.trialRows[i]?.score, diff = a.score !== prev;
          return `<tr><td>${prev ?? "—"}</td><td class="${a.error ? "fail" : a.score === 1 ? "pass" : "fail"} ${diff ? "changed" : ""}">${a.error ? `エラー：${escapeHtml(a.error)}${a.line ? `（${a.line} 行目）` : ""}` : a.score}${diff ? " ←" : ""}</td>` +
            `<td>${escapeHtml(W.trialRows[i]?.ground_truth ?? "")}</td><td class="answer">${escapeHtml((W.trialRows[i]?.response || "").slice(-200))}</td></tr>`; }).join("") + "</table>";
      r.answers.forEach((a, i) => { if (W.trialRows[i] && !a.error) W.trialRows[i].score = a.score; });   // next comparison is against this run
    }
    if ((r.tests || []).length) {
      const pass = r.tests.filter((t) => t.ok).length, fails = r.tests.filter((t) => !t.ok);
      html += `<h4>AI が作ったテスト：<span class="${fails.length ? "fail" : "pass"}">${pass} / ${r.tests.length} 件合格</span></h4>` + (fails.length ?
        `<table class="small"><tr><th>期待</th><th>実際</th><th>正解</th><th>答え（末尾）</th><th>ねらい</th></tr>${fails.map((t) =>
          `<tr><td>${t.expected}</td><td class="fail">${t.error ? escapeHtml(t.error) : t.got}</td><td>${escapeHtml(t.ground_truth)}</td><td class="answer">${escapeHtml(t.response_tail)}</td><td>${escapeHtml(t.why)}</td></tr>`).join("")}</table>` : "");
    }
    if (r.case) html += `<h4>自分で試した1件：<span class="${r.case.score === 1 ? "pass" : "fail"}">${r.case.error ? `エラー：${escapeHtml(r.case.error)}` : `点数 ${r.case.score}`}</span></h4>`;
    out.innerHTML = html;
    view.finish(true, "✅ Sandbox で実行しました（記録はクリックで開けます）");
    W.dirty = false;
    $("#c-dirty").textContent = "";
  } catch (e) { view.finish(false, "❌ 実行に失敗しました"); out.innerHTML = `<p class="fail">❌ ${escapeHtml(e.message)}</p>`; }
  finally { $("#c-run").disabled = false; sandboxBadge(); }
}

async function approve() {
  try {
    await post(`/api/custom/${encodeURIComponent(W.name)}/reward/approve`, { code: getCode() });
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
    $("#t-row").classList.remove("hidden");
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
  if (!W.runId) $("#t-row").classList.add("hidden");   // opened from the run list: status is shown there
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
