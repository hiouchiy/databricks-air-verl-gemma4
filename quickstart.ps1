# =============================================================================
# quickstart.ps1 — quickstart.sh の Windows (PowerShell) 版
#
# 対話で数項目を入力すると、以下を自動で行います（setup.md §7-2）:
#   1) UC Volume の作成
#   2) テンプレート（*.yaml / *.sh）の値をあなたの環境向けに置換 → gen\ に生成
#   3) 学習用イメージを Databricks Artifact Registry（UC）に用意する:
#        hub  = Docker Hub の検証済みイメージを取り込む（既定。Docker が必要）
#        skip = すでに UC に取り込み済みのイメージを使う（Docker 不要）
#      ※ Windows ではソースからのビルド（build）は扱いません。必要なら WSL で quickstart.sh を使ってください。
#
# 【事前に済ませておくこと】
#   - git clone（または ZIP 展開）したディレクトリの中で実行する
#   - Databricks CLI v1.19.0 以上をインストールし、databricks auth login 済みであること
#   - hub を選ぶ場合は Docker Desktop が起動していること
#
# 使い方（PowerShell）:
#   powershell -ExecutionPolicy Bypass -File .\quickstart.ps1
# =============================================================================

function Say($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Ask($prompt, $default) {
    if ($default) {
        $in = Read-Host "$prompt [$default]"
        if ([string]::IsNullOrWhiteSpace($in)) { return $default } else { return $in.Trim() }
    } else {
        return (Read-Host $prompt).Trim()
    }
}
function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }

$Root = $PSScriptRoot
Set-Location $Root

Say "対話入力（環境情報）"
$Profile_     = Ask "Databricks CLI プロファイル名 (databricks auth login で作ったもの)" ""
$Catalog      = Ask "UC カタログ名（イメージと Volume の置き場）" "main"
$Schema       = Ask "UC スキーマ名" "default"
$ImageTag     = Ask "イメージのタグ" "v1"
$ImageSource  = Ask "学習イメージの用意方法 (hub=Docker Hub から取り込み / skip=取り込み済みを使う)" "hub"
$VolumeName   = Ask "UC Volume 名" "verl_workspace"
$WsEmail      = Ask "Workspace のあなたのメールアドレス (MLflow 実験ディレクトリ用)" ""

if ($ImageSource -ne "hub" -and $ImageSource -ne "skip") { Fail "用意方法は hub または skip を指定してください（入力: $ImageSource）" }

$HubImage = "docker.io/hiouchiy/verl-gemma4:v4-verify"   # 実機検証済みの公開イメージ
$UcImage  = "$Catalog.$Schema.verl-gemma4:$ImageTag"     # YAML の environment.unity_catalog_image（ホスト名は含めない）
$Vol      = "/Volumes/$Catalog/$Schema/$VolumeName"

Write-Host ""
Write-Host "--- 確認 ---"
Write-Host "  プロファイル : $Profile_"
Write-Host "  学習イメージ : $UcImage  (用意方法: $ImageSource)"
Write-Host "  UC Volume    : $Vol"
Write-Host "  MLflow email : $WsEmail"
Write-Host "-------------"
$ok = Read-Host "この内容で進めます。よろしいですか？ [y/N]"
if ($ok -ne "y" -and $ok -ne "Y") { Fail "中止しました。" }

# ---- 0. 前提チェック --------------------------------------------------------
Say "前提チェック（Databricks CLI / docker / 認証）"
if (-not (Get-Command databricks -ErrorAction SilentlyContinue)) { Fail "databricks CLI が見つかりません。v1.19.0 以上をインストールしてください（winget install Databricks.DatabricksCLI）。" }
databricks air images push --help *> $null
if ($LASTEXITCODE -ne 0) { Fail "databricks CLI が古いです（$(databricks --version)）。v1.19.0 以上に更新してください（winget upgrade Databricks.DatabricksCLI）。" }
if ($ImageSource -eq "hub" -and -not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "docker が見つかりません。Docker Desktop を起動するか、Docker を使わない方法（setup.md §7-2 の crane）で取り込んでから、用意方法に skip を指定して再実行してください。"
}
databricks current-user me -p $Profile_ *> $null
if ($LASTEXITCODE -ne 0) { Fail "プロファイル $Profile_ で認証できません。先に databricks auth login を実行してください。" }

# ---- 1. UC Volume 作成 ------------------------------------------------------
Say "UC Volume 作成: $Vol"
$out = databricks volumes create $Catalog $Schema $VolumeName MANAGED -p $Profile_ 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { Write-Host ($out.Trim()); Write-Host "（既に存在する場合はこのままで問題ありません）" }

# ---- 2. テンプレート置換 ----------------------------------------------------
# gen\ に元の拡張子のまま出力する。生成物は Linux の学習ノードで実行されるため、
# 改行は LF・BOM なしの UTF-8 で書き出す（CRLF のままだと bash が失敗する）。
Say "テンプレートを環境向けに置換 (gen\ に生成)"
$GenDir = Join-Path $Root "gen"
New-Item -ItemType Directory -Force -Path $GenDir | Out-Null
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false
$Templates = @("grpo_gemma4.yaml","grpo_gemma4_multinode.yaml","grpo_gemma4_mm.yaml","grpo_gemma4_mm_multinode.yaml","smoke_test.yaml","prep_gsm8k_deps.yaml","prep_geo3k_deps.yaml","run_grpo.sh","run_grpo_multinode.sh")
foreach ($t in $Templates) {
    $src = Join-Path $Root $t
    if (-not (Test-Path $src)) { continue }
    $text = [System.IO.File]::ReadAllText($src, $Utf8NoBom)
    $text = $text.Replace("__IMAGE__", $UcImage).Replace("__VOL__", $Vol).Replace("__WS_EMAIL__", $WsEmail)
    $text = $text.Replace("`r`n", "`n")
    [System.IO.File]::WriteAllText((Join-Path $GenDir $t), $text, $Utf8NoBom)
    Write-Host "  生成: gen\$t"
}

# ---- 3. 学習用イメージを Artifact Registry に用意 -----------------------------
if ($ImageSource -eq "skip") {
    Say "§3: 取り込み済みの $UcImage を使用（取り込みはスキップ）"
} else {
    # pull はこの PC 上で行われる（Docker と 20GB 以上の空きディスクが必要）
    Say "§3: Docker Hub の検証済みイメージを取り込み: $HubImage → $UcImage"
    databricks air images push -p $Profile_ --source $HubImage --catalog $Catalog --schema $Schema --artifact "verl-gemma4:$ImageTag"
    if ($LASTEXITCODE -ne 0) { Fail "イメージの取り込みに失敗しました。Docker Desktop が起動しているか確認してください。" }
}

# ---- 完了 -------------------------------------------------------------------
Write-Host ""
Write-Host "==================== 環境構築が完了しました ===================="
Write-Host "次のステップ（手動で実行）:"
Write-Host ""
Write-Host "  # 疎通確認（任意・安価）"
Write-Host "  databricks air run --file gen\smoke_test.yaml -p $Profile_ --watch"
Write-Host ""
Write-Host "  # 動作確認用データ（gsm8k, テキスト）の準備"
Write-Host "  databricks air run --file gen\prep_gsm8k_deps.yaml -p $Profile_ --watch"
Write-Host ""
Write-Host "  # 単一ノード（8×H100）で GRPO"
Write-Host "  databricks air run --file gen\grpo_gemma4.yaml -p $Profile_ --watch"
Write-Host ""
Write-Host "  # マルチノード（2ノード=16×H100）で GRPO"
Write-Host "  databricks air run --file gen\grpo_gemma4_multinode.yaml -p $Profile_ --watch"
Write-Host ""
Write-Host "詳細は setup.md を参照してください。"
