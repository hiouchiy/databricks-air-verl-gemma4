#!/usr/bin/env bash
# =============================================================================
# quickstart.sh — Gemma4-26B-A4B / verl / FSDP2 / GRPO 環境の一括構築スクリプト
#
# 対話で数項目を入力すると、以下を自動で行います（setup.md の §1後半〜§3）:
#   1) UC Volume の作成
#   2) テンプレート（*.yaml / *.sh）の値をあなたの環境向けに置換
#   3) カスタム Docker イメージをビルド（1回）→ push → 登録
# 完了後は §4（データ準備 → 単一ノード学習）/ §5（マルチノード学習）を手動で実行できます。
#
# Gemma4 は SDPA を使うため、Qwen 系のような FlashAttention wheel ビルド工程はありません
# （イメージは 1回のビルドで完結）。
#
# 【事前に済ませておくこと】（このスクリプトには含めません）
#   - ソース一式のクローン（setup.md §2）— このスクリプトはそのディレクトリ内で実行
#   - `databricks auth login --host <URL> --profile <PROFILE>`（ブラウザ認証）
#   - `docker login`（Docker Hub へのログイン）
#   - `git` / `air`（databricks-air）/ `docker` がインストール済みであること
#
# 使い方:
#   bash quickstart.sh
# =============================================================================
set -euo pipefail

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
ask() { # ask VAR "prompt" "default"
  local __v="$1" __p="$2" __d="${3:-}" __in
  if [ -n "$__d" ]; then read -r -p "$__p [$__d]: " __in; __in="${__in:-$__d}"
  else read -r -p "$__p: " __in; fi
  printf -v "$__v" '%s' "$__in"
}

cd "$(dirname "$0")"

say "対話入力（環境情報）"
ask PROFILE       "Databricks CLI プロファイル名 (databricks auth login で作ったもの)"
ask DOCKERHUB_USER "Docker Hub のユーザー名"
ask IMAGE_TAG     "イメージのタグ" "v1"
ask CATALOG       "UC カタログ名" "main"
ask SCHEMA        "UC スキーマ名" "default"
ask VOLUME_NAME   "UC Volume 名" "verl_workspace"
ask WS_EMAIL      "Workspace のあなたのメールアドレス (MLflow 実験ディレクトリ用)"
ask PIP_INDEX     "pip インデックス URL" "https://pypi.org/simple"

IMAGE="docker.io/${DOCKERHUB_USER}/verl-gemma4:${IMAGE_TAG}"
IMAGE_SHORT="${DOCKERHUB_USER}/verl-gemma4:${IMAGE_TAG}"
VOL="/Volumes/${CATALOG}/${SCHEMA}/${VOLUME_NAME}"

cat <<EOF

--- 確認 ---
  プロファイル : ${PROFILE}
  学習イメージ : ${IMAGE}
  UC Volume    : ${VOL}
  MLflow email : ${WS_EMAIL}
  pip index    : ${PIP_INDEX}
-------------
EOF
read -r -p "この内容で進めます。よろしいですか？ [y/N]: " OK
[ "${OK}" = "y" ] || [ "${OK}" = "Y" ] || { echo "中止しました。"; exit 1; }

# ---- 0. 前提チェック --------------------------------------------------------
say "前提チェック（air / docker / 認証）"
command -v air >/dev/null    || { echo "air が見つかりません。'uv tool install databricks-air' を実行してください。"; exit 1; }
command -v docker >/dev/null || { echo "docker が見つかりません。Docker をインストール/起動してください。"; exit 1; }
databricks current-user me -p "${PROFILE}" >/dev/null 2>&1 \
  || { echo "プロファイル ${PROFILE} で認証できません。先に 'databricks auth login' を実行してください。"; exit 1; }

# ---- 1. UC Volume 作成 ------------------------------------------------------
say "UC Volume 作成: ${VOL}"
databricks volumes create "${CATALOG}" "${SCHEMA}" "${VOLUME_NAME}" MANAGED -p "${PROFILE}" 2>&1 | tail -3 \
  || echo "（既に存在する場合はこのままで問題ありません）"

# ---- 2. テンプレート置換 ----------------------------------------------------
# テンプレートには __IMAGE__ / __VOL__ / __WS_EMAIL__ のプレースホルダが入って
# います。作業用に .gen ファイルを生成して置換します（元テンプレは保持）。
say "テンプレートを環境向けに置換 (*.gen を生成)"
TEMPLATES="grpo_gemma4.yaml grpo_gemma4_multinode.yaml smoke_test.yaml prep_gsm8k_deps.yaml run_grpo.sh run_grpo_multinode.sh"
for t in ${TEMPLATES}; do
  [ -f "$t" ] || continue
  sed -e "s#__IMAGE__#${IMAGE_SHORT}#g" \
      -e "s#__VOL__#${VOL}#g" \
      -e "s#__WS_EMAIL__#${WS_EMAIL}#g" \
      "$t" > "${t}.gen"
  echo "  生成: ${t}.gen"
done

# ---- 3. カスタムイメージをビルド → push → 登録（1回で完結） ----------------
# Gemma4 は SDPA を使うため FlashAttention は不要（BUILD_FLASH_ATTN=0）。
# Dockerfile の各 install は --find-links /wheelhouse を使うため、/wheelhouse は
# 存在必須（空でよい。無いと uv が即エラー）。
say "§3: カスタム Docker イメージをビルド（BUILD_FLASH_ATTN=0・1回で完結）"
mkdir -p wheelhouse
docker build --platform linux/amd64 \
  --build-arg PIP_INDEX_URL="${PIP_INDEX}" \
  --build-arg BUILD_FLASH_ATTN=0 \
  -v "$PWD/wheelhouse:/wheelhouse:ro" \
  -t "${IMAGE}" -f Dockerfile . \
  || { echo "docker が -v (build mount) 非対応の場合は、Dockerfile 側で wheelhouse を COPY する方式に切り替えてください。"; exit 1; }

say "§3: イメージを push"
docker push "${IMAGE}"

say "§3: イメージを AI Runtime に登録"
air register image "${IMAGE_SHORT}" -p "${PROFILE}"

# ---- 完了 -------------------------------------------------------------------
cat <<EOF

\033[1;32m==================== 環境構築が完了しました ====================\033[0m
次のステップ（手動で実行）:

  # 疎通確認（任意・安価）
  air run --file smoke_test.yaml.gen -p ${PROFILE} --watch

  # 動作確認用データ（gsm8k, テキスト）の準備
  air run --file prep_gsm8k_deps.yaml.gen -p ${PROFILE} --watch

  # 単一ノード（8×H100）で GRPO
  air run --file grpo_gemma4.yaml.gen -p ${PROFILE} --watch

  # マルチノード（2ノード=16×H100）で GRPO
  air run --file grpo_gemma4_multinode.yaml.gen -p ${PROFILE} --watch

詳細は setup.md を参照してください。
EOF
