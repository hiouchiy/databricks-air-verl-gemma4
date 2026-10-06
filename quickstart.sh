#!/usr/bin/env bash
# =============================================================================
# quickstart.sh — Gemma4-26B-A4B / verl / FSDP2 / GRPO 環境の一括構築スクリプト
#
# 対話で数項目を入力すると、以下を自動で行います（setup.md の §1後半〜§3）:
#   1) UC Volume の作成
#   2) テンプレート（*.yaml / *.sh）の値をあなたの環境向けに置換
#   3) カスタム Docker イメージをビルド（1回）→ Databricks Artifact Registry（UC）へ push
# 完了後は §4（データ準備 → 単一ノード学習）/ §5（マルチノード学習）を手動で実行できます。
#
# Gemma4 は SDPA を使うため、Qwen 系のような FlashAttention wheel ビルド工程はありません
# （イメージは 1回のビルドで完結）。
#
# 【事前に済ませておくこと】（このスクリプトには含めません）
#   - ソース一式のクローン（setup.md §2）— このスクリプトはそのディレクトリ内で実行
#   - `databricks auth login --host <URL> --profile <PROFILE>`（ブラウザ認証）
#   - `git` / `databricks`（Databricks CLI v1.19.0 以上）/ `docker` がインストール済みであること
#     （AI Runtime CLI は Databricks CLI の `databricks air` に統合済み。旧 `air` は不要）
#   - ワークスペース管理者が Previews で「AI Runtime Beta Features」と
#     「Databricks Artifact Registry」を有効化済みであること
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
ask CATALOG       "UC カタログ名（イメージと Volume の置き場）" "main"
ask SCHEMA        "UC スキーマ名" "default"
ask IMAGE_TAG     "イメージのタグ" "v1"
ask SKIP_BUILD    "UC 上の既存イメージ ${CATALOG}.${SCHEMA}.verl-gemma4:<タグ> を使い、ビルド/push をスキップしますか？ (y/N)" "N"
ask VOLUME_NAME   "UC Volume 名" "verl_workspace"
ask WS_EMAIL      "Workspace のあなたのメールアドレス (MLflow 実験ディレクトリ用)"
ask PIP_INDEX     "pip インデックス URL" "https://pypi.org/simple"

LOCAL_IMAGE="verl-gemma4:${IMAGE_TAG}"
# YAML の environment.unity_catalog_image に入れる UC 名（レジストリのホスト名は含めない）
UC_IMAGE="${CATALOG}.${SCHEMA}.verl-gemma4:${IMAGE_TAG}"
VOL="/Volumes/${CATALOG}/${SCHEMA}/${VOLUME_NAME}"

cat <<EOF

--- 確認 ---
  プロファイル : ${PROFILE}
  学習イメージ : ${UC_IMAGE}  (ビルド/push スキップ: ${SKIP_BUILD})
  UC Volume    : ${VOL}
  MLflow email : ${WS_EMAIL}
  pip index    : ${PIP_INDEX}
-------------
EOF
read -r -p "この内容で進めます。よろしいですか？ [y/N]: " OK
[ "${OK}" = "y" ] || [ "${OK}" = "Y" ] || { echo "中止しました。"; exit 1; }

# ---- 0. 前提チェック --------------------------------------------------------
say "前提チェック（Databricks CLI / docker / 認証）"
command -v databricks >/dev/null || { echo "databricks CLI が見つかりません。v1.19.0 以上をインストールしてください。"; exit 1; }
databricks air images push --help >/dev/null 2>&1 \
  || { echo "databricks CLI が古いです（$(databricks --version)）。v1.19.0 以上に更新してください（'databricks air images push' が必要）。"; exit 1; }
case "${SKIP_BUILD}" in y|Y) ;; *)
  command -v docker >/dev/null || { echo "docker が見つかりません。Docker をインストール/起動してください。"; exit 1; } ;;
esac
databricks current-user me -p "${PROFILE}" >/dev/null 2>&1 \
  || { echo "プロファイル ${PROFILE} で認証できません。先に 'databricks auth login' を実行してください。"; exit 1; }

# ---- 1. UC Volume 作成 ------------------------------------------------------
say "UC Volume 作成: ${VOL}"
databricks volumes create "${CATALOG}" "${SCHEMA}" "${VOLUME_NAME}" MANAGED -p "${PROFILE}" 2>&1 | tail -3 \
  || echo "（既に存在する場合はこのままで問題ありません）"

# ---- 2. テンプレート置換 ----------------------------------------------------
# テンプレートには __IMAGE__ / __VOL__ / __WS_EMAIL__ のプレースホルダが入って
# います。置換結果を gen/ サブディレクトリに *元の拡張子のまま* 出力します
# （databricks air CLI は .yaml/.yml しか受け付けないため、`*.yaml.gen` にはできない）。元テンプレは保持。
# マルチモーダル(§6)用の grpo_gemma4_mm*.yaml も含める。
say "テンプレートを環境向けに置換 (gen/ に生成)"
mkdir -p gen
TEMPLATES="grpo_gemma4.yaml grpo_gemma4_multinode.yaml grpo_gemma4_mm.yaml grpo_gemma4_mm_multinode.yaml smoke_test.yaml prep_gsm8k_deps.yaml prep_geo3k_deps.yaml run_grpo.sh run_grpo_multinode.sh"
for t in ${TEMPLATES}; do
  [ -f "$t" ] || continue
  sed -e "s#__IMAGE__#${UC_IMAGE}#g" \
      -e "s#__VOL__#${VOL}#g" \
      -e "s#__WS_EMAIL__#${WS_EMAIL}#g" \
      "$t" > "gen/${t}"
  echo "  生成: gen/${t}"
done

# ---- 3. カスタムイメージをビルド → Artifact Registry へ push（1回で完結） ----
# Gemma4 は SDPA を使うため FlashAttention は不要（BUILD_FLASH_ATTN=0）。
# Dockerfile の各 install は --find-links /wheelhouse を使うため、/wheelhouse は
# 存在必須（空でよい。無いと uv が即エラー）。
# push 先は Databricks Artifact Registry（UC の <catalog>.<schema>.verl-gemma4:<tag>）。
# 旧方式の Docker Hub push + `air register image` は使わない（`databricks air` には
# register コマンドも YAML の environment.docker_image も無い）。
case "${SKIP_BUILD}" in
  y|Y)
    say "§3: ビルド/push をスキップ（既存の ${UC_IMAGE} を使用）" ;;
  *)
    say "§3: カスタム Docker イメージをビルド（BUILD_FLASH_ATTN=0・1回で完結）"
    mkdir -p wheelhouse
    docker build --platform linux/amd64 \
      --build-arg PIP_INDEX_URL="${PIP_INDEX}" \
      --build-arg BUILD_FLASH_ATTN=0 \
      -v "$PWD/wheelhouse:/wheelhouse:ro" \
      -t "${LOCAL_IMAGE}" -f Dockerfile . \
      || { echo "docker が -v (build mount) 非対応の場合は、Dockerfile 側で wheelhouse を COPY する方式に切り替えてください。"; exit 1; }

    say "§3: イメージを Databricks Artifact Registry へ push → ${UC_IMAGE}"
    databricks air images push -p "${PROFILE}" \
      --source "${LOCAL_IMAGE}" \
      --catalog "${CATALOG}" --schema "${SCHEMA}" \
      --artifact "verl-gemma4:${IMAGE_TAG}" ;;
esac

# ---- 完了 -------------------------------------------------------------------
# NOTE: plain heredoc (no \033 ANSI codes) — inside cat <<EOF they would print
# literally. Use the say() helper above if you want colored output.
cat <<EOF

==================== 環境構築が完了しました ====================
次のステップ（手動で実行）:

  # 疎通確認（任意・安価）
  databricks air run --file gen/smoke_test.yaml -p ${PROFILE} --watch

  # 動作確認用データ（gsm8k, テキスト）の準備
  databricks air run --file gen/prep_gsm8k_deps.yaml -p ${PROFILE} --watch

  # 単一ノード（8×H100）で GRPO
  databricks air run --file gen/grpo_gemma4.yaml -p ${PROFILE} --watch

  # マルチノード（2ノード=16×H100）で GRPO
  databricks air run --file gen/grpo_gemma4_multinode.yaml -p ${PROFILE} --watch

詳細は setup.md を参照してください。
EOF
