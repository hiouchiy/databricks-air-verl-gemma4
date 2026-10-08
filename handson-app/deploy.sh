#!/usr/bin/env bash
# =============================================================================
# deploy.sh — deploy the hands-on UI (Databricks App) for this repo.
#
# Creates the app (first time), grants its service principal the Unity Catalog
# privileges it needs, renders app.yaml with your values, copies the repo's job
# templates into the app, and deploys.
#
# Prerequisites:
#   - Databricks CLI v1.19.0+ and `databricks auth login -p <PROFILE>` (as someone who can
#     create apps and GRANT on the catalog/schema)
#   - The training image is already in Artifact Registry as
#     <CATALOG>.<SCHEMA>.<IMAGE_NAME>:<IMAGE_TAG> (setup.md §3)
#
# Usage:
#   PROFILE=handson CATALOG=<catalog> SCHEMA=<schema> bash handson-app/deploy.sh
#   (optional: APP_NAME=air-handson VOLUME=air_handson IMAGE_NAME=verl-gemma4 IMAGE_TAG=v1)
# =============================================================================
set -euo pipefail

: "${PROFILE:?set PROFILE}"; : "${CATALOG:?set CATALOG}"; : "${SCHEMA:?set SCHEMA}"
APP_NAME="${APP_NAME:-air-handson}"
VOLUME="${VOLUME:-air_handson}"
IMAGE_NAME="${IMAGE_NAME:-verl-gemma4}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-/Workspace/Shared/air-handson}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
BUILD="$HERE/.build"
DB=(databricks -p "$PROFILE")

say() { printf "\n==> %s\n" "$*"; }

say "前提チェック"
"${DB[@]}" current-user me >/dev/null || { echo "プロファイル $PROFILE で認証できません（databricks auth login）"; exit 1; }
ME=$("${DB[@]}" current-user me -o json | python3 -c "import json,sys;print(json.load(sys.stdin)['userName'])")

say "アプリ $APP_NAME を用意"
if ! "${DB[@]}" apps get "$APP_NAME" >/dev/null 2>&1; then
  "${DB[@]}" apps create "$APP_NAME" --description "Gemma 4 x verl GRPO hands-on (AI Runtime)" >/dev/null
fi
SP=$("${DB[@]}" apps get "$APP_NAME" -o json | python3 -c "import json,sys;print(json.load(sys.stdin)['service_principal_client_id'])")
echo "  service principal: $SP"

say "サービスプリンシパルに UC の権限を付与"
"${DB[@]}" grants update catalog "$CATALOG" \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_CATALOG\"]}]}" >/dev/null
"${DB[@]}" grants update schema "$CATALOG.$SCHEMA" \
  --json "{\"changes\":[{\"principal\":\"$SP\",\"add\":[\"USE_SCHEMA\",\"READ_VOLUME\",\"WRITE_VOLUME\",\"CREATE_VOLUME\"]}]}" >/dev/null
echo "  $CATALOG: USE_CATALOG / $CATALOG.$SCHEMA: USE_SCHEMA, READ_VOLUME, WRITE_VOLUME, CREATE_VOLUME"

say "共有フォルダ $EXPERIMENT_DIR を用意（MLflow 実験とデータ準備スクリプト）"
"${DB[@]}" workspace mkdirs "$EXPERIMENT_DIR"
"${DB[@]}" workspace import "$EXPERIMENT_DIR/prep_data.py" --file "$REPO/prep_data.py" --format AUTO --overwrite

say "ビルド（テンプレートのコピーと app.yaml の生成）"
rm -rf "$BUILD" && mkdir -p "$BUILD/templates"
cp "$HERE/app.py" "$BUILD/"
cp -R "$HERE/static" "$BUILD/"
for t in smoke_test.yaml prep_data_job.json grpo_gemma4.yaml grpo_gemma4_multinode.yaml \
         grpo_gemma4_mm.yaml grpo_gemma4_mm_multinode.yaml run_grpo.sh run_grpo_multinode.sh; do
  cp "$REPO/$t" "$BUILD/templates/"
done
cat > "$BUILD/app.yaml" <<EOF
command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
env:
  - name: AIR_CATALOG
    value: "$CATALOG"
  - name: AIR_SCHEMA
    value: "$SCHEMA"
  - name: AIR_VOLUME
    value: "$VOLUME"
  - name: AIR_IMAGE_NAME
    value: "$IMAGE_NAME"
  - name: AIR_IMAGE_TAG
    value: "$IMAGE_TAG"
  - name: AIR_EXPERIMENT_DIR
    value: "$EXPERIMENT_DIR"
EOF

say "アップロードしてデプロイ"
SRC="/Workspace/Users/$ME/apps/$APP_NAME"
"${DB[@]}" workspace mkdirs "$SRC"
"${DB[@]}" workspace import-dir "$BUILD" "$SRC" --overwrite >/dev/null
"${DB[@]}" apps deploy "$APP_NAME" --source-code-path "$SRC" -o json \
  | python3 -c "import json,sys;d=json.load(sys.stdin)['status'];print('  deploy:',d['state'],d.get('message',''))"
URL=$("${DB[@]}" apps get "$APP_NAME" -o json | python3 -c "import json,sys;print(json.load(sys.stdin)['url'])")
cat <<EOF

==================== デプロイ完了 ====================
  アプリ  : $URL
  イメージ: $CATALOG.$SCHEMA.$IMAGE_NAME:$IMAGE_TAG
  Volume  : /Volumes/$CATALOG/$SCHEMA/$VOLUME
  MLflow  : $EXPERIMENT_DIR
停止するとき（課金を止める）: databricks apps stop $APP_NAME -p $PROFILE
EOF
