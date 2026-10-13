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
LLM_ENDPOINT="${LLM_ENDPOINT:-databricks-claude-sonnet-5-5}"   # reward agent (「自分のデータで学習」)
SKILL_ID="${SKILL_ID:-grpo-reward-author}"                      # UC skill published to $CATALOG.$SCHEMA

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
BUILD="$HERE/.build"
DB=(databricks -p "$PROFILE")

say() { printf "\n==> %s\n" "$*"; }

say "前提チェック"
"${DB[@]}" current-user me >/dev/null || { echo "プロファイル $PROFILE で認証できません（databricks auth login）"; exit 1; }
ME=$("${DB[@]}" current-user me -o json | python3 -c "import json,sys;print(json.load(sys.stdin)['userName'])")

say "アプリ $APP_NAME を用意"
RESOURCES="[{\"name\":\"llm\",\"serving_endpoint\":{\"name\":\"$LLM_ENDPOINT\",\"permission\":\"CAN_QUERY\"}}]"
if ! "${DB[@]}" apps get "$APP_NAME" >/dev/null 2>&1; then
  "${DB[@]}" apps create --json "{\"name\":\"$APP_NAME\",\"description\":\"Gemma 4 x verl GRPO hands-on (AI Runtime)\",\"resources\":$RESOURCES}" >/dev/null
else
  "${DB[@]}" apps update "$APP_NAME" --json "{\"name\":\"$APP_NAME\",\"resources\":$RESOURCES}" >/dev/null
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
for f in prep_data.py prep_custom.py make_sample_raw.py; do
  "${DB[@]}" workspace import "$EXPERIMENT_DIR/$f" --file "$REPO/$f" --format AUTO --overwrite
done

say "採点関数エージェントのスキルを Unity Catalog に公開: $CATALOG.$SCHEMA.$SKILL_ID"
SKILL_DIR="$HERE/skills/$SKILL_ID"
"${DB[@]}" api post "/api/2.1/unity-catalog/skills?parent=schemas/$CATALOG.$SCHEMA&skill_id=$SKILL_ID" \
  --json '{"comment":"Writes and tests GRPO reward (scoring) functions for verl custom datasets"}' >/dev/null 2>&1 || true
TOKEN=$("${DB[@]}" auth token -o json | python3 -c "import json,sys;print(json.load(sys.stdin)['access_token'])")
HOST=$("${DB[@]}" auth describe 2>/dev/null | awk '/^Host:/{print $2}')
for f in "$SKILL_DIR"/*; do
  [ -f "$f" ] || continue   # skip __pycache__ etc.
  code=$(curl -s -o /dev/null -w "%{http_code}" -X PUT -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/octet-stream" \
    --data-binary @"$f" "$HOST/api/2.0/fs/files/Skills/$CATALOG/$SCHEMA/$SKILL_ID/$(basename "$f")?overwrite=true")
  [ "$code" = "204" ] || [ "$code" = "200" ] || { echo "  upload $(basename "$f") failed ($code)"; exit 1; }
done
"${DB[@]}" api post "/api/2.1/unity-catalog/skills/$CATALOG.$SCHEMA.$SKILL_ID/finalize" --json '{}' >/dev/null
echo "  published ($(find "$SKILL_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ') files)"

say "ビルド（テンプレートのコピーと app.yaml の生成）"
rm -rf "$BUILD" && mkdir -p "$BUILD/templates"
cp "$HERE/app.py" "$HERE/custom_flow.py" "$HERE/reward_agent.py" "$HERE/requirements.txt" "$BUILD/"
cp -R "$HERE/static" "$BUILD/"
for t in smoke_test.yaml prep_data_job.json grpo_gemma4.yaml grpo_gemma4_multinode.yaml \
         grpo_gemma4_mm.yaml grpo_gemma4_mm_multinode.yaml run_grpo.sh run_grpo_multinode.sh \
         grpo_gemma4_custom.yaml grpo_gemma4_custom_multinode.yaml grpo_gemma4_mm_custom.yaml \
         grpo_gemma4_mm_custom_multinode.yaml custom_sample_raw_job.json custom_reward.py; do
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
  - name: AIR_LLM_ENDPOINT
    value: "$LLM_ENDPOINT"
  - name: AIR_SKILL
    value: "$CATALOG.$SCHEMA.$SKILL_ID"
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
