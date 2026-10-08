# ハンズオン用 UI（Databricks App）

このリポジトリの手順（データ準備 → 学習 → 監視 → 停止）を、**ブラウザのボタン操作だけで**進めるための
Databricks App です。講師が画面を見せながら操作するデモ用を想定しています。

## できること
1. **前提チェック**：Databricks CLI、アプリの実行 ID、学習用イメージ（Artifact Registry のタグ・ダイジェスト・
   アクセラレーションの状態）、データ用 Volume、データ（gsm8k / geo3k）の有無を一覧で確認
2. **準備**：データ準備（gsm8k + geo3k。**CPU のサーバーレス Jobs なので GPU 不要**）、疎通確認（1×A10）をボタンで投入。Volume の作成もボタンで
3. **学習**：GRPO テキスト / 画像 × 単一ノード / 2ノードをボタンで投入
4. **実行状況**：ジョブの一覧（状態・開始時刻・投入者）、ジョブ実行画面と MLflow へのリンク、経過時間、
   5秒ごとに更新されるログ、**停止ボタン**（GPU の課金を止める）

## 仕組み
- アプリの中で、ハンズオンと同じ `databricks air` CLI（v1.19.0。起動時に GitHub から取得）を呼びます。
  投入する YAML とスクリプトは、リポジトリ直下のテンプレートを `deploy.sh` がコピーしたものを、
  `quickstart.sh` と同じ方法で置き換えて使います（二重管理はしません）。
- **ジョブはアプリのサービスプリンシパル（SP）の権限で投入されます。** Databricks Apps の
  「ユーザーの代理で実行する」機能のスコープには、Jobs / MLflow / AI Runtime の API が含まれないためです。
  代わりに、各ジョブに `permissions` で、画面を操作した人の `CAN_MANAGE` を付けています。
- MLflow の実験は `/Workspace/Shared/air-handson/` に作られます（全員が開けるように）。データ準備のスクリプト
  `prep_data.py` も `deploy.sh` がここに置き、CPU のサーバーレス Jobs として実行します（GPU が使えないときでも動きます）。
- ログは `air logs --tail` で取得します（`--download-to` は、サーバーレス環境から MLflow のファイル置き場に
  接続できないため、アプリの中では使えません。完全なログは MLflow のリンクから見られます）。

## デプロイ
前提：学習用イメージが Artifact Registry に `<catalog>.<schema>.verl-gemma4:<tag>` として取り込み済み
（setup.md §3）であること。デプロイする人には、アプリの作成と、そのカタログ・スキーマへの GRANT の権限が必要です。
```bash
PROFILE=<PROFILE> CATALOG=<catalog> SCHEMA=<schema> bash handson-app/deploy.sh
# 任意: APP_NAME=air-handson VOLUME=air_handson IMAGE_NAME=verl-gemma4 IMAGE_TAG=v1
```
`deploy.sh` は次を行います：アプリの作成（初回のみ）→ SP への権限付与（カタログに `USE_CATALOG`、
スキーマに `USE_SCHEMA` / `READ_VOLUME` / `WRITE_VOLUME` / `CREATE_VOLUME`）→ `/Workspace/Shared/air-handson`
の作成 → テンプレートのコピーと `app.yaml` の生成 → アップロードとデプロイ。最後にアプリの URL が表示されます。

コードを直したら、同じコマンドを再実行すれば再デプロイされます。

## 使い終わったら
Databricks Apps は**起動している間ずっと課金されます**。使わないときは止めてください。
```bash
databricks apps stop air-handson -p <PROFILE>     # 再開は databricks apps start
```
GPU のジョブは、画面の「停止」ボタン、または `databricks air cancel <RUN_ID>` で止めます。

## 注意
- 画面の「実行状況」に出るのは、**このアプリ（SP）が投入したジョブだけ**です。
- ジョブの種類や投入者の表示はアプリのメモリに持っているので、アプリを再起動すると、
  それ以前のジョブは実験名で表示されます（ジョブ自体には影響しません）。
- 2ノードのジョブは 8×H100 を2台使います。ワークスペースの GPU クォータに注意してください。
