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

## 「自分のデータで学習」タブ
CUSTOM_DATA.md の手順を、画面だけで進められるようにしたものです。上から順に1本道で進み、
各手順のボタンは基本的に1つです（前の手順が終わるまで、次の手順は押せません）。

| 手順 | 画面でやること | 裏で動くもの |
|---|---|---|
| ① どのデータを使うか | 「自分のデータ」「練習用のサンプル」「以前に変換したデータ」から選ぶ。自分のデータは、1行1問の **JSONL**（＋画像フォルダ）を Volume に置き（画面に形式の例、カタログエクスプローラーへのリンク、`databricks fs cp` のコマンドあり）、パスを入れて「読み込む」 | サンプルは `make_sample_raw.py`（CPU のサーバーレス Jobs） |
| ② 確認して変換 | 項目（質問・正解・画像）は自動で割り当て、「モデルにはこう渡されます」を自動表示。「学習用に変換する」で JSONL を parquet にし、学習用と確認用（8問）に分ける。以前のデータを選んだ場合は不要 | `prep_custom.py`（CPU のサーバーレス Jobs） |
| ③ 採点のしかた | 「数値で比べる（標準）」「文字が完全に同じなら正解」「基準を書いて AI に作ってもらう」から1つ選ぶ。どれでも自動で確認テスト（データの問題を AI に解かせ、付いた点数を見せる）が走るので、点数を見て「この採点で決定」 | AI に作ってもらう場合：**UC のスキル**（`grpo-reward-author`）を読んだ **Claude**（Foundation Model API）が採点関数とテストを書き、**Databricks Sandbox** でテスト。失敗したら直す（LangGraph のループ） |
| ④ 学習 | 単一ノード / 2ノードを選んで「学習を開始」 | 独自データ用の YAML（決定した採点関数を使う） |
| ⑤ 様子を見る | 状態・リンク・ログと、学習中の各ステップの**回答と点数**（最初のステップが終わると表示） | `trainer.rollout_data_dir` の書き出し |

- 確認テストは、データの問題を Claude に解かせ（画像のデータは画像も渡します）、その回答に採点関数が付けた点数を見せます。
  点数が妥当かを人が確かめてから決定してください。採点関数のコードは「上級者向け」に畳んであります。
- 練習用のサンプルは、テキスト（算数の文章題）と画像（図形の問題）の JSONL を `<Volume>/raw/` に作ります。
- 使うもの：`deploy.sh` が、スキルの Unity Catalog への公開、LLM エンドポイント（既定 `databricks-claude-sonnet-5-5`）の
  `CAN_QUERY` 付与まで行います。ワークスペースの Previews で **Sandbox** と **UC の Skills** が有効である必要があります。

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
