# Gemma4-26B-A4B を verl + FSDP2 で GRPO 学習する環境構築ガイド

Databricks AI Runtime のサーバーレス GPU 上で、**Gemma4-26B-A4B**（MoE + ハイブリッド注意 +
マルチモーダルのモデル）を **verl + FSDP2** で **GRPO（強化学習）** するための環境を、
**ゼロから構築する**ための手順書です。ローカル PC 上のコマンドラインツール（`air` CLI）から
学習ジョブを投入します。

このガイドは既存の成果物やコンパイル済みファイルの持ち込みを前提としません。すべて
このガイドの手順の中で、ソースから構築します。

> **実機検証済み**（単一ノード 8×H100 / マルチノード 2ノード 16×H100 の両方で
> `Training Progress 100% (3/3)` → `Job status: SUCCESS`）。本ガイドの構成はその結果に基づきます。

---

## 0. 前提と全体像

### 0-0. 対象モデルと「6つの勘所」（重要・最初に読む）
Gemma4-26B-A4B は **MoE（128エキスパート中8アクティブ、共有なし）+ ハイブリッド注意
（sliding + full）+ マルチモーダル** の新しいモデルです。verl で GRPO するには、Gemma4 固有の
以下6点への対処が必要です（すべて本ガイドの成果物に反映済み。詳細は各節と付録B）:

1. **`-it`（instruction-tuned）モデルを使う**。`google/gemma-4-26B-A4B`（base）は
   **chat_template を持たない**ため、チャット形式データの GRPO が回りません
   → `google/gemma-4-26B-A4B-it` を使用（Apache-2.0・gated ではない＝HFトークン不要）。
2. **verl の FSDP2 buffer broadcast パッチ**が必要。verl 0.7.1 は full state_dict ロード時に
   buffer をソートせず broadcast するため、Gemma4 の**非均質な rotary buffer（256要素と128要素）**で
   NCCL デッドロックする → 起動時に `fsdp_utils.py` を実行時パッチ（`run_grpo.sh` に実装済み）。
3. **NCCL タイムアウトを延長**する。25.8B の MoE は state_dict のロード/シャードに 600秒（既定）
   以上かかり、初期化中の集合通信がタイムアウトする → `nccl_timeout=3600`（`run_grpo.sh` に設定済み）。
4. **注意機構は SDPA を使う**。Gemma4 は `global_head_dim=512` で、FlashAttention2 の
   「head_dim 最大256」制約を超えるため FA2 は使えない → SDPA（`run_grpo.sh` の既定）。
5. **学習データはテキストのみ（gsm8k）** を既定にする。verl 0.7.1 は Gemma4 の画像 processor
   （`Gemma4Processor`）に未対応で、画像入りデータ（geo3k 等）はメッセージ構築で失敗する。
6. FlashAttention のソースビルドは**不要**（SDPA のため）。Qwen 系ガイドのような
   「FA wheel ビルド」工程はありません → イメージ作成は **1回のビルドで完結**。

### 0-1. 作業環境（ローカル PC）
- **OS**: macOS または Linux（x86_64 / arm64 どちらでも可）
- 必要なツール:
  - **Git**（`git clone` でソース一式を取得）
  - **Databricks CLI**（`databricks`）
  - **AI Runtime CLI**（`air`）
  - **Docker**（`docker build` / `docker push` が使えること。Docker Desktop など）
  - **Python 3.12**（ローカルでの補助スクリプト用。必須ではない）
- **GPU はローカルに不要**（学習は Databricks 側の H100 上で実行）。

### 0-2. Databricks 側の前提
- **AI Runtime（サーバーレス GPU）が有効なワークスペース**。
  現時点で AI Runtime は **AWS / Azure の US リージョン**でのみ提供。
- アクセラレータタイプ **`GPU_8xH100`** が利用可能であること。
- 学習データとチェックポイントを置く **Unity Catalog Volume** を作成できる権限。
- Docker イメージを登録するための **Docker Hub アカウント**（AI Runtime のカスタムイメージは
  Docker Hub のみ対応、イメージサイズは 20GB 未満という制約がある）。

### 0-3. なぜ「カスタム Docker イメージ」方式なのか（重要）
本構成では、学習に必要なライブラリ一式を **1 個のカスタム Docker イメージに固めます**。
`air` の「依存関係をジョブ実行時にインストールする方式（`environment.dependencies`）」では
**この学習は動きません**。理由:

1. verl の公開 wheel は `numpy<2` かつ `vllm<=0.12` を要求するが、Gemma4 に必要な
   vLLM は 0.24（gemma4 対応）。依存解決が原理的に成立しない（後述の `--no-deps` で回避する）。
2. Gemma4（`gemma4` アーキ）には **transformers ≥5.5.3** が必要で、これは vLLM 0.24 /
   torch 2.11（cu13）と揃える必要がある。実行ノードのベース環境（cu12.9）とは食い違う。

→ **CUDA 13 の devel ベースイメージ**の上に必要物を固めた**カスタムイメージ**を作るのが確実な方法です。

### 0-4. 作業の流れ
1. **ソース一式を Git リポジトリから取得（clone）する**（§2）
2. CLI と認証、UC Volume を用意する（§1）
3. **学習環境イメージを作る**（§3）。**1回のビルドで完結**（FA ビルド不要）。push → 登録。
4. **動作確認用データ（gsm8k, テキスト）を用意**（§4-2）
5. 単一ノード（8×H100）で GRPO 学習を実行（§4）
6. マルチノード（2ノード = 16×H100）で GRPO 学習を実行（§5）

> §1（CLI・認証）と §2（clone）は順不同です。本ガイドはまず §2 でソースを取得し、
> その中の `quickstart.sh`／各ファイルを使う前提で §1 以降を説明します。

> **ラクをしたい場合（推奨）**: §1後半〜§3 は付属の **`quickstart.sh`** が一括で実行します。
> 対話で数項目を入力するだけで、Volume 作成 → イメージ ビルド/push/登録 まで自動で進みます（§0-5）。

### 0-5. クイックスタート（`quickstart.sh`）
§1後半〜§3 を自動化したスクリプトです。**事前に** 以下だけ済ませておいてください:
- ソース一式のクローン（§2）— `quickstart.sh` はクローンしたディレクトリの中で実行します
- CLI 導入（`git` / `databricks` / `air` / `docker`）
- `databricks auth login --host <URL> --profile <PROFILE>`（ブラウザ認証）
- `docker login`（Docker Hub）

```bash
cd verl-gemma4        # git clone したディレクトリ（§2）
bash quickstart.sh
```
対話で `PROFILE / Docker Hub ユーザー名 / イメージタグ / カタログ / スキーマ / Volume名 /
メールアドレス / pip インデックス` を入力すると、以下を順に実行します:
1. UC Volume 作成
2. テンプレート（`*.yaml` / `*.sh`）の置換 → `*.gen` を生成
3. カスタムイメージをビルド（1回）→ push → 登録

完了後、`prep_gsm8k_deps.yaml.gen` でデータを用意し、`grpo_gemma4.yaml.gen` /
`grpo_gemma4_multinode.yaml.gen` を `air run` すれば学習できます（§4・§5）。

> 学習の実行（§4・§5）とデータ準備は quickstart には含めていません（パラメータを変えて
> 何度も回すものなので手動運用が適切）。イメージの push/登録に数十分かかります。

---

## 1. CLI・認証・置き場所の準備

### 1-1. CLI のインストール（ローカル PC）
```bash
# Databricks CLI（未導入の場合は公式手順で導入）
databricks version        # 例: Databricks CLI v0.297.2 以降を推奨

# AI Runtime CLI
uv tool install --force databricks-air --python 3.12
air --version             # 例: v1.0.0
```

### 1-2. ワークスペースへ認証
```bash
databricks auth login --host https://<ワークスペースURL> --profile PROF
databricks current-user me -p PROF     # 疎通確認
```
> 以降のコマンド例の `PROF` は自分のプロファイル名に置き換えてください。

### 1-3. Docker Hub へログイン
```bash
docker login docker.io       # 使用する Docker Hub アカウントで
```
> 以降、イメージ名の `<DOCKERHUB_USER>` は自分の Docker Hub ユーザー名に置き換えます。

### 1-4. UC Volume の作成（データ・チェックポイント置き場）
```bash
databricks volumes create <catalog> <schema> verl_workspace MANAGED -p PROF
```
→ 作成されたパス `/Volumes/<catalog>/<schema>/verl_workspace` を控える。
以降このパスを **`$VOL`** と表記します。

---

## 2. ソース一式の取得（Git clone）

ソース一式（Dockerfile・スクリプト・YAML・本ガイド）は Git リポジトリで配布されます。
まずローカル PC にクローンし、以降の作業はそのディレクトリの中で行います。
```bash
git clone <REPO_URL> verl-gemma4
cd verl-gemma4
```
> `<REPO_URL>` は配布された Git リポジトリの URL に置き換えてください。リポジトリが
> **非公開（private）**の場合は、事前に閲覧権限の付与とアクセス認証（GitHub なら
> Personal Access Token または SSH 鍵、`gh auth login` 等）が必要です。
> Git を使わず ZIP で受け取った場合は、展開して `cd` するだけで同じです。

クローンすると以下のファイルが揃います。

| ファイル | 役割 |
|---|---|
| `setup.md` | 本ガイド |
| `README.md` | 英語の概要 |
| `quickstart.sh` | §1後半〜§3 を自動化するスクリプト（§0-5） |
| `Dockerfile` | カスタムイメージ定義（CUDA13ベース + 全ライブラリ） |
| `run_grpo.sh` | 単一ノード GRPO 起動スクリプト（Gemma4 向け設定込み） |
| `run_grpo_multinode.sh` | 2ノード用 GRPO 起動スクリプト（Ray クラスタ形成込み） |
| `grpo_gemma4.yaml` | 単一ノード（8×H100）テキストの `air` ジョブ定義 |
| `grpo_gemma4_multinode.yaml` | 2ノード（16×H100）テキストの `air` ジョブ定義 |
| `grpo_gemma4_mm.yaml` | 単一ノード **マルチモーダル（画像）** の `air` ジョブ定義（§6） |
| `grpo_gemma4_mm_multinode.yaml` | 2ノード **マルチモーダル（画像）** の `air` ジョブ定義（§6） |
| `prep_gsm8k_deps.yaml` | 動作確認用の小さな学習データ（gsm8k, テキスト）を用意するジョブ |
| `smoke_test.yaml` / `smoke_test.py` | 依存疎通確認（1×A10、安価） |

各ファイルには環境依存の値がプレースホルダで入っています。`quickstart.sh` を使う場合は
対話入力から自動で置換されます（§0-5）。手動で進める場合は、次の値を自分の環境に合わせて
置換してください:
- `__IMAGE__` / `<DOCKERHUB_USER>/verl-gemma4`（学習イメージ名）
- `__VOL__` / `$VOL`（UC Volume パス）
- `__WS_EMAIL__` / `/Workspace/Users/<自分のメール>/...`（MLflow 実験ディレクトリ）
- 各コマンド例の `PROF`（Databricks CLI プロファイル名）

---

## 3. 学習環境イメージの作成

学習に必要なライブラリ一式（torch / vLLM 0.24 / transformers / verl 等）を **1 個のカスタム
Docker イメージ**に固めます。`quickstart.sh` を使えばこの §3 が自動化されます（§0-5）。
以下は手動で行う場合の手順です。

### 3-1. ビルドの考え方
- ベースは AI Runtime 公式の **CUDA 13 devel イメージ** `databricksruntime/air:dcs-base-aws-devel-cu13`。
- その上に**確定バージョン**（§付録A）で torch 2.11 / vLLM 0.24 / transformers / verl などを入れます。
- **Gemma4 は SDPA を使うため FlashAttention は不要**。したがって Qwen 系のような「FA wheel を
  H100 で作る」工程はなく、**イメージ作成は 1回のビルドで完結**します。
  （`Dockerfile` は `BUILD_FLASH_ATTN=0` で FA install をスキップします。）

### 3-2. イメージをビルド → push → 登録
```bash
# クローンしたディレクトリ（§2 の verl-gemma4）の中で実行します
mkdir -p wheelhouse        # 空でよい（Dockerfile が --find-links /wheelhouse を使うため必須）
docker build --platform linux/amd64 \
  --build-arg PIP_INDEX_URL=https://pypi.org/simple \
  --build-arg BUILD_FLASH_ATTN=0 \
  -v "$PWD/wheelhouse:/wheelhouse:ro" \
  -t docker.io/<DOCKERHUB_USER>/verl-gemma4:v1 -f Dockerfile .
docker push docker.io/<DOCKERHUB_USER>/verl-gemma4:v1
air register image <DOCKERHUB_USER>/verl-gemma4:v1 -p PROF
```
- **`wheelhouse/` ディレクトリは必ず作成してマウント**してください（空で構いません）。`Dockerfile`
  の各 install 手順は `--find-links /wheelhouse` を使うため、`/wheelhouse` が**存在しないと
  ビルドが即失敗**します（空なら各 wheel は `PIP_INDEX_URL` から取得されます）。
- torch(≈530MB)・vLLM(≈270MB) 等の大きな wheel を取得するため、**ネットワークの安定した環境**で。
- `--platform linux/amd64` … AI Runtime ノードは x86_64。arm Mac でも本指定で amd64 イメージに
  なります（エミュレーションで時間がかかる場合あり。可能なら x86_64 Linux 上でのビルドを推奨）。
- `Image registered: sha256:...` で登録完了。この `:v1` が **学習に使うイメージ**です。
- Docker の `build` で `-v`（ビルド時マウント）が使えない場合は、`wheelhouse/` を `COPY` するか
  `RUN --mount=type=bind,source=wheelhouse,target=/wheelhouse` に切り替えてください。

> **イメージサイズに注意**: 20GB 未満に収める必要があります（AI Runtime の登録制約）。
> Dockerfile には `UV_NO_CACHE=1` を設定済みです。

**ここまでで学習可能な状態です。**

---

## 4. 単一ノード（8×H100）で学習

### 4-1. 疎通確認（1×A10、安価。推奨）
```bash
air run --file smoke_test.yaml -p PROF --watch
```
→ `SMOKE TEST: PASS`（torch / transformers / vllm / verl / gemma4 認識 が OK）を確認。

### 4-2. 動作確認用データの準備（gsm8k, テキスト）
```bash
air run --file prep_gsm8k_deps.yaml -p PROF --watch
```
→ `$VOL/gsm8k/{train,test}.parquet`（64 train / 8 test の少量データ）が作られます。
本番は実データに差し替えます（データ形式は verl の gsm8k 形式に準拠）。

> **なぜ gsm8k（テキスト）か**: verl 0.7.1 は Gemma4 の画像 processor（`Gemma4Processor`）に
> 未対応で、画像入りデータ（geo3k 等）はメッセージ構築時に
> `AssertionError: processor is needed to process image and video` で失敗します。テキストのみの
> gsm8k なら画像 processor が不要で、MoE + FSDP2 + GRPO の疎通を確認できます（§付録B）。

### 4-3. 単一ノード（8×H100）で GRPO を実行
```bash
air run --file grpo_gemma4.yaml -p PROF --watch
```
→ `Training Progress: 100% (3/3)` → `Job status: SUCCESS` で学習が回っています
（既定で SDPA + `use_remove_padding=False`）。実機では step ごとに
`critic/rewards/mean`・`grad_norm` 等のメトリクスが出ます。

本番学習への切り替え:
- `prep_gsm8k_deps.yaml` の `MAXN` を増やす／実データに差し替え
- `grpo_gemma4.yaml` の `parameters.total_training_steps` を増やす／エポック学習へ
- `run_grpo.sh` の `train_batch_size` / `max_response_length` を本番規模へ

---

## 5. マルチノード（2ノード = 16×H100）

### 5-1. 単一ノードとの違い
verl は複数ノードを **Ray クラスタ**で束ねます（torchrun ではありません）。AI Runtime では
ノード間で Ray が自動接続しないため、ジョブの `command` の中で **Ray のヘッド/ワーカーを
明示的に起動**する必要があります。`num_accelerators` を増やすだけでは動きません。
この処理は `run_grpo_multinode.sh` に実装済みです（`NODE_RANK` で分岐し、ヘッドが
`ray start --head`、ワーカーが `ray start --address` で参加）。

| 設定 | 単一ノード | 2ノード |
|---|---|---|
| `compute.num_accelerators` | 8 | 16（= 合計 GPU 数。ノード数は 16/8=2 に自動導出） |
| `compute.accelerator_type` | GPU_8xH100 | GPU_8xH100 |
| 起動スクリプト | `run_grpo.sh` | `run_grpo_multinode.sh`（Ray 起動込み） |
| `trainer.nnodes` | 1 | 2 |
| `trainer.n_gpus_per_node` | 8 | 8 |
| `train_batch_size` | 8 | **16**（16 GPU の最小バッチ単位 16 で割り切れる必要あり） |
| `fsdp_size` | 8 | 8（ノード内シャード）または 16（全ノードシャード） |
| vLLM tensor parallel | ≤8 | ≤8（ノード内に限定） |

### 5-2. 実行
```bash
air run --file grpo_gemma4_multinode.yaml -p PROF --watch
```
成功時、ログに以下が順に出ます:
- 2ノードで Ray クラスタが形成される（`nRanks 2 nNodes 2`）
- `Training Progress: ... 3/3` → `Job status: SUCCESS`
- 学習完了時に `[head] training finished (rc=0); stopping Ray head` が出て、ワーカーも自動終了。

### 5-3. 注意点
- 16 GPU では verl の最小バッチ単位が 16 になります。`train_batch_size × rollout.n` が
  16 の倍数になるようにしてください（既定は `train_batch_size=16`, `rollout.n=5` → 80、OK）。
- `train_batch_size` 以上の学習データ件数が必要です（動作確認データは十分な件数にしておく）。
- ワーカーノードはヘッドの Ray クラスタに参加後、学習完了（ヘッドの GCS ポート消失）を
  検知して自動終了します（`run_grpo_multinode.sh` に実装済み。§付録B）。

---

## 6. マルチモーダル（画像入り）で学習する

§4・§5 は既定でテキストのみ（gsm8k）です。Gemma4 は**マルチモーダル（画像＋テキスト）**モデルなので、
**画像入りデータ（geo3k 等）**でも GRPO できます。ただし追加の要件があります（実機検証済み・単ノード
8×H100／マルチノード 2ノード16×H100 の両方で `Training Progress 100% (3/3)`）。

### 6-1. なぜ追加要件が要るか（重要）
- **verl のリリース版（0.8.0 含む）は Gemma4 の画像プロセッサ（`Gemma4Processor`）に未対応**で、
  画像データを渡すと `Unsupported processor type: Gemma4Processor` /
  `processor is needed to process image and video` で失敗します。
  Gemma4 画像対応は **verl の `main` ブランチ**に入っています（PR #4759。リリース未反映）。
  → **`MULTIMODAL=1` を指定すると `run_grpo.sh` が起動時に verl main を `--no-deps` で入れ替え**ます
  （併せて main の新依存 `TransferQueue` も導入。transformers 5.14 等は維持）。
- 画像は **vision タワー + 多数の画像トークン**でメモリを多く使うため、単ノードではメモリ調整が必要
  （下記 6-3）。

### 6-2. 実行（単ノード / マルチノード）
画像入りデータ（verl の geo3k 形式、`images` 列を持つ parquet）を Volume に用意した上で:
```bash
# 単ノード（8×H100）
air run --file grpo_gemma4_mm.yaml -p PROF --watch
# マルチノード（2ノード = 16×H100）
air run --file grpo_gemma4_mm_multinode.yaml -p PROF --watch
```
`grpo_gemma4_mm*.yaml` は `MULTIMODAL=1` / `IMAGE_KEY=images` と、下記のメモリ設定を `env_variables`
で渡します。成功時 `Training Progress 100% (3/3)` → `Job status: SUCCESS`。

### 6-3. マルチモーダル固有のメモリ設定（実機で確定）
単ノード 8×H100 に 25.8B MoE + vision + vLLM を同居させるとメモリが逼迫します。以下は実機検証で
確定した設定です（`grpo_gemma4_mm.yaml` に既定値。すべて `run_grpo.sh` が env で受け取り）:

| 設定 | 値 | 理由 |
|---|---|---|
| `ROLLOUT_GPU_MEM_UTIL` | 0.5 | verl 推奨 0.5–0.7。**0.4 は低すぎて vLLM init が落ちた** |
| `ENFORCE_EAGER` | True | CUDA グラフ捕捉（起動時に数 GiB 消費）を無効化 |
| `ACT_OFFLOAD` | True | 活性値を CPU offload → actor 更新のピークメモリ削減 |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | **画像1枚 = 2496 トークン**を下回ると vLLM が拒否する |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH` / `ROLLOUT_N` | 単ノード 4/4/2、2ノード 8/8/2 | actor 更新 OOM をバッチ削減で回避。16GPU は積 ×n が 16 の倍数要件 |
| `MAX_PROMPT_LEN` / `MAX_RESPONSE_LEN` | 768 / 512 | 画像でプロンプトが伸びるため短縮 |

> 本番のメモリ・スループット最適化はデータや画像解像度で変わります。上記は「小データで動くこと」を
> 確認した保守的な値です。

### 6-4. 既知の課題（正直な注記）
- 上記の疎通確認では学習は完走しますが、**報酬が 0 のまま**でした
  （`critic/rewards/mean = 0`）。geo3k の報酬関数が Gemma4 の応答フォーマット（`\boxed{}` 抽出など）を
  拾えていないためと考えられます（テキストの gsm8k では報酬 0.7〜0.97 が出ます）。
  **実学習として意味を持たせるには、データのプロンプト整形／報酬関数を対象モデルの応答に合わせて
  調整**してください（パイプラインの疎通自体は成立しています）。

---

## 付録A: 確定バージョン（再現性のため全明記）

### 実行基盤（Databricks 側）
| 項目 | バージョン |
|---|---|
| AI Runtime CLI (`air`) | v1.0.0 |
| Databricks CLI | v0.297.2 |
| ベースイメージ | `databricksruntime/air:dcs-base-aws-devel-cu13` |
| CUDA (nvcc) | 13.0.88 |
| Python | 3.12.3 |
| アクセラレータ | `GPU_8xH100`（単一ノード8枚／2ノードで16枚） |

### イメージに固める Python パッケージ（実測・固定推奨）
| パッケージ | バージョン |
|---|---|
| torch | 2.11.0（cu13） |
| vllm | 0.24.0（gemma4 対応。`Gemma4ForConditionalGeneration` を含む） |
| transformers | 5.14.1（`gemma4` は 5.5.3 以降で対応） |
| verl | 0.7.1（`--no-deps` でインストール。起動時に fsdp_utils.py をパッチ＝付録B） |
| ray | 2.56.1 |
| opencv-python-headless | **4.12.0.88（固定）** |
| mathruler | 0.1.0（gsm8k 等の報酬計算） |
| numpy | 2.2.6 |

> **注**: flash-attn は**使いません**（Gemma4 は SDPA）。causal_conv1d / flash-linear-attention は
> Gemma4 には不要ですが、Qwen と共通の `Dockerfile` を流用しているため入っています（無害。
> スリム化する場合は Dockerfile から外せます）。

### 対象モデル
| 項目 | 値 |
|---|---|
| モデル | `google/gemma-4-26B-A4B-it`（Hugging Face、Apache-2.0・公開・**トークン不要**） |
| アーキ | `gemma4`（`Gemma4ForConditionalGeneration`、decoder 層 `Gemma4TextDecoderLayer`） |
| 構成 | MoE（128エキスパート/8アクティブ・共有なし）+ ハイブリッド注意（sliding+full）+ マルチモーダル |
| 主要諸元 | hidden 2816 / head_dim 256 / global_head_dim 512 / vocab 262144 / text 30層 |

---

## 付録B: 設計上の要点（Gemma4 固有のつまずき回避）

実機検証で判明した Gemma4 固有の対処（すべて成果物に反映済み。§0-0 の要約の詳細版）:

- **base ではなく `-it` モデルを使う**。base（`google/gemma-4-26B-A4B`）は tokenizer に
  chat_template が無く、`apply_chat_template` が `ValueError: tokenizer.chat_template is not set`
  で失敗する。`-it` は chat_template 同梱。
- **verl 0.7.1 の FSDP2 buffer broadcast デッドロックをパッチ**する。verl 0.7.1 の
  `fsdp2_load_full_state_dict` は `model.named_buffers()` を**ソートせず** broadcast する。FSDP2 は
  buffer をランク毎に異なる順序で返しうるため、Gemma4 の**非均質な rotary buffer（256要素と128要素）**
  で「同じ collective に異なるサイズ」→ NCCL デッドロック（初期化中に無限ハング）。
  `run_grpo.sh` は起動時に `fsdp_utils.py` を `sorted(model.named_buffers(), key=...)` に書き換える
  （冪等）。verl 本体の新しい版では修正済み。
- **NCCL タイムアウトを 3600 秒に延長**する。25.8B の MoE は full state_dict のロード/シャードが
  既定の 600 秒を超え、初期化中の集合通信が watchdog に落とされる → `actor_rollout_ref.nccl_timeout=3600`
  （`run_grpo.sh` に設定済み）。
- **注意機構は SDPA**。Gemma4 は `global_head_dim=512` で、FlashAttention2 forward の
  「head_dim 最大256」制約を超えるため `RuntimeError: FlashAttention forward only supports head
  dimension at most 256` になる。`run_grpo.sh` は既定で
  `override_config.attn_implementation=sdpa` + `use_remove_padding=False`。
- **学習データはテキストのみ（gsm8k）を既定**にする。verl 0.7.1 の processor 工場は
  Qwen/GLM/Mllama 系のみ許可し、Gemma4 の `Gemma4Processor` を
  `Unsupported processor type` で弾く。画像入りデータ（geo3k）は
  `AssertionError: processor is needed to process image and video` で失敗する。画像経路が必要な
  場合は verl 側で Gemma4 の vision processor 対応（`get_rope_index` の bind 等）が必要。
- **verl は `--no-deps` でインストール**する（公開 wheel の `numpy<2` / `vllm<=0.12` 制約を回避。
  実行時依存は Dockerfile 側で個別に明示インストール済み）。
- **opencv-python-headless は 4.12.0.88 に固定**。5.0 系はバンドルする libcrypto が FIPS を強制し、
  `import cv2` の時点でプロセスがクラッシュする（Dockerfile で対処済み）。
- **FSDP2 のラップ対象**を明示指定: `fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer]`
  （`run_grpo.sh` に設定済み）。
- **actor→vLLM の重み同期バケット**を拡大: `rollout.checkpoint_engine.update_weights_bucket_megabytes=6144`
  （Gemma4 の埋め込みが大きいため。`run_grpo.sh` に設定済み）。
- **マルチノードは Ray クラスタの明示起動が必要**（§5。`run_grpo_multinode.sh` に実装済み）。
