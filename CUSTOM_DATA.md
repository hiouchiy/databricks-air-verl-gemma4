# 自分のデータで GRPO 学習する（カスタムデータセット）

このリポジトリのサンプルは、公開データセットの **gsm8k（テキストの算数問題）** と **geo3k（図形の画像付き問題）** を
使っています。ここでは、**お客様が持っているデータ**で同じ学習（Gemma 4 × verl GRPO）を回す方法を説明します。

学習スクリプト（`run_grpo.sh`）や学習用イメージは**そのまま使えます**。用意するものは次の2つです。

1. **データ**：お客様のデータを、verl が読める形式（parquet）に変換したもの
2. **採点関数**：モデルの答えが正しいかを判定する、小さな Python の関数

> **実機検証済み（2026-10-13）**：独自の `data_source`（`acme/math-qa` / `acme/geometry`）で、CSV（テキスト）と
> JSONL ＋ 画像（画像）の生データから変換し、ひな形の採点関数で学習しました。テキスト（単一ノード 8×H100）・画像
> （単一ノード 8×H100）とも `Training Progress 100% (3/3)` → `SUCCESS`（報酬 mean：テキスト 1.0 / 0.975 / 1.0、
> 画像 0.5 / 0.75 / 0.5）。データの変換は CPU のサーバーレス Jobs で各1分前後でした。

> 「データさえ用意すれば学習できる」と思われがちですが、強化学習では **2. の採点関数が欠かせません**。
> まず次の「背景」を読んでから、手順に進んでください。

---

## 1. 背景：なぜ「採点関数」が必要なのか

### 教師あり学習（SFT）と強化学習（GRPO）の違い
| | 教師あり学習（SFT） | 強化学習（このリポジトリの GRPO） |
|---|---|---|
| データに入っているもの | 質問と、**お手本の回答文** | 質問と、**正解（答えだけ）** |
| モデルが学ぶこと | お手本の回答文を**まねる** | 自分で考えて答えを出し、**点数が高くなる答え方を身につける** |
| 何が「良い答え」かを決めるもの | お手本の回答文そのもの | **採点関数**（答えに点数を付ける関数） |

GRPO の1回の学習ステップでは、次のことが起きています。

1. データから質問を取り出す
2. モデルが、その質問に対して**答えを何通りか**書く（このリポジトリの既定は5通り）
3. **採点関数**が、それぞれの答えに点数を付ける（例：正解なら 1.0、不正解なら 0.0）
4. 同じ質問の中で、**点数が高かった答え方が出やすくなるように**モデルを更新する

つまり、モデルにとって「何が良い答えか」を教えているのは、**データそのものではなく採点関数**です。
採点関数がなければ、モデルは何を目指して学習すればよいか分かりません。データにある正解（`ground_truth`）は、
採点関数が答え合わせに使う**材料**にすぎません。

### なぜ自分で用意する必要があるのか
verl は、データの **`data_source` 列の値**を見て、どの採点関数を使うかを決めます。組み込みの採点関数が
用意されているのは、`openai/gsm8k` や `hiyouga/geometry3k` など**有名な公開データセットだけ**です。

お客様のデータの `data_source`（例：`acme/math-qa`）は verl が知らない名前なので、そのままでは学習の最初の
採点で次のエラーになり、止まります。
```
NotImplementedError: Reward function is not implemented for data_source='acme/math-qa'
```
そのため、**「このデータはこの関数で採点してください」と verl に教える**必要があります。それが
`custom_reward.py`（採点関数）と、YAML に足す2行の設定です。

### 採点関数の考え方
- 答えが1つに決まる問題（数値、選択肢、短い単語など）が向いています。このリポジトリのひな形は、
  「モデルの答えの最後の `\boxed{...}` を取り出し、正解と比べて 1.0 か 0.0 を返す」ものです。
- 採点のしかたが学習の方向そのものを決めます。**甘すぎる採点**（何でも正解にする）や**厳しすぎる採点**
  （表記ゆれで不正解になる）では、うまく学習しません。お客様のデータの答え方に合わせて調整してください。
- 指示文（「最後に答えを `\boxed{}` に書いてください」など）と採点関数は**セット**です。指示した書き方で
  答えを取り出せるようにしておきます。

---

## 2. 全体の流れ

```
お客様の生データ（CSV / JSONL ＋ 画像ファイル）  ← Volume に置く
        │  手順 B: prep_custom.py で変換（CPU のサーバーレス Jobs・GPU 不要）
        ▼
verl 形式の parquet（train.parquet / test.parquet）
        │  手順 C: grpo_gemma4_custom.yaml（テキスト）/ grpo_gemma4_mm_custom.yaml（画像）
        ▼          ＋ custom_reward.py（採点関数）
GRPO 学習（8×H100）
```
まず試してみたい場合は、手順 A でサンプルの生データを作ってから B・C に進むと、全体の流れを確認できます。

| ファイル | 役割 |
|---|---|
| `prep_custom.py` | 生データ（CSV / JSONL ＋ 画像）を verl 形式の parquet に変換する |
| `custom_prep_job.json` | `prep_custom.py` を CPU のサーバーレス Jobs で動かす定義 |
| `custom_reward.py` | 採点関数のひな形 |
| `grpo_gemma4_custom.yaml` / `grpo_gemma4_mm_custom.yaml` | 独自データ用の学習 YAML（テキスト / 画像） |
| `make_sample_raw.py` / `custom_sample_raw_job.json` | 練習用のサンプル生データを作る（手順 A。お客様のデータがあれば不要） |

---

## 3. データの形式

### 3-1. 用意する生データ（変換前）
1行に1問です。`prep_custom.py` は CSV と JSONL を読めます。

**テキストの場合**（CSV の例）
```csv
question,answer
"A shop sells 3 apples for $2. How much do 12 apples cost?",8
```

**画像の場合**（JSONL ＋ 画像ファイルの例）
```json
{"question": "Find x.", "answer": "3", "image": "images/0000.png"}
```
- `image` には画像ファイルのパスを書きます（JSONL / CSV の場所からの相対パス、または `/Volumes/...` の絶対パス）。
  1問に複数の画像がある場合は `|` で区切ります（例：`images/a.png|images/b.png`）。
- 列名が違う場合は、`--question-col` / `--answer-col` / `--image-col` で指定できます。

### 3-2. 変換後の形式（verl が読む parquet）
`prep_custom.py` が自動で作ります。手で作る場合や、ほかのツールで作る場合は、この形にしてください。

| 列 | 中身 | 例 |
|---|---|---|
| `data_source` | データの名前。**採点関数の選択に使われる**（§1） | `acme/math-qa` |
| `prompt` | モデルへの入力（チャット形式のリスト） | `[{"role": "user", "content": "質問文 + 指示文"}]` |
| `images` | 画像（**画像データのときだけ**） | `[{"bytes": <画像のバイト列>, "path": ""}]` |
| `reward_model` | 採点に使う正解 | `{"style": "rule", "ground_truth": "8"}` |
| `ability` | 任意のラベル | `custom` |
| `extra_info` | 補足情報（任意） | `{"split": "train", "index": 0, "question": "...", "answer": "8"}` |

**画像データの注意（重要）**：`prompt` の文章の中に、**画像を入れる位置として `<image>` を書く**必要があります。
verl は `<image>` が出てきた順に、`images` の画像を当てはめます。**`<image>` を書き忘れると、画像はエラーにならずに
無視され**、文字だけで学習してしまいます。`prep_custom.py` は、画像の枚数分の `<image>` を自動で先頭に付けます。

---

## 4. 手順

前提：setup.md の §1〜§3（CLI・認証・イメージ・`gen/` の生成）が済んでいること。`quickstart.sh` を使った場合、
このページで使うスクリプトのアップロードと `gen/` の生成も済んでいます。以下の `PROF` は自分のプロファイル名に、
`$VOL` は自分の Volume のパス（例：`/Volumes/<catalog>/<schema>/verl_workspace`）に置き換えてください。

### 手順 A（任意）：練習用のサンプル生データを作る
お客様のデータがすでにある場合は飛ばしてください。gsm8k / geo3k から、お客様の生データに見立てた
CSV（テキスト）と JSONL ＋ 画像（画像）を作ります（CPU・約1分）。
```bash
databricks jobs submit --json @gen/custom_sample_raw_job.json -p PROF
```
→ `$VOL/raw/text/questions.csv` と `$VOL/raw/image/data.jsonl` ＋ `$VOL/raw/image/images/*.png` ができます。

### 手順 B：生データを verl 形式に変換する（CPU・GPU 不要）
1. お客様の生データを Volume に置きます（例：`$VOL/raw/text/questions.csv`、`$VOL/raw/image/data.jsonl` と `images/`）。
   ```bash
   databricks fs cp -r ./my_raw_data dbfs:$VOL/raw/image -p PROF
   ```
2. `gen/custom_prep_job.json` の次の値を、自分のデータに合わせて書き換えます（テキストだけ・画像だけの場合は、
   不要なタスクを消してください）。

   | 項目 | 意味 |
   |---|---|
   | `--input` | 生データのファイル（CSV / JSONL） |
   | `--image-dir` | 画像の相対パスの基準になるフォルダ（画像のときだけ） |
   | `--data-source` | **データの名前**（自由に決めてよい。例：`acme/math-qa`） |
   | `--out` | 変換結果の置き場所（学習 YAML の `train_files` / `val_files` と合わせる） |

   必要なら、`--instruction`（質問の後ろに付ける指示文）、`--n-test`（検証用に分ける件数。既定 8）も指定できます。
3. 実行します（約1分）。
   ```bash
   databricks jobs submit --json @gen/custom_prep_job.json -p PROF
   ```
   → `--out` に `train.parquet` / `test.parquet` ができます。

### 手順 C：採点関数を用意する
`custom_reward.py` を開き、必要に応じて直します。ひな形は次の動きをします。
- モデルの答えから、最後の `\boxed{...}` の中身を取り出す（なければ `####` の後ろ、それもなければ最後の数値）
- 空白・カンマ・`$` などをそろえてから正解（`ground_truth`）と比べ、一致すれば 1.0、しなければ 0.0
- 数値なら `3` と `3.0` を同じとみなす

関数の形は次のとおりです（verl がこの名前・引数で呼び出します）。
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    # solution_str: モデルが書いた答え全体 / ground_truth: データの正解
    return 1.0 if 正解なら else 0.0
```
直したら、`gen/custom_reward.py` にも同じ内容を反映します（`quickstart.sh` は `gen/` にコピーします）。

### 手順 D：学習する（8×H100）
```bash
# テキストの場合（単一ノード）
databricks air run --file gen/grpo_gemma4_custom.yaml -p PROF --watch
# 画像の場合（単一ノード）
databricks air run --file gen/grpo_gemma4_mm_custom.yaml -p PROF --watch
```
成功すると、ログに `Training Progress: 100% (3/3)` と `critic/rewards/mean`（採点関数が付けた点数の平均）が出て、
`[run_grpo] verl exit code=0` でジョブが終了します。

---

## 5. YAML のどこを直しているか

`grpo_gemma4_custom.yaml` は `grpo_gemma4.yaml` をもとに、**`# <-- CUSTOM` の印を付けた3か所だけ**を変えたものです
（画像用の `grpo_gemma4_mm_custom.yaml` も同じです）。自分で YAML を作る場合も、この3か所を直します。

```yaml
code_source:
  type: snapshot
  snapshot:
    root_path: .
    include_paths:
      - run_grpo.sh
      - custom_reward.py        # (1) 採点関数のファイルも一緒にアップロードする

parameters:
  train_files: __VOL__/custom/text/train.parquet   # (2) 自分のデータ
  val_files: __VOL__/custom/text/test.parquet

command: |
  cd $CODE_SOURCE_PATH
  bash run_grpo.sh \
    reward.custom_reward_function.path=$CODE_SOURCE_PATH/custom_reward.py \
    reward.custom_reward_function.name=compute_score      # (3) この関数で採点するよう verl に指示する
```
- (3) の `bash run_grpo.sh` の後ろに書いた設定は、そのまま verl に渡されます。学習スクリプトを直す必要はありません。

データに合わせて、次の値も見直してください（`env_variables` に書くと `run_grpo.sh` が読み取ります）。

| 設定 | 既定（テキスト / 画像） | 見直す目安 |
|---|---|---|
| `TRAIN_BATCH_SIZE` | 8 / 4 | **学習データの件数以下**にする。2ノードでは 16 の倍数の制約あり（setup.md §5-3） |
| `MAX_PROMPT_LEN` | 1024 / 768 | 質問文（＋画像のトークン）が収まる長さ。**超えた行は学習から外されます** |
| `MAX_RESPONSE_LEN` | 1024 / 512 | モデルが答えを書くのに必要な長さ |
| `parameters.total_training_steps` | 3 | 動作確認は 3。本番はデータ件数とエポック数に合わせて増やす |
| `MULTIMODAL` / `IMAGE_KEY` | なし / `1`・`images` | 画像データのときは `grpo_gemma4_mm_custom.yaml` を使えば設定済み |

---

## 6. よくあるつまずき

| 症状 | 原因と対処 |
|---|---|
| `NotImplementedError: Reward function is not implemented for data_source=...` | 採点関数が指定されていません。§5 の (1)(3) を確認してください |
| `critic/rewards/mean` が最後までずっと 0 | 採点関数が答えを取り出せていない可能性があります。指示文の書き方（`\boxed{}` など）と `extract_answer` が合っているか確認してください。小さいデータでは最初の数ステップが 0 になることはあります |
| 画像なのに、文字だけで学習しているように見える | `prompt` に `<image>` が入っていません（§3-2） |
| `AssertionError: image_offset ... >= len(images)` | `<image>` の数が画像の枚数より多くなっています |
| 学習データが少なすぎて始まらない | `TRAIN_BATCH_SIZE` を学習データの件数以下にしてください |
| 一部の行が学習に使われない | `MAX_PROMPT_LEN` を超えた行は外されます。長さを見直してください |
