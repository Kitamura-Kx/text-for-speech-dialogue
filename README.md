# weekend-dialgen — blueprint 2層方式の対話台本テキスト生成

週末の過ごし方を話題にした A/B 二者の日本語雑談テキストを LLM (gemma-4-31B-it) で
大量生成するコード。mstts (multistream TTS) の入力を想定した設計。

## 仕組み (2層)

- **Layer 1: blueprint sampler (非LLM)** — 対話ごとに構造を決定的に採番:
  ターン数分布 / 締め方4種 / 出だし5例文 / A自身のエピソード / 明示的コールバック /
  イベント E1〜E7 (話題転換・Aへの直接質問・聞き返し・言いさし・蒸し返し・締めの揺らぎ・割り込み5型) /
  continuer 目標率 / 温度ジッタ。ペルソナは格子4軸(均衡巡回)×多様化5軸(idx-seeded乱数抽選)。
- **Layer 2: realizer (LLM)** — 設計図に従って対話本文を生成。
- 後処理: 交互化 / 末尾言いかけトリム / **continuer 機械トップアップ**
  (話者の文を読点で分割し聞き手の「うん」を挟む。テキスト無損失・目標率まで較正) /
  品質ゲート (名前・英字・伏せ字・「、の方は」抜け殻・役割混乱 lint、失敗時はseedを変えてリトライ)。

## 実行

```bash
uv sync   # transformers 5.13.0.dev0 (git build) 等を uv.lock で固定
# モデル: google/gemma-4-31b-it を models/instruct31b に配置 (hf download)
# 1シャード:
uv run python gen_dataset_v4.py --model-dir models/instruct31b \
    --out datasets/out --count 300 --nshard 4 --shard 0
# ABCI: scripts/run_dataset_v4_shard.sh (rt_HG) / run_dataset_v4_hf.sh (rt_HF 8GPU並列)
```

出力: `dialogues/` (各行 [\"A\"|\"B\",発話] の jsonl・厳密交互) / `raw/` / `manifest/`
(spec+blueprint+生成条件の全記録) / `config/`。シャード安全・冪等 (既存skip)。

## 汎用15話題版

週末版を残したまま、同じ blueprint・短発話・相づち補充・品質ゲート・リトライ方式を
15話題へ汎用化した `gen_dataset_general.py` を用意している。ペルソナ候補と固定話題文は
コードに重複定義せず、次の Markdown spec から直接読み込む。

- `docs/general_dialogue_persona_spec.md`
- `docs/general_dialogue_topic_spec.md`

標準設定では各話題15,000件、合計225,000件を列挙する。話題内の固定文は順番に均等使用し、
乗り気度は話題ごとに 6,818件 : 6,818件 : 1,364件とする。

```bash
# 全15話題、1シャード分
uv run python gen_dataset_general.py --model-dir models/instruct31b \
    --out datasets/general_v1 --nshard 64 --shard 0

# 小規模確認（24は全話題の固定文数4/6/8で割り切れる）
uv run python gen_dataset_general.py --model-dir models/instruct31b \
    --out datasets/general_pilot --count-per-topic 24 --topics T01,T09
```

固定文は記載順に巡回する。`--count-per-topic` が固定文数で割り切れる場合は完全均等になり、
割り切れない小規模パイロットでは先頭の固定文から1件ずつ多く割り当てる。

閲覧・研究発表用に、発話内容を変更せずMarkdownへ変換できる。

```bash
# 1対話だけ変換
python3 scripts/dialogue_jsonl_to_markdown.py path/to/dialogue.jsonl

# dialogues以下をディレクトリ構造ごと一括変換
python3 scripts/dialogue_jsonl_to_markdown.py datasets/general_pilot_2/dialogues \
    --out datasets/general_pilot_2/markdown
```

汎用版では各 `dialogues/Txx/<bucket>/<id>.jsonl` に対して、
`metadata/Txx/<bucket>/<id>.json` を1ファイルずつ保存する。対話JSONLには発話だけを入れ、
メタデータJSONにはペルソナ、`fixed_opener`、blueprint、seed、品質結果、生成条件を入れる。
話題ドメイン名、Topic ID、固定文の通し番号は入れない。

## 出所

0378 dialogue_text_gen リポジトリのコミット 1026c7f9 からのコード抽出 (2026-07-06)。
研究データ・実験記録は含まない。内部利用限定。
