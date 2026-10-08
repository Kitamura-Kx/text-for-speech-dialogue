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
uv sync --frozen   # Python 3.12 (.python-version)、transformers 5.13.0.dev0 等を固定
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

全体データセットの進行状況、確定設定、品質上の判断、セッション間の引き継ぎ事項は
`docs/dataset_production_status.md` を正本として逐次更新する。作業開始前に必ず確認する。

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

ABCI のGPUノードでは `scripts/run_dataset_general_shard.sh` を使える。

```bash
# 1件だけの動作確認
qsub -v OUT=datasets/setup_smoke,COUNT_PER_TOPIC=1,TOPICS=T01,MAX_NEW_TOKENS=512,MAX_RETRIES=0 \
    scripts/run_dataset_general_shard.sh

# 本生成の1シャード
qsub -v OUT=datasets/general_v1,COUNT_PER_TOPIC=15000,NSHARD=64,SHARD=0 \
    scripts/run_dataset_general_shard.sh
```

`rt_HF` 1ノードの8 GPUをすべて使う場合は、8シャードを並列起動する。

```bash
# 標準設定: T01を6500件、T01〜T15全体ルートへ生成
qsub scripts/run_dataset_general_hf.sh
```

T02〜T15を並行して各6500件生成する場合は、Topic別manifest・config・ログを使う。

```bash
# 投入内容だけ表示
scripts/submit_dataset_general_t02_t15.sh

# 14本を投入（各ジョブrt_HF 8 GPU、walltime 12時間）
scripts/submit_dataset_general_t02_t15.sh --submit
```

manifestは`t02_shard_0000_of_0008.jsonl`、configは`run_t02_shard_0000.json`のように保存し、
ログは`logs/generation/T02/<PBS_JOBID>/`以下へ保存する。14ジョブの並行実行でもファイル名は衝突しない。

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

## データ作成後の検査・修復フロー

汎用版の成果物は、対話本文だけでなく1対1のmetadataを生成条件の正本として扱い、次の順で検査する。

1. metadataから固定開始文、Bのペルソナ、blueprint、seed、生成パラメータを与えて対話を生成する。
2. 名前・禁止英字・伏せ字・役割混乱・固定開始文などの通常品質ゲートを実行する。
   英字は原則禁止だが、全Topicの`SNS`、`YouTube`、`URL`、`ICT`、`DIY`、`Tシャツ`、`AI`、`IT`、
   `BGM`、`ビタミンC`、`DM`、`CM`、`Amazon`、`Web`、`OS`、`Google`、`Switch`、`RPG`と、
   T09の`VR`、`AR`は許可する。
   許可語の全角・大小文字の揺れは、保存前に仕様記載の半角表記へ統一する。
3. リトライ後も`metadata.output.clean`が`false`の対話は、dialogue、raw、metadata、manifest行を削除する。
   発生数が少なくmetadata自体にも問題があり得るため、現在は件数補充を目的とした再生成をしない。
4. `clean: true`の対話にも、Bがまだ話していないペルソナをAが先取りする問題があり得るため、
   属性語・言い換え規則で候補を高再現率に抽出する。
5. 候補Aターンまでの会話prefixだけをLLMへ渡し、`leak`、`grounded`、`uncertain`で判定する。
   未来のB発話は判定入力へ入れない。`uncertain`は初回結果を見せず独立プロンプトで一度だけ再判定する。
6. `leak`は再生成対象、`grounded`は維持、二度とも`uncertain`なら自動変更せず要確認とする。
   最終確認ではmetadataとの一致自体を問題にせず、候補A発話がそれ以前のB発話から自然に推測・一般化・
   言い換え・確認質問として発言可能かを判定する。文脈から発言可能なら維持し、それでも不許容なら再生成する。
7. 再生成は局所的な文面修正ではなく、保存済みmetadataの条件を維持した全対話生成とする。
   seedを変更し、Bの全ペルソナはAの事前知識ではないという共通規則と、検出属性ごとの注意を追加する。
8. 再生成物へ通常品質ゲートと先取り監査を再実行し、合格したものだけ採用する。元ファイル、判定、
   旧新seed、追加プロンプト、試行履歴は`provenance/`へ保存し、metadataとmanifestを同期する。
   所定の追加seedを試しても合格しない少数例は、dialogue、raw、metadata、manifest行を同期して削除する。

T01の候補抽出と8 GPU判定の例:

```bash
python3 scripts/detect_persona_leak_candidates.py datasets/general_t01_t15 \
    --topic T01 \
    --output datasets/general_t01_t15/provenance/persona_leak_audit_20260802/candidates.jsonl
qsub scripts/run_persona_leak_judge_hf.sh
```

T02〜T15をTopic別の14本・各8 GPUで並行判定する場合は、各Topicの`candidates.jsonl`を作成した後、
次の投入スクリプトを使う。既定はdry-runで、既存判定JSONLがあるTopicには上書き投入しない。

```bash
scripts/submit_persona_leak_judgement_t02_t15.sh
scripts/submit_persona_leak_judgement_t02_t15.sh --submit
```

判定完了後は結果を集約し、`leak`だけをstagingへ再生成する。全対象が合格したことを確認してから
元データをprovenanceへ退避し、一括反映する。

```bash
python3 scripts/summarize_persona_leak_judgements.py \
    datasets/general_t01_t15/provenance/persona_leak_audit_20260802/judgements \
    --expected-candidates 4020 \
    --output-dir datasets/general_t01_t15/provenance/persona_leak_audit_20260802
qsub scripts/run_persona_leak_regeneration_hf.sh

python3 scripts/apply_persona_leak_regenerations.py \
    --dataset-root datasets/general_t01_t15 \
    --targets datasets/general_t01_t15/provenance/persona_leak_audit_20260802/regeneration_targets.jsonl \
    --stage-dir datasets/general_t01_t15/provenance/persona_leak_audit_20260802/regeneration_staging \
    --archive-dir datasets/general_t01_t15/provenance/persona_leak_audit_20260802/replaced_originals_YYYYMMDD
```

`apply_persona_leak_regenerations.py`は、対象が一つでも未処理・不合格なら本体を変更せず停止する。
同名のarchive directoryが既にある場合も停止するため、上書きせず実行日などを付けた新しい保存先を指定する。

候補件数は漏洩件数ではない。地域名が食品や旅行先として出ただけの例や、Bが先に明示した事実への
自然な言い換えも候補に含め、LLMのprefix判定で区別する。確定設定と実行結果は
`docs/dataset_production_status.md`を参照する。

## 出所

0378 dialogue_text_gen リポジトリのコミット 1026c7f9 からのコード抽出 (2026-07-06)。
研究データ・実験記録は含まない。内部利用限定。
