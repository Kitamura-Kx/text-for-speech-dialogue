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

## 出所

0378 dialogue_text_gen リポジトリのコミット 1026c7f9 からのコード抽出 (2026-07-06)。
研究データ・実験記録は含まない。内部利用限定。
