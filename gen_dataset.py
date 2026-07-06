"""Controlled dialogue DATASET generation for a single narrow topic.

Design goals (see discussion):
  - NARROW topic = 週末の過ごし方 (matches the eval platform topic).
  - COVERAGE by a B-side taxonomy (cartesian product of axes) so the
    conversation space is enumerable and the distribution is fully controlled.
  - A = fixed conversational partner that follows a FLOW and ALWAYS opens with
    a greeting + topic proposal, and never drifts off the topic (this is the
    fix for the "model suddenly goes off-topic" problem).
  - Short dialogues so mstts(abe) synthesis stays in its stable range.
  - SHARD-SAFE parallelism: spec[i] is deterministic from i; each shard only
    touches files for its own indices; existing files are skipped (idempotent /
    resumable); nothing is ever deleted or overwritten. Globally-unique ids.
  - Provenance recorded for every dialogue (topic / cell axes / seed) so model
    behaviour can later be correlated with data composition.

Builds on gen_scenario.py (generate / parse_turns / sanitize).

Run one shard:
  uv run --no-sync python gen_dataset.py --model-dir models/instruct31b \
      --out datasets/weekend --count 60000 --nshard 64 --shard 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

import datetime
import re

from gen_scenario import generate, parse_turns  # reuse helpers

TOPIC = "週末の過ごし方"
PROMPT_VERSION = "weekend-v3-noname-nolatin-filler-stance"  # 上げたら必ず追記

# 名前呼びかけ・伏せ字の検出（mstts が壊れる/モデルが「〇〇さん」と言う のを防ぐ）
# 「Aさん」「Bさん」(全半角) と プレースホルダ記号 を不可とする。
# 「本屋さん」「たくさん」等は さん 単体なので誤検出しない。
NAME_RE = re.compile(r"[ABＡＢ]\s*さん|[〇○△□×✕●▲■◯]")
# 英単語/アルファベット混入も不可（rinna/mstts が壊れる。外来語はカタカナで）
LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")


def is_clean(turns):
    return not any(NAME_RE.search(utt) or LATIN_RE.search(utt) for _, utt in turns)


def to_alternating(turns):
    """Merge consecutive same-speaker turns -> strictly A/B alternating (lossless).
    abe(mstts) ignores labels and assigns speakers by turn parity, so the data
    must alternate. We make the canonical dialogue alternating at save time."""
    out = []
    for spk, utt in turns:
        if out and out[-1][0] == spk:
            sep = "" if out[-1][1][-1:] in "。、！？" else "、"
            out[-1][1] += sep + utt
        else:
            out.append([spk, utt])
    return out

# ------------------------------------------------------------ B-side taxonomy

TAXONOMY = {
    # 過ごし方タイプ -> 具体的な活動(セル内の多様性のため循環使用)
    "type": {
        "アウトドア":   ["キャンプ", "ハイキング", "釣り", "バーベキュー", "サイクリング"],
        "インドア":     ["映画鑑賞", "ゲーム", "読書", "アニメの一気見", "料理"],
        "外出":         ["ショッピング", "カフェ巡り", "日帰り温泉", "美術館", "ドライブ"],
        "休息・家事":   ["一日中寝る", "部屋の片付け", "洗濯と掃除", "だらだら過ごす"],
        "趣味":         ["ジムで筋トレ", "フットサル", "楽器の練習", "写真撮影", "家庭菜園"],
        "人と会う":     ["友人とランチ", "家族と外食", "恋人とデート", "同窓会"],
        "特に何もしない": ["なんとなくスマホ", "特に予定がなかった", "疲れて寝ていた"],
    },
    "mood": ["充実した", "疲れが残った", "退屈だった", "リフレッシュできた", "あまり満足できなかった"],
    "companion": ["一人で", "友人と", "家族と", "恋人と"],
    "talkativeness": ["よく話す方", "口数は少なめ"],
}

# 多様化オーバーレイ（分布制御の格子とは別。within-cell の多様性を出すため、
# idx から互いに素な倍率で decorrelate して決定的に割り当て、provenance に記録する。
# AttrPrompt(Yu+ 2023) / Persona-Hub(Chan+ 2024) の属性・ペルソナ条件付けに相当）
DIVERSITY = {
    "age": ["20代", "30代", "40代", "50代", "60代"],
    "occupation": ["会社員", "大学生", "主婦・主夫", "フリーランス", "自営業",
                   "公務員", "看護師", "エンジニア", "教員", "販売員"],
    "trait": ["おっとり", "せっかち", "のんびり屋", "真面目", "明るい",
              "人見知り", "凝り性", "面倒くさがり"],
    "episode": ["天気に恵まれた", "少し予定が変わった", "初めて挑戦してみた",
                "思ったより人が多かった", "偶然知り合いに会った",
                "ちょっとした失敗があった", "前から楽しみにしていた",
                "急に思い立って出かけた"],
    # 会話のダイナミクス（Aが質問ばかりにならないよう、やり取りの型も振る）
    "stance": [
        "聞かれたことに素直に答える受け身タイプ",
        "A にも質問し返すタイプ（A も自分の週末の話を少しする）",
        "自分から主導でよく話すタイプ",
        "あまり乗り気でなく、A が少しずつ話を引き出すタイプ",
        "話をどんどん広げて展開していくタイプ",
    ],
}
# 互いに素な倍率で各軸を独立に回す
_MULT = {"age": 7, "occupation": 3, "trait": 5, "episode": 11, "stance": 13}


def enumerate_specs(count, base_seed):
    """Deterministic, balanced enumeration of `count` dialogue specs.

    Cells = type × mood × companion × talkativeness (cartesian product).
    Each pass over the cells cycles the concrete activity within each type, so
    repeated coverage of a cell still varies the surface content. spec idx and
    seed are fully determined by position -> reproducible & shard-safe.
    """
    types = list(TAXONOMY["type"].keys())
    # type を最内側に回す → 連続する少数バッチでも全7型を早くカバーできる
    cells = [(t, m, c, tk)
             for m in TAXONOMY["mood"]
             for c in TAXONOMY["companion"]
             for tk in TAXONOMY["talkativeness"]
             for t in types]
    specs = []
    rep = 0
    while len(specs) < count:
        for (t, m, c, tk) in cells:
            if len(specs) >= count:
                break
            idx = len(specs)
            acts = TAXONOMY["type"][t]
            # 活動も idx で decorrelate（rep依存だと小バッチで活動が偏るため）
            act = acts[(idx * 7 + rep) % len(acts)]
            div = {ax: DIVERSITY[ax][(idx * _MULT[ax]) % len(DIVERSITY[ax])]
                   for ax in DIVERSITY}
            specs.append({
                "id": f"wk_{idx:06d}",
                "idx": idx,
                "topic": TOPIC,
                "type": t,
                "activity": act,
                "mood": m,
                "companion": c,
                "talkativeness": tk,
                **div,
                "seed": base_seed + idx,
            })
        rep += 1
    return specs


# ------------------------------------------------------------ A flow + prompt

A_PERSONA = """話者A（対話モデル側。固定。最後までブレさせない）:
・役割: 相手(B)に週末の過ごし方を気持ちよく話してもらう、親しみやすい聞き上手。ただし質問ばかりにはせず、Bに振られたら自分の週末の話も少しする。
・口調: 明るく自然なカジュアル敬語。フレンドリーだが軽すぎない。
・絶対に守る: この会話は最初から最後まで「週末の過ごし方」の話。別の話題に勝手に逸れない。"""


def build_prompt(spec):
    return (
        "二人(AとB)の自然な日本語の雑談を作ってください。\n\n"
        + A_PERSONA + "\n\n"
        "相手B（今回の人物）:\n"
        f"・人物像: {spec['age']}・{spec['occupation']}、{spec['trait']}な性格。\n"
        f"・この前の週末は「{spec['activity']}」をして過ごした（{spec['type']}系）。\n"
        f"・同伴: {spec['companion']}。\n"
        f"・その週末の気分: {spec['mood']}。\n"
        f"・その週末の小さなエピソード: {spec['episode']}（気分と矛盾しないよう自然に会話へ織り込む）。\n"
        f"・話し方: {spec['talkativeness']}。\n"
        f"・会話のスタンス: {spec['stance']}。このスタンスに沿ってやり取りの型を変える。\n\n"
        "会話の流れ（Aはこの流れを守る）:\n"
        "1. 【必ず最初の発話はAの挨拶＋話題提案】例:「こんにちは。今日は週末の過ごし方について話しませんか？」\n"
        "2. A が「最近の週末はどう過ごしました？」のように尋ねる。\n"
        "3. B が上の設定に沿って答え、A が深掘りする（何を/誰と/どこで/きっかけ）。\n"
        "4. A は相づち・共感を返しながら、話を広げる。\n"
        "5. 週末の過ごし方の話として一区切りついたら自然に締める。\n\n"
        "【相づちを多くする】最重要:\n"
        "・相手が話す区切り（読点のところ）で、短い相づち（うん／うんうん／へえ／なるほど／そうなんですね／あー）を"
        "【独立した1ターン】で頻繁に挟む。相づちだけのターンを全体の3〜4割は入れる。\n"
        "・話し手は長い内容を1ターンに詰め込まず、2〜3ターンに区切って言い、その合間に相手の相づちが入る。\n"
        "・相づちを入れても相手が続けないときは、A が短い質問や感想で話を進める。\n\n"
        "【どちらも長く話さない】:\n"
        "・A も B も、一度に3文以上続けて話さない。1発話は短く（相づち1〜5字、内容のある発話も20字程度まで）。\n"
        "・片方だけが延々と話す独白にしない。二人で交互にテンポよく。\n"
        "・A は質問ばかりにしない。B のスタンスに応じて、B が質問し返したら A も自分の週末の話を少しする。\n\n"
        "【フィラーは“考える場所”に置く】:\n"
        "・質問されたり話を振られたりして考えるとき、答えの冒頭にフィラー（えーと／うーん／そうですね、、、／なんだろう）を置く。沈黙＝考えている間を埋めるイメージ。\n"
        "・相づち（うん／へえ）は聞いている合図で役割が違う。フィラーはむやみに散らさず、考える必然のある所に置く。\n\n"
        "【話題は週末の枠内】:\n"
        "・話の流れで別の話に移りたくなっても、週末の過ごし方の枠内（次の週末の予定、別の過ごし方、その活動の周辺）にとどめる。仕事や恋愛そのものなど、全く別の話題には逸れない。\n"
        "【表記】言いよどみは「…」ではなく「、、、」。音声で読み上げるので「笑」などの記号は使わず、笑いは「アハハ」と書く。顔文字・記号は使わない。\n"
        "【名前を呼ばない】二人の会話なので、相手を名前で呼ばない。「Aさん」「Bさん」「〇〇さん」のような呼びかけ・名前は一切使わない。"
        "相手に尋ねるときも名前を付けず「最近の週末はどうでした？」のように言う。\n"
        "【日本語だけ】英単語やアルファベットを使わない。外来語はカタカナで書く（probably→たぶん／Netflix→ネットフリックス）。\n"
        "【一貫性】話題は終始「週末の過ごし方」。B の設定（活動・同伴・気分）から外れない。"
        "伏せ字（○○ 〇〇 △△ ×× 等）は絶対に使わない。具体名が思いつかなければ一般名詞で言う。\n\n"
        "出力は各行 `A: 発話` または `B: 発話` の形式のみ。40〜60ターンで自然に締める。見出し・説明は書かない。"
    )


# ------------------------------------------------------------ io helpers

def dialogue_path(outroot, spec):
    bucket = f"{spec['idx'] // 1000 * 1000:06d}"  # 1000 files per dir
    d = os.path.join(outroot, "dialogues", bucket)
    return d, os.path.join(d, f"{spec['id']}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/instruct31b")
    ap.add_argument("--out", required=True, help="dataset root (never deleted)")
    ap.add_argument("--count", type=int, required=True, help="TOTAL dataset size")
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--max-turns", type=int, default=0,
                    help="0 = keep FULL dialogue (truncation is done downstream "
                         "for abe input, so raw data is never lost)")
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--max-retries", type=int, default=3,
                    help="re-generate (bumped seed) if names/placeholders/latin appear")
    ap.add_argument("--save-raw", action="store_true", default=True,
                    help="also save the raw model decode under raw/ (default on)")
    ap.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    ap.add_argument("--overwrite", action="store_true",
                    help="(default off) regenerate even if the file exists")
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    specs_all = enumerate_specs(args.count, args.base_seed)
    mine = [s for s in specs_all if s["idx"] % args.nshard == args.shard]
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} / {len(specs_all)} dialogues",
          file=sys.stderr)

    # manifest is per-shard -> no cross-process write contention
    man_dir = os.path.join(args.out, "manifest")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, f"shard_{args.shard:04d}_of_{args.nshard:04d}.jsonl")

    # run-level config (taxonomy / params / prompt version): 後からフィルタ・再現に使う。
    gen_params = {"temperature": args.temperature, "top_p": args.top_p,
                  "repetition_penalty": args.repetition_penalty,
                  "max_new_tokens": args.max_new_tokens, "max_turns": args.max_turns,
                  "max_retries": args.max_retries}
    cfg_dir = os.path.join(args.out, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f"run_shard_{args.shard:04d}.json"), "w") as f:
        json.dump({
            "topic": TOPIC, "prompt_version": PROMPT_VERSION,
            "model_dir": args.model_dir, "count": args.count,
            "nshard": args.nshard, "shard": args.shard, "base_seed": args.base_seed,
            "gen_params": gen_params, "taxonomy": TAXONOMY, "diversity": DIVERSITY,
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_template_example": build_prompt(enumerate_specs(1, args.base_seed)[0]),
        }, f, ensure_ascii=False, indent=2)

    raw_dir = os.path.join(args.out, "raw")
    if args.save_raw:
        os.makedirs(raw_dir, exist_ok=True)

    from transformers import set_seed as _set_seed
    print(f"[load] {args.model_dir}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_dir)

    done = skipped = 0
    with open(man_path, "a") as man:  # append = resumable, never truncates
        for spec in mine:
            ddir, dpath = dialogue_path(args.out, spec)
            if os.path.exists(dpath) and not args.overwrite:
                skipped += 1
                continue
            # generate, retrying with a bumped seed if names/placeholders/latin appear
            turns, clean, seed_used, raw = [], False, spec["seed"], ""
            for attempt in range(args.max_retries + 1):
                seed_used = spec["seed"] + attempt * 100000
                _set_seed(seed_used)
                raw = generate(model, processor, build_prompt(spec),
                               args.max_new_tokens, args.temperature,
                               args.top_p, args.repetition_penalty)
                turns = to_alternating(parse_turns(raw))   # canonical: strictly A/B
                if args.max_turns and len(turns) > args.max_turns:
                    turns = turns[:args.max_turns]
                if is_clean(turns):
                    clean = True
                    break
            n_attempts = attempt + 1
            # atomic write: temp file in same dir -> os.replace (no partial files)
            os.makedirs(ddir, exist_ok=True)
            tmp = dpath + f".tmp{args.shard}"
            with open(tmp, "w") as f:
                for spk, utt in turns:
                    f.write(json.dumps([spk, utt], ensure_ascii=False) + "\n")
            os.replace(tmp, dpath)
            if args.save_raw:
                with open(os.path.join(raw_dir, f"{spec['id']}.txt"), "w") as f:
                    f.write(raw)
            man.write(json.dumps({
                **spec, "file": os.path.relpath(dpath, args.out),
                "n_turns": len(turns), "n_chars": sum(len(u) for _, u in turns),
                "clean": clean, "seed_used": seed_used, "n_attempts": n_attempts,
                "model_dir": args.model_dir, "prompt_version": PROMPT_VERSION,
                "gen_params": gen_params,
            }, ensure_ascii=False) + "\n")
            man.flush()
            done += 1
            if done % 50 == 0:
                print(f"  [shard {args.shard}] {done} done, {skipped} skipped",
                      file=sys.stderr)

    print(f"[shard {args.shard}] DONE: {done} generated, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
