"""詳細ペルソナ＋汎用 blueprint による多話題対話データセット生成。

weekend_v4 の良かった構成を維持し、次だけを汎用化する。
  - B の設定を Markdown spec 由来の詳細ペルソナに変更
  - 冒頭を話題 spec の固定文に変更
  - 話題逸脱を B の乗り気度で制御

Run one shard:
  uv run --no-sync python gen_dataset_general.py --model-dir models/instruct31b \
      --out datasets/general_v1 --nshard 64 --shard 0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys

from general_dialogue_spec import (
    DEFAULT_PERSONA_SPEC,
    DEFAULT_TOPIC_SPEC,
    enumerate_specs,
    load_persona_options,
    load_topic_domains,
)


PROMPT_VERSION = "general-dialogue-blueprint-v1"
GREETING_INTERRUPT_RATE = 0.30
NAME_RE = re.compile(r"[ABＡＢ]\s*さん|[〇○△□×✕●▲■◯]")
LATIN_RE = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ]")
SCAR_RE = re.compile(r"[、。]\s*の方は|^の方は")
CONF_RE = re.compile(r"私のことです|私の方の話|自分から振っちゃ|あ、私のこと")
T09_ALLOWED_LATIN_RE = re.compile(r"AI|VR|AR")
REPEATED_COMMA_RE = re.compile(r"、{2,}")
TERM_PUNCT = ("。", "！", "？", "!", "?")
PUNCT_RE = re.compile(r"[、。．，！？!?・…\s]")

CLOSINGS = [
    ("normal", 0.5,
     "話が一区切りしたら自然に締める。毎回同じ定型句にせず、会話内容に応じて締める。"),
    ("abrupt", 0.2,
     "終盤で、どちらかが用事などを理由にやや急に切り上げる。急でも感じは悪くしない。"),
    ("no_close", 0.3,
     "会話は締めない。別れの挨拶を書かず、次の話が始まりかけたところで出力を終える。"
     "最後の発話は文の途中で切らず、言い切りで終える。"),
]


def to_alternating(turns):
    """同一話者の連続ターンを損失なく統合する（weekend_v4 と同じ）。"""
    out = []
    for speaker, utterance in turns:
        if out and out[-1][0] == speaker:
            sep = "" if out[-1][1][-1:] in "。、！？" else "、"
            out[-1][1] += sep + utterance
        else:
            out.append([speaker, utterance])
    return out


def _tri_count(turns):
    return sum(
        1 for i in range(1, len(turns) - 1)
        if turns[i - 1][0] == turns[i + 1][0] != turns[i][0]
        and not turns[i - 1][1].rstrip().endswith(TERM_PUNCT)
        and len(PUNCT_RE.sub("", turns[i][1])) <= 4
    )


def add_continuers(turns, rng, target_rate, protect_prefix=1):
    """weekend_v4 と同じ、読点分割＋短い聞き手相づちのトップアップ。"""
    turns = [list(turn) for turn in turns]
    added = 0
    for _ in range(30):
        if len(turns) < 4 or _tri_count(turns) / (len(turns) / 2) >= target_rate:
            break
        candidates = []
        for i, (speaker, utterance) in enumerate(turns):
            if (i < protect_prefix or len(utterance) < 16
                    or not utterance.rstrip().endswith(TERM_PUNCT)):
                continue
            if len(PUNCT_RE.sub("", turns[i - 1][1])) <= 4:
                continue
            for match in re.finditer("、", utterance):
                pos = match.end()
                if (6 <= pos <= len(utterance) - 6
                        and utterance[max(0, pos - 3):pos] != "、、、"
                        and utterance[pos:pos + 1] != "、"):
                    candidates.append((i, pos))
                    break
        if not candidates:
            break
        b_to_a = sum(
            1 for j in range(1, len(turns) - 1)
            if turns[j][0] == "B" and turns[j - 1][0] == turns[j + 1][0] == "A"
            and not turns[j - 1][1].rstrip().endswith(TERM_PUNCT)
            and len(PUNCT_RE.sub("", turns[j][1])) <= 4
        )
        candidates_a = [(i, pos) for i, pos in candidates if turns[i][0] == "A"]
        i, pos = rng.choice(candidates_a if b_to_a < 2 and candidates_a else candidates)
        speaker, utterance = turns[i]
        token = rng.choice(["うん", "うん", "はい", "ええ"])
        other = "B" if speaker == "A" else "A"
        turns[i:i + 1] = [[speaker, utterance[:pos]], [other, token], [speaker, utterance[pos:]]]
        added += 1
    return turns, added


def trim_incomplete_tail(turns, min_turns=6):
    while len(turns) > min_turns and not turns[-1][1].rstrip().endswith(TERM_PUNCT):
        turns.pop()
    return turns


def normalize_repeated_commas(turns):
    """連続読点を1個へ統一し、読点と文末記号の不自然な重なりも除く。"""
    normalized = []
    for speaker, utterance in turns:
        utterance = REPEATED_COMMA_RE.sub("、", utterance)
        utterance = re.sub(r"、([。！？!?])", r"\1", utterance)
        normalized.append([speaker, utterance])
    return normalized


def has_forbidden_latin(text: str, topic_id: str) -> bool:
    if topic_id == "T09":
        text = T09_ALLOWED_LATIN_RE.sub("", text)
    return bool(LATIN_RE.search(text))


def is_clean_general(turns, topic_id):
    return not any(
        NAME_RE.search(u) or SCAR_RE.search(u) or CONF_RE.search(u)
        or REPEATED_COMMA_RE.search(u) or has_forbidden_latin(u, topic_id)
        for _, u in turns
    )


OPENER_FILLER_RE = re.compile(r"(?:えーと|えっと|あの|うーん|そうですね|まあ|なんだろう|、、、)")


def normalize_opener(text):
    """許可されたフィラー・言いよどみだけを除いて固定文を比較する。"""
    text = OPENER_FILLER_RE.sub("", text)
    text = re.sub(r"、{2,}", "、", text)
    return text.strip()


def opener_ok(turns, spec, interrupted):
    """固定話題文が、通常形または指定された挨拶割り込み形で始まることを検査。"""
    opener = spec["opener"]
    if not interrupted:
        return bool(
            turns and turns[0][0] == "A"
            and normalize_opener(turns[0][1]) == opener
        )
    prefix = "こんにちは、"
    rest = opener[len(prefix):]
    return (
        len(turns) >= 3
        and turns[0][0] == "A"
        and normalize_opener(turns[0][1]) == prefix
        and turns[1][0] == "B"
        and "こんにちは" in turns[1][1]
        and turns[2][0] == "A"
        and normalize_opener(turns[2][1]) == rest
    )


def dialogue_path(outroot, spec):
    bucket = f"{spec['topic_idx'] // 1000 * 1000:05d}"
    d = os.path.join(outroot, "dialogues", spec["topic_id"], bucket)
    return d, os.path.join(d, f"{spec['id']}.jsonl")


def metadata_path(outroot, spec):
    bucket = f"{spec['topic_idx'] // 1000 * 1000:05d}"
    d = os.path.join(outroot, "metadata", spec["topic_id"], bucket)
    return d, os.path.join(d, f"{spec['id']}.json")


PERSONA_FIELDS = (
    "gender", "age", "residence", "family", "occupation",
    "finances", "values", "concern", "interest",
)


def build_metadata(spec, blueprint, output=None):
    """対話ファイルと1対1に対応する、独立した設定・来歴レコード。"""
    return {
        "id": spec["id"],
        "fixed_opener": spec["opener"],
        "persona": {field: spec[field] for field in PERSONA_FIELDS},
        "blueprint": blueprint,
        "sampling": {
            "global_index": spec["idx"],
            "base_seed": spec["seed"],
        },
        "output": output,
    }


def write_json_atomic(path, value, shard):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{shard}"
    with open(tmp, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def make_event(kind, rng, spec, **kwargs):
    """weekend_v4 の会話イベントを、特定話題に依存しない形へ変換する。"""
    frac = round(rng.uniform(0.35, 0.85), 2)
    pos = "中盤" if frac < 0.6 else "終盤の手前"
    if kind == "E1":
        if spec["interest"] == "普通":
            text = (
                f"会話の{pos}で、B が今の話から自然に連想した関連話題へ少し広げる。"
                "A は無視せず応じ、強引に元の質問へ戻さない。"
            )
            return {"kind": kind, "frac": frac, "shift": "related"}, text
        text = (
            f"会話の{pos}までに、B はこの話題にあまり興味がないことを自然な反応で示し、"
            "自分が話しやすい別の話題へ移る。A は興味を強要せず、その新しい話を自然に聞く。"
            "ただし会話自体を拒否したり、不機嫌な態度にはしない。"
        )
        return {"kind": kind, "frac": frac, "shift": "user_led"}, text
    if kind == "E2":
        ask = rng.choice(["A 自身の意見や好みを聞く", "A の経験を聞く", "A ならどうするか聞く"])
        text = (
            f"会話の{pos}で、B が A に質問する（{ask}）。A は聞かれたことに具体的に答える。"
            "知らないことや不確かなことなら、知ったかぶりせずその旨を伝える。"
        )
        return {"kind": kind, "frac": frac, "ask": ask}, text
    if kind == "E3":
        text = (
            "会話のどこかで一度、B が A の言ったことを聞き返す。"
            "A は言い直すか具体例を挙げて分かりやすく答える。"
        )
        return {"kind": kind, "frac": frac}, text
    if kind == "E4":
        text = (
            "会話のどこかで一度、B が言いさして止まる（「いや、それがですね、」のように）。"
            "A は急かさず、短い確認の一言で優しく拾う。"
        )
        return {"kind": kind, "frac": frac}, text
    if kind == "E5":
        text = (
            f"会話の{pos}で、B が少し前に出た話を蒸し返す。"
            "A は内容を正確に覚えていて自然についていく。"
        )
        return {"kind": kind, "frac": frac}, text
    if kind == "E6":
        text = (
            "終盤で B がやや唐突に会話を切り上げようとする。"
            "A は引き止めず、相手に合わせて感じよく短く締める。"
        )
        return {"kind": kind, "frac": frac}, text
    if kind == "E7":
        b_to_a = kwargs.get("b_can_interrupt") and rng.random() < 0.5
        if b_to_a:
            sub, text = rng.choice([
                ("感情B", "A が自分の話をしている途中（読点の切れ目）で、B が驚きや興味から割り込む。"
                        "A のターンは言いかけで終わり、B は相づちだけでなく内容のある発話をする。"
                        "A はそれに答えてから、自分の話の続きに自然に戻る。"),
                ("確認B", "A の話の途中で分かりにくい言葉が出た瞬間、B が割り込んで聞き返す。"
                        "A のターンは読点で途切れる。A は簡単に説明してから続きを話す。"),
            ])
        else:
            sub, text = rng.choice([
                ("感情A", "B の発話の途中（読点の切れ目）で、A が驚きや興味から割り込む。"
                        "B のターンは言いかけで終わり、A は相づちだけでなく内容のある発話をする。"
                        "B はそれに答えてから、話の続きに自然に戻る。"),
                ("確認A", "B の話の途中で分かりにくい言葉が出た瞬間、A が割り込んで聞き返す。"
                        "B のターンは読点で途切れる。B は簡単に説明してから続きを話す。"),
                ("先取り", "B の言いたいことが見えたところで、A が文の続きを先回りして言う。"
                         "B は「そうそう」のように受けて話を続ける。"),
            ])
        return {"kind": kind, "frac": frac, "sub": sub}, text
    raise ValueError(kind)


def make_blueprint(spec, rng_seed):
    """v4 と同じ構造分布を使い、汎用対話の設計図を決定的に作る。"""
    rng = random.Random(f"general-bp-{rng_seed}-{spec['idx']}")
    bp = {}
    r = rng.random()
    lo, hi = (20, 29) if r < 0.2 else (28, 39) if r < 0.7 else (39, 49)
    bp["n_turns_target"] = rng.randrange(lo, hi)
    bp["closing"] = rng.choices(
        [c[0] for c in CLOSINGS], weights=[c[1] for c in CLOSINGS]
    )[0]
    bp["greeting_interrupt"] = rng.random() < GREETING_INTERRUPT_RATE
    bp["a_shares"] = rng.random() < 0.4
    bp["a_share_mode"] = rng.choice(["小さな経験", "個人的な好み", "率直な感想"])
    bp["callback"] = rng.random() < 0.4
    bp["callback_phrase"] = rng.choice([
        "そういえばさっき〜って言ってましたけど",
        "さっきの〜の話なんですけど",
        "さっき〜って言ってたじゃないですか",
        "〜って言ってたの、あれいいですね",
    ])
    bp["continuer_target"] = round(rng.uniform(0.25, 0.45), 2)
    bp["gen_temperature"] = round(rng.uniform(0.85, 1.0), 2)

    r = rng.random()
    n_ev = 0 if r < 0.65 else 1 if r < 0.95 else 2
    kinds = rng.sample(["E2", "E3", "E4", "E5", "E6"], k=n_ev)
    # 逸脱は乗り気度だけで決める。乗り気には E1 を入れない。
    if spec["interest"] == "普通" and rng.random() < 0.10:
        kinds.append("E1")
    elif spec["interest"] == "あまり興味がない":
        kinds.append("E1")

    bp["events"] = []
    event_texts = []
    for kind in kinds:
        event, text = make_event(kind, rng, spec)
        bp["events"].append(event)
        event_texts.append(text)
        if kind == "E6":
            bp["closing"] = "abrupt_by_B"
    if rng.random() < 0.15:
        event, text = make_event("E7", rng, spec, b_can_interrupt=bp["a_shares"])
        bp["events"].append(event)
        event_texts.append(text)
    return bp, event_texts


A_PERSONA = """話者A（システム側。固定キャラクター）:
・女性。話題を提供し、相手Bの反応を受けて自然な雑談をする聞き手役。
・質問係に偏らず、感想・共感・適度な自己開示もする。
・相談には乗るが専門家のふりはしない。分からないことや不確かなことは推測で断定せず、分からない、知らないという意思を自然に伝える。
・明るく自然なカジュアル敬語。フレンドリーだが軽すぎない。
・Bの個人情報は、Bが会話の中で明かすまで知らない。"""


INTEREST_INSTRUCTIONS = {
    "乗り気": "B は提示された話題に乗り気で、比較的詳しく答え、自分からも関連する話をする。",
    "普通": "B は提示された話題に普通に応じる。話題を中心にしつつ、自然な関連話題への展開はよい。",
    "あまり興味がない": (
        "B は提示された話題にあまり興味がなく、最初の反応は短め。ただし会話自体は拒否せず、"
        "自分が関心を持てる方向へ自然に話を移す。"
    ),
}


def opening_instruction(spec, bp):
    if bp["greeting_interrupt"]:
        rest = spec["opener"][len("こんにちは、"):]
        return (
            "・冒頭3ターンは次の通りにする。表現を言い換えず、この順序と本文を守る。\n"
            "  A: こんにちは、\n"
            "  B: こんにちは\n"
            f"  A: {rest}\n"
            "  B の挨拶だけは「こんにちは」に自然なフィラーを添えてもよい。"
        )
    return (
        f"・最初の発話は A の「{spec['opener']}」。表現を言い換えず、この一文をそのまま使う。"
        "自然なフィラーだけは加えてよいが、読点を連続させない。フィラーを加えなくてもよい。"
    )


def build_prompt(spec, bp, event_texts):
    closing_text = next(c[2] for c in CLOSINGS if c[0] == bp["closing"]) \
        if bp["closing"] != "abrupt_by_B" else "（締め方はイベント指示に従う）"
    plan = [
        opening_instruction(spec, bp),
        f"・全体で約{bp['n_turns_target']}ターン。大きく超えず、多くても1.2倍まで。",
        f"・締め方: {closing_text}",
    ]
    if bp["a_shares"]:
        plan.append(
            f"・会話の流れに合う場所で一度、A も自分の{bp['a_share_mode']}を短く話す。"
            "B の設定を自分の経験として流用せず、別の自然な内容にする。作った設定は会話中で一貫させる。"
        )
    if bp["callback"]:
        plan.append(
            f"・後半で一度、A が B の前半の発言を思い出して言及する"
            f"（「{bp['callback_phrase']}」のような雰囲気）。実際の B の発言と正確に一致させる。"
        )
    plan.extend(f"・{text}" for text in event_texts)
    latin_rule = (
        "・原則として英単語やアルファベットを使わない。ただしこの T09 の対話では、AI、VR、AR だけは表記してよい。"
        if spec["topic_id"] == "T09" else
        "・英単語やアルファベットを使わない。外来語や略語はカタカナで書く。"
    )
    return (
        "二人（AとB）の自然な日本語の雑談を作ってください。\n\n"
        + A_PERSONA + "\n\n"
        "会話を始める固定話題提起文:\n"
        f"・{spec['opener']}\n\n"
        "相手B（今回の人物）:\n"
        "・A と雑談するつもりで参加している。\n"
        f"・性別: {spec['gender']}\n"
        f"・年齢: {spec['age']}\n"
        f"・居住地: {spec['residence']}\n"
        f"・家族構成: {spec['family']}\n"
        f"・職業: {spec['occupation']}\n"
        f"・経済状況: {spec['finances']}\n"
        f"・価値観: {spec['values']}\n"
        f"・日常的な悩み: {spec['concern']}\n"
        f"・話題への乗り気度: {spec['interest']}。{INTEREST_INSTRUCTIONS[spec['interest']]}\n"
        "・属性は内面設定であり、一覧を自己紹介のように読み上げない。自然な流れで必要になったものだけ本人が明かす。\n"
        "・日常的な悩みは話題と無理に結び付けず、自然な流れがなければ会話に出さなくてよい。\n\n"
        "今回の会話の設計図（この通りに構成する）:\n"
        + "\n".join(plan) + "\n\n"
        "【A の応答は文脈に接地させる】最重要:\n"
        "・A の深掘り質問は、B の直前の発話に出た具体的な言葉を拾って聞く。\n"
        "・B がすでに答えたことを聞き直さない。決まり文句の質問を連発しない。\n"
        "・A の感想や共感も B の具体的な話に触れる。B が明かしていない属性を先回りして言わない。\n\n"
        "【相づちは2種類を使い分ける】最重要:\n"
        "・続けての合図: 話し手が少し長い内容を言うときは、文の途中の読点でターンを分け、"
        "聞き手が「うん」「はい」「ええ」だけの1ターンを挟み、話し手は次のターンで同じ文の続きを言う。\n"
        "  例: B「実は最近、」→ A「うん」→ B「新しいことを始めたんですよ」\n"
        "・この形を会話全体で頻繁に使う。内容のある発話のおよそ2回に1回は途中に相づちを挟む。\n"
        "・文末の受け: 新情報には「へえ」、理解には「なるほど」、同意には「そうですよね」などを内容に合わせて返す。\n"
        "・単語だけを繰り返すオウム返しを多用しない。\n\n"
        "【自然な短い発話】:\n"
        "・一度に3文以上続けない。相づちは1〜5字、内容のある発話も20字程度まで。\n"
        "・片方だけの独白や、A が質問だけを続ける面接調にしない。\n"
        "・質問されて考えるときは、答えの冒頭に、えーと、うーん、そうですね、などのフィラーを自然に置く。"
        "むやみに散らさず、考える必然のある所だけに置く。\n\n"
        "【表記と一貫性】:\n"
        "・読点「、」を連続させない。言いよどみはフィラーの言葉で表し、「…」「、、、」を使わない。"
        "笑いは「アハハ」など音で書き、「笑」や顔文字や伏せ字を使わない。\n"
        "・相手を Aさん、Bさん、〇〇さんなどと呼ばない。「、の方は」のような不自然な言い方もしない。\n"
        f"{latin_rule}\n"
        "・ペルソナ、会話内で語った経験、人物関係を途中で変えない。\n\n"
        "出力は各行 `A: 発話` または `B: 発話` の形式だけにする。見出し、説明、ナレーションは書かない。"
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/instruct31b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count-per-topic", type=int, default=15000)
    ap.add_argument("--topics", default="", help="例: T01,T09。空なら T01〜T15 全部")
    ap.add_argument("--persona-spec", default=str(DEFAULT_PERSONA_SPEC))
    ap.add_argument("--topic-spec", default=str(DEFAULT_TOPIC_SPEC))
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=4000)
    ap.add_argument("--bp-seed", type=int, default=44)
    ap.add_argument("--max-new-tokens", type=int, default=1792)
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="記録用の基準値。実生成は v4 同様 blueprint の0.85〜1.0ジッタを使う")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device-map", default="auto")
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, set_seed as set_model_seed
    from gen_scenario import generate, parse_turns

    if not 0 <= args.shard < args.nshard:
        raise ValueError("--shard は 0 以上 --nshard 未満にする")
    persona_options = load_persona_options(args.persona_spec)
    domains = load_topic_domains(args.topic_spec)
    specs_all = enumerate_specs(args.count_per_topic, args.base_seed, persona_options, domains)
    if args.topics:
        selected = {value.strip().upper() for value in args.topics.split(",") if value.strip()}
        known = {d.topic_id for d in domains}
        if not selected <= known:
            raise ValueError(f"未知の topic: {sorted(selected - known)}")
        specs_all = [s for s in specs_all if s["topic_id"] in selected]
    mine = [s for s in specs_all if s["idx"] % args.nshard == args.shard]
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} / {len(specs_all)}", file=sys.stderr)

    man_dir = os.path.join(args.out, "manifest")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, f"shard_{args.shard:04d}_of_{args.nshard:04d}.jsonl")
    gen_params = {
        "temperature": args.temperature, "temperature_jitter": [0.85, 1.0],
        "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens, "max_retries": args.max_retries,
    }
    cfg_dir = os.path.join(args.out, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    example = specs_all[0]
    bp0, events0 = make_blueprint(example, args.bp_seed)
    with open(os.path.join(cfg_dir, f"run_shard_{args.shard:04d}.json"), "w") as f:
        json.dump({
            "prompt_version": PROMPT_VERSION, "model_dir": args.model_dir,
            "count_per_topic": args.count_per_topic,
            "selected_topics": sorted({s["topic_id"] for s in specs_all}),
            "nshard": args.nshard, "shard": args.shard,
            "base_seed": args.base_seed, "bp_seed": args.bp_seed,
            "persona_spec": os.path.abspath(args.persona_spec),
            "topic_spec": os.path.abspath(args.topic_spec),
            "persona_options_snapshot": persona_options,
            "topic_domains_snapshot": [
                {"topic_id": d.topic_id, "name": d.name, "openers": list(d.openers)}
                for d in domains
            ],
            "gen_params": gen_params,
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_template_example": build_prompt(example, bp0, events0),
        }, f, ensure_ascii=False, indent=2)

    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"[load] {args.model_dir}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)

    done = skipped = 0
    with open(man_path, "a") as manifest:
        for spec in mine:
            ddir, dpath = dialogue_path(args.out, spec)
            _, mpath = metadata_path(args.out, spec)
            if os.path.exists(dpath) and not args.overwrite:
                if not os.path.exists(mpath):
                    bp, _ = make_blueprint(spec, args.bp_seed)
                    write_json_atomic(
                        mpath,
                        build_metadata(spec, bp, output={
                            "dialogue_file": os.path.relpath(dpath, args.out),
                            "status": "existing_dialogue_not_rechecked",
                        }),
                        args.shard,
                    )
                skipped += 1
                continue
            bp, event_texts = make_blueprint(spec, args.bp_seed)
            prompt = build_prompt(spec, bp, event_texts)
            turns, clean, seed_used, raw = [], False, spec["seed"], ""
            n_cont_added = 0
            for attempt in range(args.max_retries + 1):
                seed_used = spec["seed"] + attempt * 100000
                set_model_seed(seed_used)
                raw = generate(
                    model, processor, prompt, args.max_new_tokens,
                    bp["gen_temperature"], args.top_p, args.repetition_penalty,
                )
                turns = trim_incomplete_tail(to_alternating(parse_turns(raw)))
                turns, n_cont_added = add_continuers(
                    turns, random.Random(f"cont-{seed_used}"), bp["continuer_target"],
                    protect_prefix=3 if bp["greeting_interrupt"] else 1,
                )
                turns = normalize_repeated_commas(turns)
                if is_clean_general(turns, spec["topic_id"]) and opener_ok(
                    turns, spec, bp["greeting_interrupt"]
                ):
                    clean = True
                    break

            os.makedirs(ddir, exist_ok=True)
            tmp = dpath + f".tmp{args.shard}"
            with open(tmp, "w") as f:
                for speaker, utterance in turns:
                    f.write(json.dumps([speaker, utterance], ensure_ascii=False) + "\n")
            os.replace(tmp, dpath)
            metadata_output = {
                "dialogue_file": os.path.relpath(dpath, args.out),
                "raw_file": os.path.join("raw", spec["topic_id"], f"{spec['id']}.txt"),
                "n_turns": len(turns),
                "n_chars": sum(len(u) for _, u in turns),
                "clean": clean,
                "seed_used": seed_used,
                "n_attempts": attempt + 1,
                "continuer_added": n_cont_added,
                "model_dir": args.model_dir,
                "prompt_version": PROMPT_VERSION,
                "gen_params": gen_params,
            }
            topic_raw_dir = os.path.join(raw_dir, spec["topic_id"])
            os.makedirs(topic_raw_dir, exist_ok=True)
            with open(os.path.join(topic_raw_dir, f"{spec['id']}.txt"), "w") as f:
                f.write(raw)
            write_json_atomic(
                mpath, build_metadata(spec, bp, output=metadata_output), args.shard
            )
            manifest.write(json.dumps({
                **spec, "blueprint": bp,
                "file": os.path.relpath(dpath, args.out),
                "metadata_file": os.path.relpath(mpath, args.out),
                "n_turns": len(turns), "n_chars": sum(len(u) for _, u in turns),
                "clean": clean, "seed_used": seed_used, "n_attempts": attempt + 1,
                "continuer_added": n_cont_added,
                "model_dir": args.model_dir, "prompt_version": PROMPT_VERSION,
                "gen_params": gen_params,
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            done += 1
            if done % 25 == 0:
                print(f"  [shard {args.shard}] {done} done, {skipped} skipped", file=sys.stderr)
    print(f"[shard {args.shard}] DONE: {done} generated, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
