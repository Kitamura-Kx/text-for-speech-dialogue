"""weekend_v4: blueprint 2層方式の対話データセット生成。

設計文書: docs/weekend_v4_design.md (コミット 14a28b4 で合意)

v3 (gen_dataset.py) からの変更点:
  - Layer1 = blueprint sampler (非LLM): 対話ごとに構造 (ターン数/締め方/出だし) と
    イベント (E1話題転換〜E6締めの揺らぎ) と A 自身のエピソードを乱数で決定し、
    Layer2 = LLM はその設計図の肉付けだけを行う。固定Aフロー単一枠組みをやめる。
  - 内容条件つき質問 (Aの深掘りは B の直前発話の具体語を含む・再質問禁止) を常時ルール化。
  - 明示的コールバック (「さっき〜って言ってた」) を確率的に配置。
    注: zoom1 の分あたり実測 (~0.05回/分) より約5倍の意図的過剰サンプル。
    現実分布の模倣でなく「文脈を読む勾配」を立てる教師信号として。manifest に記録され
    後から ablation 可能。
  - on-topic 規則の緩和: A は自分からは逸れないが、B が別の話を始めたら無視せず応じる
    (user-led)。E1 の飛び先は zoom1 T02-T15 からサンプル。
  - A の自己開示エピソードをタクソノミからサンプル (「カフェで読書」固定の過剰記憶を断つ)。
  - 品質ゲートに抜け殻検出 SCAR_RE (「、の方は」) を追加。名前禁止の指示にも
    「の方は」と言わず「そちらは？」と言う旨を明記 (発生源でも対処)。

タクソノミ・多様化軸・シャード安全設計・冪等 resume は v3 をそのまま流用 (import)。

Run one shard:
  uv run --no-sync python gen_dataset_v4.py --model-dir models/instruct31b \
      --out datasets/weekend_v4_pilot1 --count 300 --nshard 4 --shard 0
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from gen_scenario import generate, parse_turns
from gen_dataset import (TAXONOMY, DIVERSITY, _MULT, NAME_RE, LATIN_RE,
                         to_alternating, enumerate_specs, dialogue_path)


def enumerate_specs_v9(count, base_seed):
    """sampler v2: グリッド4軸は均衡巡回のまま、overlay を idx-seeded RNG の独立抽選に変更。
    v3/v8 の決定的巡回は joint 周期が lcm=40 で「各セル= overlay 固定ペア」になり、
    実現ペルソナが 1,240 通りに縮退していた (100k 計画時に発覚)。抽選化で ~10万通りへ。"""
    import random as _rnd
    types = list(TAXONOMY["type"].keys())
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
            act = acts[(idx * 7 + rep) % len(acts)]
            r = _rnd.Random(f"ovl-{base_seed}-{idx}")
            div = {ax: r.choice(DIVERSITY[ax]) for ax in DIVERSITY}
            specs.append({
                "id": f"wk_{idx:06d}", "idx": idx, "topic": TOPIC,
                "type": t, "activity": act, "mood": m, "companion": c,
                "talkativeness": tk, **div, "seed": base_seed + idx,
            })
        rep += 1
    return specs

TOPIC = "週末の過ごし方"
PROMPT_VERSION = "weekend-v4-blueprint-events-9"  # v9: sampler-v2 (overlay抽選化=ペルソナ縮退解消)+温度ジッタ0.85-1.0。v8: 両方向化  # 上げたら必ず追記

# 「、の方は」型の抜け殻 (名前禁止令の scar tissue)。詳細は memory
# synthetic-data-artifact-hunting。名前を落とした骨格文をゲートで弾く。
SCAR_RE = re.compile(r"[、。]\s*の方は|^の方は")
# 役割混乱の自己修復マーカー (「そっちは？」を A が誤発話→取り繕う痕跡)
CONF_RE = re.compile(r"私のことです|私の方の話|自分から振っちゃ|あ、私のこと|私の週末はどうでした")


def is_clean_v4(turns):
    return not any(NAME_RE.search(u) or LATIN_RE.search(u) or SCAR_RE.search(u)
                   or CONF_RE.search(u) for _, u in turns)


TERM_PUNCT = ("。", "！", "？", "!", "?")
_P_ALL = re.compile(r"[、。．，！？!?・…\s]")


def _tri_count(turns):
    return sum(1 for i in range(1, len(turns) - 1)
               if turns[i - 1][0] == turns[i + 1][0] != turns[i][0]
               and not turns[i - 1][1].rstrip().endswith(TERM_PUNCT)
               and len(_P_ALL.sub("", turns[i][1])) <= 4)


def add_continuers(turns, rng, target_rate):
    """continuer トップアップ: 話者テキスト無損失で「読点分割+聞き手の一言」を
    目標率まで機械挿入する (LLM の自然な挿入 ~0.17 を土台に較正)。
    ガード: 文末記号で終わる発話のみ分割 (E4/E7 の言いかけ断片は保護) /
    「、、、」内では割らない / 相づち隣接には挿入しない。"""
    turns = [list(t) for t in turns]
    added = 0
    for _ in range(30):
        if len(turns) < 4 or _tri_count(turns) / (len(turns) / 2) >= target_rate:
            break
        cands = []
        for i, (spk, u) in enumerate(turns):
            if i == 0 or len(u) < 16 or not u.rstrip().endswith(TERM_PUNCT):
                continue
            if len(_P_ALL.sub("", turns[i - 1][1])) <= 4:
                continue
            for m in re.finditer("、", u):
                p = m.end()
                if 6 <= p <= len(u) - 6 and u[max(0, p - 3):p] != "、、、" and u[p:p + 1] != "、":
                    cands.append((i, p))
                    break
        if not cands:
            break
        b2a = sum(1 for j in range(1, len(turns) - 1)
                  if turns[j][0] == "B" and turns[j - 1][0] == turns[j + 1][0] == "A"
                  and not turns[j - 1][1].rstrip().endswith(TERM_PUNCT)
                  and len(_P_ALL.sub("", turns[j][1])) <= 4)
        cands_a = [(i, p) for i, p in cands if turns[i][0] == "A"]
        i, p = rng.choice(cands_a if (b2a < 2 and cands_a) else cands)
        spk, u = turns[i]
        tok = rng.choice(["うん", "うん", "はい", "ええ"])
        other = "B" if spk == "A" else "A"
        turns[i:i + 1] = [[spk, u[:p]], [other, tok], [spk, u[p:]]]
        added += 1
    return turns, added


ASK_RE = re.compile(r"(そちらは|そっちは)")
SELF_RE = re.compile(r"^(私は|私も|私の|僕は)")
ASK_SELF_SAME_RE = re.compile(r"(そちらは|そっちは)[^。！？]*？\s*(私は|私も|僕は)")


def has_self_ask(turns):
    """役割混乱: A が「そちらは？」と聞いて A 自身が答えるパターン (2型)。
    (a) 次ターン型: 聞いた後の A ターンが自分語りで始まる (10k監査 2.4%)
    (b) 同一発話型: 「そっちは何してた？私は、」と1発話内で自問自答 (同 0.6%)"""
    for i, (s, u) in enumerate(turns):
        if s != "A":
            continue
        if ASK_SELF_SAME_RE.search(u):
            return True
        if ASK_RE.search(u) and ("？" in u or "どう" in u):
            if any(t[0] == "A" and SELF_RE.match(t[1]) for t in turns[i + 1:i + 4]):
                return True
    return False


def trim_incomplete_tail(turns, min_turns=6):
    """末尾が言いかけ (文末記号なし) のターンを言い切りまで巻き戻す (max_new_tokens 切れ対策)。"""
    while len(turns) > min_turns and not turns[-1][1].rstrip().endswith(("。", "！", "？", "!", "?")):
        turns.pop()
    return turns


# ------------------------------------------------------------ Layer 1: blueprint

ZOOM1_TOPICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "seeds", "zoom1_topics.json")

# 出だし: ユーザー承認済みの5パターン (2026-07-05 議論)。原則 = 第1発話で話題確定 /
# 外部事実(天気・時間帯・お疲れ様・久しぶり)を仮定しない / 依頼形(教えて・聞かせて)禁止。
# 例文ほぼそのまま + 語尾レベルの揺らぎのみ許可。OPENER_BAN_RE で lint (リトライ条件)。
OPENER_HINTS = [  # (weight, 第1発話の例文)
    (0.35, "こんにちは。今日は週末の過ごし方について話しませんか？"),
    (0.20, "こんにちは。今日は週末の話をしましょう。最近どうでした？"),
    (0.20, "こんにちは。最近の週末って、どんなふうに過ごしてました？"),
    (0.10, "こんにちは。この前の週末、何してました？"),
    (0.15, "こんにちは。週末の過ごし方について話しませんか？いい過ごし方があったら、まねしたいので。"),
]

OPENER_BAN_RE = re.compile(r"教えて|聞かせて|伺|天気|おはよう|こんばんは|お疲れ|久しぶり")


def opener_ok(turns):
    """A の第1発話 lint: 週末を含み話題を確定させる / 禁止語なし。"""
    if not turns or turns[0][0] != "A":
        return False
    first = turns[0][1]
    return ("週末" in first) and not OPENER_BAN_RE.search(first)


CLOSINGS = [  # (kind, weight, 指示文)
    ("normal", 0.5,
     "週末の話が一区切りしたら自然に締める。ただし毎回ありがちな定型の締め方"
     "（「お互い良い週末を」等）にせず、会話の内容に応じた締め方にする。"),
    ("abrupt", 0.2,
     "終盤で、どちらかが用事などを理由にやや急に会話を切り上げる。急でも感じは悪くしない。"),
    ("no_close", 0.3,
     "会話は締めない。別れの挨拶を書かず、次の話が始まりかけたあたりで出力を終える"
     "（実際の雑談の録音を途中で切ったような終わり方）。ただし最後の発話は文の途中で切らず、言い切りで終える。"),
]


def _load_shift_topics():
    with open(ZOOM1_TOPICS_PATH) as f:
        items = json.load(f)
    # T01(趣味・休日の過ごし方) は週末とほぼ同一話題なので E1 の飛び先から除外
    return [it["topic"] for it in items if "T01" not in it["tag"]]


def _pos_label(frac):
    return "中盤" if frac < 0.6 else "終盤の手前"


def make_event(kind, rng, shift_topics, **kwargs):
    """イベント種別 -> (manifest 記録用 dict, プロンプト指示文)"""
    frac = round(rng.uniform(0.35, 0.85), 2)
    pos = _pos_label(frac)
    if kind == "E1":
        dest = rng.choice(shift_topics)
        back = rng.random() < 0.5
        text = (f"会話の{pos}で、B が「そういえば全然関係ないんですけど」のように切り出して"
                f"「{dest}」の話を始める。A はこれを無視したり週末の話に引き戻したりせず、"
                f"いったん相手の新しい話題に自然に応じて数ターンやり取りする。"
                + ("そのあと、どちらかが自然なきっかけで週末の話に戻す。" if back
                   else "そのままその話題で会話が続いてよい（週末の話に戻らなくてよい）。"))
        return {"kind": kind, "frac": frac, "dest_topic": dest, "return": back}, text
    if kind == "E2":
        ask = rng.choice(["おすすめ（場所・物・過ごし方など、流れに合うもの）を聞く",
                          "A 自身の意見や好みを聞く", "A の過去の経験を聞く"])
        text = (f"会話の{pos}で、B が A に質問する（{ask}）。A は自分のありきたりな話で"
                f"ごまかさず、聞かれたことに具体的で嚙み合った答えを返す。")
        return {"kind": kind, "frac": frac, "ask": ask}, text
    if kind == "E3":
        text = ("会話のどこかで一度、B が A の言ったことを聞き返す（「え、どういうことですか？」など）。"
                "A は言い直したり具体例を挙げたりして分かりやすく答える。")
        return {"kind": kind, "frac": frac}, text
    if kind == "E4":
        text = ("会話のどこかで一度、B が言いさして止まる（「いや、それがですね、、、」のように）。"
                "A は急かさず、短い確認の一言で優しく拾う。")
        return {"kind": kind, "frac": frac}, text
    if kind == "E5":
        text = (f"会話の{pos}で、B が少し前に出た話を蒸し返す（「そういえばさっきの〜の話なんですけど」）。"
                f"A はその内容をちゃんと覚えていて、自然についていく。")
        return {"kind": kind, "frac": frac}, text
    if kind == "E6":
        text = ("終盤で B がやや唐突に会話を切り上げようとする。A は引き止めず、"
                "相手に合わせて感じよく短く締める。")
        return {"kind": kind, "frac": frac}, text
    if kind == "E7":
        b_to_a = kwargs.get("b_can_interrupt") and rng.random() < 0.5
        if b_to_a:
            sub, text = rng.choice([
                ("感情B", "A が自分の週末の話をしている途中（読点の切れ目）で、B が驚きや興味を抑えられず"
                        "割り込む（A のターンは言いかけで終わり、B が一言でなく発話する）。"
                        "A はそれに答えてから、自分の話の続きに自然に戻る。"),
                ("確認B", "A の話の途中で分かりにくい言葉が出た瞬間、B が割り込んで聞き返す"
                        "（A のターンは読点で途切れる）。A は簡単に説明してから続きを話す。"),
            ])
            return {"kind": kind, "frac": frac, "sub": sub}, text
        sub, text = rng.choice([
            ("感情", "話の途中で一度、B の発話の途中（読点の切れ目）で A が驚きを抑えられず割り込む"
                    "（B のターンは言いかけで終わり、A が「え、待って、〜なんですか？」のように一言でなく発話する）。"
                    "B はそれに答えてから話の続きに戻る。"),
            ("確認", "B の話の途中で分かりにくい言葉が出た瞬間、A が割り込んで聞き返す"
                    "（B のターンは読点で途切れる）。B は簡単に説明してから続きを話す。"),
            ("先取り", "B の言いたいことが見えたところで、A が先回りして文の続きを言ってしまう。"
                      "B は「そうそう」のように受けて話を続ける。"),
            ("連想", "A が自分の話をしている途中で、B が連想した自分の話を割り込ませる"
                    "（A のターンは読点で途切れる）。少しやり取りした後、どちらかが元の話に戻す。"),
            ("引き取り", "B が言葉に詰まって言いよどんだところで、A がその言葉を引き取って言う。"
                        "B は「そうそれ」と肯定するか、「いや〜です」と訂正して続ける。"),
        ])
        return {"kind": kind, "frac": frac, "sub": sub}, text
    raise ValueError(kind)


def make_blueprint(spec, rng_seed):
    """spec(idx) から決定的に対話の設計図を引く。全フィールド manifest に記録。"""
    rng = random.Random(f"wkv4-{rng_seed}-{spec['idx']}")
    shift_topics = make_blueprint._topics
    bp = {}
    # 構造
    r = rng.random()
    lo, hi = (20, 29) if r < 0.2 else (28, 39) if r < 0.7 else (39, 49)
    bp["n_turns_target"] = rng.randrange(lo, hi)
    ck = rng.choices([c[0] for c in CLOSINGS], weights=[c[1] for c in CLOSINGS])[0]
    bp["closing"] = ck
    bp["opener_hint"] = rng.choices([o[1] for o in OPENER_HINTS],
                                weights=[o[0] for o in OPENER_HINTS])[0]
    # A 自身の週末エピソード (固定「カフェで読書」をやめ、B と同じ活動リストから引く)
    a_type = rng.choice(list(TAXONOMY["type"].keys()))
    bp["a_activity"] = rng.choice(TAXONOMY["type"][a_type])
    bp["a_type"] = a_type
    bp["a_shares"] = (spec["stance"].startswith("A にも質問し返す")
                      or rng.random() < 0.4)
    bp["b_asks_back_phrase"] = rng.choice(
        ["そっちは何してたんですか？", "そちらはどうでした？", "そっちは？"])
    # 明示的コールバック (zoom1 の分あたり実測の ~5倍の意図的過剰サンプル)
    bp["callback"] = rng.random() < 0.4
    bp["continuer_target"] = round(rng.uniform(0.25, 0.45), 2)
    bp["gen_temperature"] = round(rng.uniform(0.85, 1.0), 2)
    bp["callback_phrase"] = rng.choice(
        ["そういえばさっき〜って言ってましたけど", "さっきの〜の話なんですけど",
         "さっき〜って言ってたじゃないですか", "〜って言ってたの、あれいいですね"])
    # イベント (0個65% / 1個30% / 2個5%)
    r = rng.random()
    n_ev = 0 if r < 0.65 else 1 if r < 0.95 else 2
    kinds = rng.sample(["E1", "E2", "E3", "E4", "E5", "E6"], k=n_ev)
    bp["events"] = []
    ev_texts = []
    for k in kinds:
        ev, text = make_event(k, rng, shift_topics)
        bp["events"].append(ev)
        ev_texts.append(text)
        if k == "E6":  # E6 は締め方を上書き (矛盾防止)
            bp["closing"] = "abrupt_by_B"
    # E7 割り込み (E1-E6 とは独立に p=0.15。サブタイプは make_event 内で採番)
    if rng.random() < 0.15:
        ev, text = make_event("E7", rng, shift_topics, b_can_interrupt=bp["a_shares"])
        bp["events"].append(ev)
        ev_texts.append(text)
    return bp, ev_texts


make_blueprint._topics = None  # main() でロード


# ------------------------------------------------------------ Layer 2: prompt

A_PERSONA_V4 = """話者A（対話モデル側。性格は固定）:
・役割: 相手(B)に週末の過ごし方を気持ちよく話してもらう、親しみやすい聞き上手。ただし質問係ではなく、一人の対話相手として自分の話も感想も言う。
・口調: 明るく自然なカジュアル敬語。フレンドリーだが軽すぎない。
・話題との距離感: A は自分からは週末の話の枠を保つ。ただし B が別の話を始めたときは、無視したり強引に週末へ戻したりせず、まず相手の話に応じる。"""


def build_prompt_v4(spec, bp, ev_texts):
    closing_text = next(c[2] for c in CLOSINGS if c[0] == bp["closing"]) \
        if bp["closing"] != "abrupt_by_B" else "（締め方はイベント指示に従う）"
    plan = [
        f"・最初の発話は A の「{bp['opener_hint']}」。ほぼこの通りでよいが、語尾や言い回しは自然に少し変えてよい。"
        "挨拶は「こんにちは」のままにし、天気・時間帯・「お疲れ様」など状況に依存する言葉は加えない。",
        f"・全体で約{bp['n_turns_target']}ターン。これを大きく超えない（多くても1.2倍まで）。",
        f"・締め方: {closing_text}",
    ]
    if bp["a_shares"]:
        plan.append(
            f"・会話の途中で一度だけ、B が A にも週末を尋ね返す（「{bp['b_asks_back_phrase']}」のような言い方。"
            f"尋ね返すのは B。A からは聞き返さない）。A はそこで、自分がこの前の週末に「{bp['a_activity']}」を"
            f"して過ごした話を短くする（毎回同じ定番の話にしない）。"
            f"A が語っている間は、B も文の途中の読点の切れ目で「うん」「はい」を挟む。")
    if bp["callback"]:
        plan.append(
            f"・会話の後半で少なくとも1回、A が B の前半の発言を思い出して言及する"
            f"（「{bp['callback_phrase']}」のような雰囲気で。言い回しはこの通りでなくてよい）。"
            f"言及内容は実際に B が言ったことと正確に一致させる。引用するのは相手の発言だけ"
            f"（B が自分の発言を「言ってましたけど」と引用しない）。")
    for t in ev_texts:
        plan.append(f"・{t}")
    return (
        "二人(AとB)の自然な日本語の雑談を作ってください。\n\n"
        + A_PERSONA_V4 + "\n\n"
        "相手B（今回の人物）:\n"
        f"・人物像: {spec['age']}・{spec['occupation']}、{spec['trait']}な性格。\n"
        f"・この前の週末は「{spec['activity']}」をして過ごした（{spec['type']}系）。\n"
        f"・同伴: {spec['companion']}。\n"
        f"・その週末の気分: {spec['mood']}。\n"
        f"・その週末の小さなエピソード: {spec['episode']}（気分と矛盾しないよう自然に会話へ織り込む）。\n"
        f"・話し方: {spec['talkativeness']}。\n"
        f"・会話のスタンス: {spec['stance']}。このスタンスに沿ってやり取りの型を変える。\n\n"
        "今回の会話の設計図（この通りに構成する）:\n"
        + "\n".join(plan) + "\n\n"
        "【A の応答は文脈に接地させる】最重要:\n"
        "・A の深掘り質問は、B の直前の発話に出てきた具体的な言葉を1つ以上拾って聞く"
        "（B「家庭菜園をしてました」→ A「へえ、何を育ててるんですか？」のように）。\n"
        "・文脈と無関係な決まり文句の質問をしない。B が既に答えたことを聞き直さない。\n"
        "・A の相づち以外の発話（感想・共感）も、B の話の具体的な内容に触れる。\n\n"
        "【相づちは2種類を使い分ける】最重要:\n"
        "・（あ）続けての合図: 話し手が少し長い内容を言うときは、文の途中の読点の切れ目でターンを区切り、"
        "聞き手が「うん」「はい」「ええ」だけの1ターンを挟み、話し手は次のターンで同じ文の続きを言う。\n"
        "　例: B「実は昨日、」→ A「うん」→ B「駅前の新しいお店に行ってきたんですよ」\n"
        "　この形を会話全体で頻繁に使う（内容のある発話のおよそ2回に1回はこの形で途中に挟む）。\n"
        "・（い）文末の受け: 相手の文が終わったあとに独立ターンで返す。新情報には「へえ」「えー」、"
        "説明が腑に落ちたら「なるほど」、意見への同意は「そうですよね」「わかります」と使い分ける。\n"
        "・相手の単語だけを繰り返すオウム返し相づちは多用しない。\n\n"
        "【どちらも長く話さない】:\n"
        "・一度に3文以上続けて話さない。1発話は短く（相づち1〜5字、内容のある発話も20字程度まで）。\n"
        "・片方だけが延々と話す独白にしない。二人で交互にテンポよく。\n\n"
        "【フィラーは“考える場所”に置く】:\n"
        "・質問されて考えるとき、答えの冒頭にフィラー（えーと／うーん／そうですね、、、／なんだろう）を置く。"
        "むやみに散らさず、考える必然のある所に置く。\n\n"
        "【表記】言いよどみは「…」ではなく「、、、」。笑いは「アハハ」と書き、「笑」や顔文字・記号は使わない。\n"
        "【名前を呼ばない】相手を名前で呼ばない。「Aさん」「Bさん」「〇〇さん」は一切使わない。"
        "名前を省いた「、の方は？」のような不自然な言い方もしない。\n"
        "【聞き直し禁止】一度答えてもらったことを聞き直さない。B が自分の週末を語り終えた後に"
        "A が「そっちはどうでした？」のように聞き返すのは誤り（週末を尋ね返してよいのは B→A の方向だけ）。\n"
        "【日本語だけ】英単語やアルファベットを使わない。外来語・略語はカタカナで書く（SF→エスエフ、SNS→エスエヌエス）。\n"
        "【一貫性】B の設定（活動・同伴・気分）から外れない。B の職業・性格などの設定は、"
        "B 本人が会話の中で明かすまで A は知らない前提で話す。設定項目を無理に全部会話へ出さなくてよい。"
        "伏せ字（○○ △△ 等）は絶対に使わない。"
        "具体名が思いつかなければ一般名詞で言う。\n\n"
        "出力は各行 `A: 発話` または `B: 発話` の形式のみ。見出し・説明は書かない。"
    )


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/instruct31b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--base-seed", type=int, default=4000)
    ap.add_argument("--bp-seed", type=int, default=44,
                    help="blueprint 乱数の系列 (base-seed とは独立)")
    ap.add_argument("--max-new-tokens", type=int, default=1792)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    make_blueprint._topics = _load_shift_topics()

    specs_all = enumerate_specs_v9(args.count, args.base_seed)
    mine = [s for s in specs_all if s["idx"] % args.nshard == args.shard]
    print(f"[shard {args.shard}/{args.nshard}] {len(mine)} / {len(specs_all)}", file=sys.stderr)

    man_dir = os.path.join(args.out, "manifest")
    os.makedirs(man_dir, exist_ok=True)
    man_path = os.path.join(man_dir, f"shard_{args.shard:04d}_of_{args.nshard:04d}.jsonl")

    gen_params = {"temperature": args.temperature, "top_p": args.top_p,
                  "repetition_penalty": args.repetition_penalty,
                  "max_new_tokens": args.max_new_tokens, "max_retries": args.max_retries}
    cfg_dir = os.path.join(args.out, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    _bp0, _ev0 = make_blueprint(specs_all[0], args.bp_seed)
    with open(os.path.join(cfg_dir, f"run_shard_{args.shard:04d}.json"), "w") as f:
        json.dump({
            "topic": TOPIC, "prompt_version": PROMPT_VERSION,
            "model_dir": args.model_dir, "count": args.count,
            "nshard": args.nshard, "shard": args.shard,
            "base_seed": args.base_seed, "bp_seed": args.bp_seed,
            "gen_params": gen_params, "taxonomy": TAXONOMY, "diversity": DIVERSITY,
            "closings": CLOSINGS, "opener_hints": OPENER_HINTS,  # (weight, text)
            "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "prompt_template_example": build_prompt_v4(specs_all[0], _bp0, _ev0),
        }, f, ensure_ascii=False, indent=2)

    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    from transformers import set_seed as _set_seed
    print(f"[load] {args.model_dir}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_dir)

    done = skipped = 0
    with open(man_path, "a") as man:
        for spec in mine:
            ddir, dpath = dialogue_path(args.out, spec)
            if os.path.exists(dpath) and not args.overwrite:
                skipped += 1
                continue
            bp, ev_texts = make_blueprint(spec, args.bp_seed)
            prompt = build_prompt_v4(spec, bp, ev_texts)
            turns, clean, seed_used, raw = [], False, spec["seed"], ""
            for attempt in range(args.max_retries + 1):
                seed_used = spec["seed"] + attempt * 100000
                _set_seed(seed_used)
                raw = generate(model, processor, prompt,
                               args.max_new_tokens, bp["gen_temperature"],
                               args.top_p, args.repetition_penalty)
                turns = trim_incomplete_tail(to_alternating(parse_turns(raw)))
                turns, n_cont_added = add_continuers(
                    turns, random.Random(f"cont-{seed_used}"), bp["continuer_target"])
                if is_clean_v4(turns) and opener_ok(turns) and not has_self_ask(turns):
                    clean = True
                    break
            os.makedirs(ddir, exist_ok=True)
            tmp = dpath + f".tmp{args.shard}"
            with open(tmp, "w") as f:
                for spk, utt in turns:
                    f.write(json.dumps([spk, utt], ensure_ascii=False) + "\n")
            os.replace(tmp, dpath)
            with open(os.path.join(raw_dir, f"{spec['id']}.txt"), "w") as f:
                f.write(raw)
            man.write(json.dumps({
                **spec, "blueprint": bp,
                "file": os.path.relpath(dpath, args.out),
                "n_turns": len(turns), "n_chars": sum(len(u) for _, u in turns),
                "clean": clean, "seed_used": seed_used, "n_attempts": attempt + 1,
                "continuer_added": n_cont_added,
                "model_dir": args.model_dir, "prompt_version": PROMPT_VERSION,
                "gen_params": gen_params,
            }, ensure_ascii=False) + "\n")
            man.flush()
            done += 1
            if done % 25 == 0:
                print(f"  [shard {args.shard}] {done} done, {skipped} skipped", file=sys.stderr)

    print(f"[shard {args.shard}] DONE: {done} generated, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
