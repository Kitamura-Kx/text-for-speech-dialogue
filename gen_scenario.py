"""Two-stage dialogue generation for a NARROW topic with a FIXED speaker-A persona.

Stage 1 (scenario):  given a narrow topic + fixed A persona, ask the model for
                     N mutually-distinct scenarios (B persona / situation /
                     trigger / obstacle / direction). This is what gives variety
                     without churning out the same conversation.
Stage 2 (dialogue):  for each scenario, generate the A/B dialogue grounded in
                     that scenario. A keeps the fixed persona; because every
                     turn is anchored to the scenario, empathy ("わかる") no
                     longer contradicts the surrounding context.

Model is loaded once and reused for both stages.

  results_scenario/<subdir>/scenarios.json     parsed scenarios
  results_scenario/<subdir>/stage1_raw.txt     raw stage-1 decode
  results_scenario/<subdir>/scen<NN>.jsonl     dialogue per scenario (mstts-ready)
  results_scenario/<subdir>/scen<NN>_raw.txt   raw dialogue decode

Usage:
  uv run --no-sync python gen_scenario.py --model-dir models/instruct31b \
      --topic 運動不足 --n 8 --subdir undou31b --seed-rng 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

# ------------------------------------------------------------ fixed A persona

A_PERSONA = """話者A（固定キャラクター。最後までブレさせない）:
・立場: 運動・健康指導の専門家（パーソナルトレーナー兼 保健師のような人）。運動不足の相談に乗るのが仕事。
・口調: フレンドリーで親しみやすい。堅すぎないカジュアルな敬語に時々タメ口が混じる感じ（「〜だよね」「〜しましょっか」「いいですね」）。説教くさく・上から目線にはしない。
・スタンス: まず相手の状況と気持ちに共感する。否定しない。そのうえで、相手に合った“小さく続けられる一歩”を一緒に考えて提案する。専門用語は噛み砕く。"""

# ------------------------------------------------------------ generation helper


def generate(model, processor, instruction, max_new_tokens, temperature,
             top_p, repetition_penalty):
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    tok = getattr(processor, "tokenizer", processor)
    return tok.decode(out[0][in_len:], skip_special_tokens=True)


# ------------------------------------------------------------ stage 1: scenarios

def build_scenario_prompt(topic, n):
    return (
        f"「{topic}」というテーマで、専門家A に相談する相手B との会話シナリオを {n} 個つくってください。\n\n"
        + A_PERSONA + "\n\n"
        "条件:\n"
        f"・{n}個のシナリオは互いに【できるだけ違う】状況にすること。同じような設定を繰り返さない。\n"
        "・Bの人物像・運動不足になっている事情・相談のきっかけ・つまずく理由・会話の落とし所を、それぞれ変える。\n"
        "・現実的で具体的に。固有名詞や数字も適度に。\n\n"
        "出力は次のJSON配列【のみ】。説明やコードフェンス(```)は書かない:\n"
        '[{"id":1,"b_persona":"Bの年代/性別/性格/生活","situation":"運動不足の具体的な状況",'
        '"trigger":"相談しようと思ったきっかけ","obstacle":"続かない/できない理由",'
        '"direction":"この会話で向かう落とし所"}, ...]'
    )


def parse_scenarios(raw):
    """Extract the first top-level JSON array from the raw decode."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in stage-1 output")
    return json.loads(raw[start:end + 1])


# ------------------------------------------------------------ stage 2: dialogue

def build_dialogue_prompt(topic, sc):
    return (
        "以下の設定に【完全に沿って】、二人(AとB)の自然な日本語の対話を作ってください。\n\n"
        + A_PERSONA + "\n\n"
        "相手B（今回の相談者）:\n"
        f"・人物: {sc.get('b_persona','')}\n"
        f"・状況: {sc.get('situation','')}\n"
        f"・相談のきっかけ: {sc.get('trigger','')}\n"
        f"・うまくいかない理由: {sc.get('obstacle','')}\n"
        f"・この会話の落とし所: {sc.get('direction','')}\n\n"
        "会話の作り方:\n"
        f"・A はこのテーマ（{topic}）の専門家として、B の話をよく聞き、共感し、B に合った小さく続けられる一歩を一緒に見つける。説教・上から目線にしない。\n"
        "・共感の言葉（それわかります／大変でしたね／いいですね）を返す。\n\n"
        "【相槌の入れ方】ここが特に大事。相槌の仕組みを正しく再現すること:\n"
        "・相槌は「あなたの話を聞いているよ、続けていいよ」という合図。だから、片方が話を続けている途中の【区切り（＝読点のところ）】で、もう片方が短い相槌（うん／うんうん／へえ／なるほど／あー／たしかに）だけを入れる。\n"
        "・具体的には、一人の話を複数のターンに区切り、その合間にもう一人の短い相槌ターンを挟む。例:\n"
        "    B: 最近ね、在宅勤務になってから\n"
        "    A: うん\n"
        "    B: ほとんど歩かなくなって\n"
        "    A: あー\n"
        "    B: 気づいたら一日中座りっぱなしで\n"
        "    A: なるほど\n"
        "・つまり毎回きっちり交互に内容を言い合うのではなく、『話し手＋聞き手の相槌』のかたまりを作る。\n"
        "・【重要】相槌を入れても話し手がそれ以上続けない（黙ってしまう）ときは、聞き手側が話を引き取って自分から話す（質問する・感想を言う・話を進める）。相槌だけで会話を止めない。\n\n"
        "【発話は短く】:\n"
        "・1発話は短め。相槌は1〜5字。内容のある発話も20字程度まで。\n"
        "・テンポよく。フィラー（えーと／あの／なんか／まあ）も自然に混ぜる。\n\n"
        "【表記の制約（音声で読み上げるため必ず守る）】:\n"
        "・言い淀み・口ごもりは「…」(三点リーダ)を使わず、「、、、」で表す（例:「うーん、、、そうだなあ」）。\n"
        "・音声合成しにくい記号は使わない。特に「笑」「(笑)」などの表記は禁止。笑い声は「アハハ」「ふふ」のように音で書く。\n"
        "・使ってよいのは ふつうのかな・漢字・カタカナ と 、。？！ と 「、、、」 くらい。顔文字・記号・絵文字は使わない。\n\n"
        "【一貫性が最重要】:\n"
        "・上の設定（状況・理由・落とし所）から外れない。前の発言を受けて矛盾なくつなぐ。Bの設定を途中で勝手に変えない。\n"
        "・具体を出す（『○○』のような伏せ字は禁止）。\n\n"
        "出力は各行 `A: 発話` または `B: 発話` の形式のみ。"
        "相槌が多い分ターン数は増えてよい。80〜120ターンで、落とし所まで話して自然に締める。見出し・説明・ナレーションは書かない。"
    )


TURN_RE = re.compile(r"^\s*(?:final|analysis|commentary|model|thought)?\s*([AB])\s*[:：]\s*(.+?)\s*$")


def sanitize(u):
    """Remove TTS-unfriendly notation as a safety net (prompt also forbids it)."""
    for e in ("……", "…", "‥"):
        u = u.replace(e, "、、、")
    u = u.replace("（笑）", "").replace("(笑)", "")  # leave 笑顔/笑う intact
    return u.strip()


def parse_turns(text):
    turns = []
    for line in text.splitlines():
        m = TURN_RE.match(line)
        if not m:
            continue
        utt = sanitize(m.group(2).strip().strip("「」\"' 　"))
        if utt:
            turns.append([m.group(1), utt])
    return turns


# ------------------------------------------------------------ main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/instruct31b")
    p.add_argument("--topic", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--subdir", required=True, help="output dir under results_scenario/")
    p.add_argument("--outroot", default="results_scenario")
    p.add_argument("--scenario-max-new-tokens", type=int, default=2048)
    p.add_argument("--dialogue-max-new-tokens", type=int, default=2560)
    p.add_argument("--scenario-temperature", type=float, default=1.0)
    p.add_argument("--dialogue-temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--seed-rng", type=int, default=1)
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = os.path.join(args.outroot, args.subdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"[load] {args.model_dir}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map=args.device_map)
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # ---- stage 1
    print(f"[stage1] generating {args.n} scenarios for 「{args.topic}」", file=sys.stderr)
    set_seed(args.seed_rng)
    s1_raw = generate(model, processor, build_scenario_prompt(args.topic, args.n),
                      args.scenario_max_new_tokens, args.scenario_temperature,
                      args.top_p, args.repetition_penalty)
    with open(os.path.join(outdir, "stage1_raw.txt"), "w") as f:
        f.write(s1_raw)
    scenarios = parse_scenarios(s1_raw)
    with open(os.path.join(outdir, "scenarios.json"), "w") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)
    print(f"[stage1] parsed {len(scenarios)} scenarios", file=sys.stderr)

    # ---- stage 2
    for i, sc in enumerate(scenarios, 1):
        set_seed(args.seed_rng + i)  # vary per dialogue
        tag = f"scen{i:02d}"
        d_raw = generate(model, processor, build_dialogue_prompt(args.topic, sc),
                         args.dialogue_max_new_tokens, args.dialogue_temperature,
                         args.top_p, args.repetition_penalty)
        with open(os.path.join(outdir, f"{tag}_raw.txt"), "w") as f:
            f.write(d_raw)
        turns = parse_turns(d_raw)
        with open(os.path.join(outdir, f"{tag}.jsonl"), "w") as f:
            for spk, utt in turns:
                f.write(json.dumps([spk, utt], ensure_ascii=False) + "\n")
        print(f"  [{tag}] {len(turns)} turns "
              f"(B={str(sc.get('situation',''))[:24]}…)", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
