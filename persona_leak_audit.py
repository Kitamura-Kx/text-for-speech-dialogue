"""ペルソナ先取り監査の候補抽出・LLM判定用共通処理。"""
from __future__ import annotations

import json
import re
from pathlib import Path


FIELD_LABELS = {
    "gender": "性別", "age": "年齢", "residence": "居住地",
    "family": "家族構成", "occupation": "職業", "finances": "経済状況",
    "values": "価値観", "concern": "日常的な悩み", "interest": "話題への乗り気度",
}

# A発話に現れたときだけ候補にする高再現率の語彙。最終判定は文脈を見るLLMが行う。
VALUE_ALIASES = {
    "gender": {
        "女性": ["女性", "女の人", "女の方"], "男性": ["男性", "男の人", "男の方"],
    },
    "age": {
        "10代後半": ["十代", "高校生くらい", "若いんですね"],
        "20代": ["二十代"], "30代": ["三十代"], "40代": ["四十代"],
        "50代": ["五十代"], "60代以上": ["六十代", "シニア", "高齢者"],
    },
    "residence": {
        "北海道の都市部": ["北海道", "札幌"],
        "東北地方の都市部": ["東北"], "東北地方の郊外": ["東北"],
        "関東地方の都心部": ["関東", "都心", "東京"],
        "関東地方の郊外": ["関東"],
        "中部地方の都市部": ["中部", "東海"], "中部地方の郊外": ["中部", "東海"],
        "関西地方の都市部": ["関西"], "関西地方の郊外": ["関西"],
        "中国・四国地方": ["中国地方", "四国"],
        "九州地方": ["九州"], "沖縄県": ["沖縄"],
    },
    "family": {
        "親と同居": ["親御さんと", "ご両親と", "親と一緒に"],
        "一人暮らし": ["一人暮らし", "お一人で暮ら"],
        "配偶者・パートナーと二人暮らし": ["ご夫婦二人", "パートナーと二人", "奥さんと二人", "旦那さんと二人"],
        "配偶者・パートナーと子ども": ["お子さん", "子どもさん", "ご家族で"],
        "ひとり親として子どもと暮らしている": ["ひとり親", "お一人でお子さん", "お子さん"],
        "三世代で同居": ["三世代", "お孫さんと", "大家族"],
        "兄弟姉妹と同居": ["きょうだいと", "お兄さんと", "お姉さんと", "弟さんと", "妹さんと"],
        "成人した子どもと同居": ["成人したお子さん", "大きなお子さん", "お子さん"],
        "高齢の親を支えながら同居": ["親御さんの介護", "高齢の親御さん", "親御さんを支え"],
    },
    "occupation": {
        "中学生・高校生": ["中学生", "高校生", "学校生活"],
        "大学生・大学院生": ["大学生", "大学院生", "大学の授業"],
        "専門学校生": ["専門学校"], "新卒・若手会社員": ["新卒", "若手社員"],
        "一般会社員": ["会社員", "勤め人"], "管理職": ["管理職", "部下"],
        "公務員": ["公務員"], "教員": ["先生のお仕事", "教員"],
        "医療・介護職": ["医療職", "介護職", "病院勤務"],
        "エンジニア": ["エンジニア", "技術者"],
        "クリエイティブ職": ["クリエイティブ", "制作のお仕事"],
        "営業・販売職": ["営業職", "販売のお仕事"],
        "接客・サービス職": ["接客業", "サービス業"],
        "製造・技術職": ["製造業", "工場勤務", "技術職"],
        "運輸・物流職": ["運送", "物流", "配送のお仕事"],
        "建設・現場職": ["建設業", "現場仕事"], "自営業": ["自営業"],
        "フリーランス": ["フリーランス"], "農林水産業": ["農業", "林業", "漁業"],
        "パート・アルバイト": ["パート", "アルバイト"],
        "家事・育児を主に担っている": ["専業主婦", "専業主夫", "家事や育児が中心"],
        "求職中": ["求職中", "仕事を探して"], "休職中": ["休職中", "お休み中なんですね"],
        "定年退職後・無職": ["定年退職", "退職後", "無職"],
    },
    "finances": {
        "生活費にかなり余裕がない": ["生活が苦しい", "余裕がないんですね", "家計が厳しい"],
        "節約すれば問題なく暮らせる": ["節約すれば大丈夫", "節約して暮ら"],
        "標準的で大きな不安はない": ["お金の心配はない", "家計は安定"],
        "比較的裕福で余裕がある": ["裕福", "余裕があるんですね"],
    },
    "values": {
        "安定した生活を重視する": ["安定を大事", "安定重視"],
        "新しい経験や変化を楽しみたい": ["新しい経験を大事", "変化を楽しみ"],
        "家族や身近な人との時間を大切にする": ["家族との時間を大切", "身近な人との時間を大切"],
        "一人の時間や自分らしさを大切にする": ["一人の時間を大切", "自分らしさを大切"],
        "仕事・学業での成長や達成を重視する": ["成長を重視", "達成感を大切"],
        "趣味や日々の楽しみを大切にする": ["趣味を大切", "日々の楽しみを大切"],
        "健康的で無理のない生活を重視する": ["健康を重視", "無理のない生活を大切"],
        "お金や将来への備えを重視する": ["将来への備えを重視", "貯蓄を重視"],
    },
    "concern": {
        "仕事や学業が忙しい": ["仕事が忙しいんですね", "学業が忙しいんですね"],
        "仕事・学業・進路が自分に合っているかわからない": ["進路に迷って", "仕事が合わない"],
        "職場・学校・友人との人間関係に悩んでいる": ["人間関係で悩んで", "職場の人間関係"],
        "家族やパートナーとの関係に悩んでいる": ["家族関係で悩んで", "パートナーとの関係"],
        "睡眠・運動・食事などの生活習慣が乱れている": ["生活習慣が乱れて", "睡眠不足なんですね"],
        "疲れやストレスがたまっている": ["ストレスがたまって", "お疲れなんですね"],
        "収入・出費・貯金などのお金に不安がある": ["お金に不安", "貯金が心配"],
        "将来の仕事や生活に漠然とした不安がある": ["将来が不安", "先行きが不安"],
        "趣味や自分のための時間を取れない": ["自分の時間が取れない", "趣味の時間がない"],
    },
    "interest": {
        "乗り気": ["かなり乗り気", "興味津々"],
        "普通": [],
        "あまり興味がない": ["興味がなさそう", "あまり興味ないんですね", "乗り気じゃない"],
    },
}


def load_turns(path: Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_candidates(root: Path, topic: str) -> list[dict]:
    candidates = []
    for metadata_path in sorted((root / "metadata" / topic).rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        turns = load_turns(root / metadata["output"]["dialogue_file"])
        for turn_index, (speaker, utterance) in enumerate(turns):
            if speaker != "A":
                continue
            for field, value in metadata["persona"].items():
                aliases = VALUE_ALIASES.get(field, {}).get(value, [])
                matched = sorted({alias for alias in aliases if alias in utterance})
                if matched:
                    candidates.append({
                        "candidate_id": f"{metadata['id']}:{turn_index}:{field}",
                        "dialogue_id": metadata["id"], "field": field,
                        "field_label": FIELD_LABELS[field], "persona_value": value,
                        "a_turn_index": turn_index, "a_utterance": utterance,
                        "matched_aliases": matched, "prefix": turns[:turn_index + 1],
                        "metadata_file": str(metadata_path.relative_to(root)),
                    })
    return candidates


def extract_candidates_from_turns(metadata: dict, turns: list[list[str]]) -> list[dict]:
    """再生成ステージなど、未配置の対話を同じ規則で検査する。"""
    candidates = []
    for turn_index, (speaker, utterance) in enumerate(turns):
        if speaker != "A":
            continue
        for field, value in metadata["persona"].items():
            matched = sorted({
                alias for alias in VALUE_ALIASES.get(field, {}).get(value, [])
                if alias in utterance
            })
            if matched:
                candidates.append({
                    "candidate_id": f"{metadata['id']}:{turn_index}:{field}",
                    "dialogue_id": metadata["id"], "field": field,
                    "field_label": FIELD_LABELS[field], "persona_value": value,
                    "a_turn_index": turn_index, "a_utterance": utterance,
                    "matched_aliases": matched, "prefix": turns[:turn_index + 1],
                    "metadata_file": "staging",
                })
    return candidates


def targeted_regeneration_instruction(fields: list[str]) -> str:
    labels = "、".join(FIELD_LABELS[field] for field in fields)
    return (
        "\n\n【この対話の再生成で特に修正すること】最重要:\n"
        "・前回の生成には、AがBの未開示ペルソナを先取りした問題があった。前回の本文は再利用せず、"
        "同じ設計条件から会話全体を新しく作る。\n"
        f"・特に対象となった属性は「{labels}」。Bがそれ以前の発話で具体的に明かすまでは、Aはその値を"
        "述べず、当てず、確認の前提にも使わない。曖昧な発話から補完しない。\n"
        "・この注意は対象属性だけの許可リストではない。metadata内のBの全ペルソナはAの事前知識ではなく、"
        "Aが使えるのはBがその時点までに具体的に明かした事実だけである。\n"
        "・Bに対象属性を明かすことは強制しない。必要ならAは値を含めない中立的な質問をする。"
    )


def build_contextual_final_prompt(candidate: dict, verifier: bool = False) -> str:
    """metadata一致を判断材料にしない、最終段階の文脈許容性判定。"""
    prefix = "\n".join(f"{s}{i}: {u}" for i, (s, u) in enumerate(candidate["prefix"]))
    extra = "これは独立した二回目の確認です。最初の判定は参照せず判断してください。" if verifier else ""
    return f"""あなたは日本語対話の最終品質判定者です。{extra}
次の会話は候補となったA発話までのprefixです。未来の発話は使えません。
{prefix}

最後のA発話「{candidate['a_utterance']}」が、それ以前のB発話だけから自然に発言可能か判定してください。

最重要規則:
- metadataや内部ペルソナとの一致は一切判断材料にしません。
- Bの過去発話から自然に推測できる、一般化できる、言い換えられる、または自然な確認質問として聞けるならacceptableです。
- Bが「家族」と言った後の「ご家族」、Bが「子供」と言った後の「お子さん」のように、Aが具体的な家族構成まで述べていなければacceptableです。
- 質問形でも、過去の文脈に手掛かりがなく、特定の地域・職種・家族状況などを突然正しく提示した場合はleakです。
- Bの発話から複数の解釈が可能でも、そのうち自然な推測の一つを質問として確かめることが会話上妥当ならacceptableです。
- Aが断定した内容が、Bの過去発話からは導けない場合はleakです。

JSONだけを1行で出力してください:
{{"verdict":"acceptable|leak","evidence_turns":[根拠となるBターン番号],"confidence":0.0,"reason":"短い理由"}}"""


def parse_contextual_judgement(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("JSON object not found")
    payload = re.sub(r'(?<!["\w])B(\d+)(?!["\w])', r'\1', match.group(0))
    payload = re.sub(r'"B(\d+)"', r'\1', payload)
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        verdict = re.search(r'["\']verdict["\']\s*:\s*["\'](acceptable|leak)["\']', payload)
        if not verdict:
            raise
        evidence = re.search(r'["\']evidence_turns["\']\s*:\s*\[([^]]*)\]', payload)
        confidence = re.search(r'["\']confidence["\']\s*:\s*([01](?:\.\d+)?)', payload)
        result = {"verdict": verdict.group(1),
                  "evidence_turns": [int(x) for x in re.findall(r'\d+', evidence.group(1))] if evidence else [],
                  "confidence": float(confidence.group(1)) if confidence else 0.0,
                  "reason": "JSON形式を補正して判定を回収"}
    if result.get("verdict") not in {"acceptable", "leak"}:
        raise ValueError(f"invalid verdict: {result.get('verdict')}")
    result["evidence_turns"] = [int(str(x).lstrip("B")) for x in result.get("evidence_turns", [])]
    result["confidence"] = float(result.get("confidence", 0.0))
    result["reason"] = str(result.get("reason", ""))
    return result


def extract_candidates_from_turns(metadata: dict, turns: list[list[str]]) -> list[dict]:
    """再生成ステージなど、未配置の対話を同じ規則で検査する。"""
    candidates = []
    for turn_index, (speaker, utterance) in enumerate(turns):
        if speaker != "A":
            continue
        for field, value in metadata["persona"].items():
            matched = sorted({
                alias for alias in VALUE_ALIASES.get(field, {}).get(value, [])
                if alias in utterance
            })
            if matched:
                candidates.append({
                    "candidate_id": f"{metadata['id']}:{turn_index}:{field}",
                    "dialogue_id": metadata["id"], "field": field,
                    "field_label": FIELD_LABELS[field], "persona_value": value,
                    "a_turn_index": turn_index, "a_utterance": utterance,
                    "matched_aliases": matched, "prefix": turns[:turn_index + 1],
                    "metadata_file": "staging",
                })
    return candidates


def targeted_regeneration_instruction(fields: list[str]) -> str:
    labels = "、".join(FIELD_LABELS[field] for field in fields)
    return (
        "\n\n【この対話の再生成で特に修正すること】最重要:\n"
        "・前回の生成には、AがBの未開示ペルソナを先取りした問題があった。前回の本文は再利用せず、"
        "同じ設計条件から会話全体を新しく作る。\n"
        f"・特に対象となった属性は「{labels}」。Bがそれ以前の発話で具体的に明かすまでは、Aはその値を"
        "述べず、当てず、確認の前提にも使わない。曖昧な発話から補完しない。\n"
        "・この注意は対象属性だけの許可リストではない。metadata内のBの全ペルソナはAの事前知識ではなく、"
        "Aが使えるのはBがその時点までに具体的に明かした事実だけである。\n"
        "・Bに対象属性を明かすことは強制しない。必要ならAは値を含めない中立的な質問をする。"
    )


def build_judge_prompt(candidate: dict, verifier: bool = False) -> str:
    prefix = "\n".join(f"{s}{i}: {u}" for i, (s, u) in enumerate(candidate["prefix"]))
    extra = (
        "これは曖昧例の独立再判定です。以前の判定は与えられていません。保守的かつ文脈だけで判断してください。"
        if verifier else ""
    )
    return f"""あなたは日本語対話データの品質判定者です。{extra}
Bの内部ペルソナのうち、対象項目は「{candidate['field_label']}」、設定値は「{candidate['persona_value']}」です。
この設定値は判定用の正解情報であり、Aの事前知識ではありません。

次の会話は、問題候補のA発話までしか含みません。未来の発話を想像して根拠にしてはいけません。
{prefix}

最後のA発話「{candidate['a_utterance']}」を判定してください。
- leak: それ以前のB発話に具体的な根拠がないのに、Aが設定値を知っている、正しく推測する、または前提にする。
- grounded: それ以前のB発話が対象事実を明示した、または同じ事実の明白な言い換えを述べている。
- uncertain: B発話が曖昧で、対象事実まで言えたか一意に決められない。
「地元」「仕事」「家族」だけでは具体的な地域・職種・家族構成の根拠になりません。
一般論、A自身、第三者について述べた語が偶然一致するだけならgroundedとし、内部設定の利用とは見なしません。

JSONオブジェクトだけを1行で出力してください:
{{"verdict":"leak|grounded|uncertain","evidence_turns":[Bのターン番号],"confidence":0.0,"reason":"短い理由"}}"""


def parse_judgement(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("JSON object not found")
    payload = match.group(0)
    # Gemmaは根拠番号を指示した整数でなく B7 / "B7" と返すことがある。
    payload = re.sub(r'(?<!["\w])B(\d+)(?!["\w])', r'\1', payload)
    payload = re.sub(r'"B(\d+)"', r'\1', payload)
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        # verdictだけは厳密に回収し、説明文中の引用符など軽微なJSON崩れを
        # 大量の要確認へ変換しない。verdict自体がなければ失敗させる。
        verdict_match = re.search(r'["\']verdict["\']\s*:\s*["\'](leak|grounded|uncertain)["\']', payload)
        if not verdict_match:
            raise
        evidence_match = re.search(r'["\']evidence_turns["\']\s*:\s*\[([^]]*)\]', payload)
        confidence_match = re.search(r'["\']confidence["\']\s*:\s*([01](?:\.\d+)?)', payload)
        reason_match = re.search(r'["\']reason["\']\s*:\s*["\'](.*?)["\']\s*[,}]', payload, re.S)
        result = {
            "verdict": verdict_match.group(1),
            "evidence_turns": [int(x) for x in re.findall(r'\d+', evidence_match.group(1))] if evidence_match else [],
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.0,
            "reason": reason_match.group(1) if reason_match else "JSON形式を補正して判定を回収",
        }
    if result.get("verdict") not in {"leak", "grounded", "uncertain"}:
        raise ValueError(f"invalid verdict: {result.get('verdict')}")
    result["evidence_turns"] = [int(str(x).lstrip("B")) for x in result.get("evidence_turns", [])]
    result["confidence"] = float(result.get("confidence", 0.0))
    result["reason"] = str(result.get("reason", ""))
    return result
