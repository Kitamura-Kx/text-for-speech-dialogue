"""汎用対話データ生成用の Markdown spec 読み込みと決定的サンプリング。"""
from __future__ import annotations

from dataclasses import dataclass
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PERSONA_SPEC = ROOT / "docs" / "general_dialogue_persona_spec.md"
DEFAULT_TOPIC_SPEC = ROOT / "docs" / "general_dialogue_topic_spec.md"

PERSONA_HEADINGS = {
    "性別": "gender",
    "年齢": "age",
    "居住地": "residence",
    "家族構成": "family",
    "職業": "occupation",
    "経済状況": "finances",
    "価値観": "values",
    "日常的な悩み": "concern",
    "話題への乗り気度": "interest",
}

INTEREST_CYCLE = ["乗り気"] * 5 + ["普通"] * 5 + ["あまり興味がない"]


@dataclass(frozen=True)
class TopicDomain:
    topic_id: str
    name: str
    openers: tuple[str, ...]


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_persona_options(path: str | Path = DEFAULT_PERSONA_SPEC) -> dict[str, list[str]]:
    """Markdown の各候補節から箇条書きを読み込む。"""
    options = {key: [] for key in PERSONA_HEADINGS.values()}
    current = None
    for line in _read(path).splitlines():
        m = re.match(r"^###\s+(.+?)(?:（\d+種類）)?\s*$", line)
        if m:
            current = PERSONA_HEADINGS.get(m.group(1))
            continue
        if current and line.startswith("- "):
            options[current].append(line[2:].strip())
    missing = [key for key, values in options.items() if not values]
    if missing:
        raise ValueError(f"persona spec に候補がない項目: {', '.join(missing)}")
    return options


def load_topic_domains(path: str | Path = DEFAULT_TOPIC_SPEC) -> list[TopicDomain]:
    """T01〜T15 の節と番号付き固定話題文を Markdown から読み込む。"""
    domains: list[TopicDomain] = []
    topic_id = name = None
    openers: list[str] = []

    def flush() -> None:
        nonlocal topic_id, name, openers
        if topic_id is not None:
            if not openers:
                raise ValueError(f"{topic_id} に話題提起文がない")
            domains.append(TopicDomain(topic_id, name or "", tuple(openers)))
        topic_id = name = None
        openers = []

    for line in _read(path).splitlines():
        m = re.match(r"^###\s+(T\d{2})：(.+?)\s*$", line)
        if m:
            flush()
            topic_id, name = m.groups()
            continue
        if topic_id is not None:
            m = re.match(r"^\d+\.\s+(.+?)\s*$", line)
            if m:
                openers.append(m.group(1))
    flush()

    ids = [d.topic_id for d in domains]
    expected = [f"T{i:02d}" for i in range(1, 16)]
    if ids != expected:
        raise ValueError(f"topic spec のID/順序が不正: expected={expected}, actual={ids}")
    if any(not opener.startswith("こんにちは、") for d in domains for opener in d.openers):
        raise ValueError("全話題提起文は「こんにちは、」で始める必要がある")
    return domains


# 「珍しい組み合わせは不採用」の方針に合わせた保守的な適合表。
AGE_OCCUPATIONS = {
    "10代後半": {"中学生・高校生"},
    "20代": {
        "大学生・大学院生", "専門学校生", "新卒・若手会社員", "一般会社員",
        "公務員", "教員", "医療・介護職", "エンジニア", "クリエイティブ職",
        "営業・販売職", "接客・サービス職", "製造・技術職", "運輸・物流職",
        "建設・現場職", "自営業", "フリーランス", "農林水産業",
        "パート・アルバイト", "家事・育児を主に担っている", "求職中", "休職中",
    },
    "30代": {
        "一般会社員", "管理職", "公務員", "教員", "医療・介護職", "エンジニア",
        "クリエイティブ職", "営業・販売職", "接客・サービス職", "製造・技術職",
        "運輸・物流職", "建設・現場職", "自営業", "フリーランス", "農林水産業",
        "パート・アルバイト", "家事・育児を主に担っている", "求職中", "休職中",
    },
    "40代": {
        "一般会社員", "管理職", "公務員", "教員", "医療・介護職", "エンジニア",
        "クリエイティブ職", "営業・販売職", "接客・サービス職", "製造・技術職",
        "運輸・物流職", "建設・現場職", "自営業", "フリーランス", "農林水産業",
        "パート・アルバイト", "家事・育児を主に担っている", "求職中", "休職中",
    },
    "50代": {
        "一般会社員", "管理職", "公務員", "教員", "医療・介護職", "エンジニア",
        "クリエイティブ職", "営業・販売職", "接客・サービス職", "製造・技術職",
        "運輸・物流職", "建設・現場職", "自営業", "フリーランス", "農林水産業",
        "パート・アルバイト", "家事・育児を主に担っている", "求職中", "休職中",
    },
    "60代以上": {
        "一般会社員", "管理職", "公務員", "教員", "医療・介護職", "エンジニア",
        "クリエイティブ職", "営業・販売職", "接客・サービス職", "製造・技術職",
        "運輸・物流職", "建設・現場職", "自営業", "フリーランス", "農林水産業",
        "パート・アルバイト", "家事・育児を主に担っている", "休職中",
        "定年退職後・無職",
    },
}

AGE_FAMILIES = {
    "10代後半": {"親と同居", "三世代で同居", "兄弟姉妹と同居"},
    "20代": {
        "親と同居", "一人暮らし", "配偶者・パートナーと二人暮らし",
        "配偶者・パートナーと子ども", "ひとり親として子どもと暮らしている",
        "三世代で同居", "兄弟姉妹と同居",
    },
    "30代": {
        "親と同居", "一人暮らし", "配偶者・パートナーと二人暮らし",
        "配偶者・パートナーと子ども", "ひとり親として子どもと暮らしている",
        "三世代で同居", "兄弟姉妹と同居", "高齢の親を支えながら同居",
    },
    "40代": {
        "親と同居", "一人暮らし", "配偶者・パートナーと二人暮らし",
        "配偶者・パートナーと子ども", "ひとり親として子どもと暮らしている",
        "三世代で同居", "高齢の親を支えながら同居",
    },
    "50代": {
        "一人暮らし", "配偶者・パートナーと二人暮らし",
        "配偶者・パートナーと子ども", "ひとり親として子どもと暮らしている",
        "三世代で同居", "成人した子どもと同居", "高齢の親を支えながら同居",
    },
    "60代以上": {
        "一人暮らし", "配偶者・パートナーと二人暮らし", "三世代で同居",
        "成人した子どもと同居", "高齢の親を支えながら同居",
    },
}

def _compatible(options: list[str], allowed: set[str], label: str) -> list[str]:
    values = [value for value in options if value in allowed]
    if not values:
        raise ValueError(f"{label} の適合候補が spec にない")
    return values


def sample_persona(
    options: dict[str, list[str]], idx: int, seed: int, interest: str | None = None
) -> dict[str, str]:
    """idx ごとに再現可能で、年齢との不整合を避けた B ペルソナを作る。"""
    rng = random.Random(f"general-persona-{seed}-{idx}")
    age = options["age"][idx % len(options["age"])]
    occupations = _compatible(options["occupation"], AGE_OCCUPATIONS[age], f"{age}の職業")
    families = _compatible(options["family"], AGE_FAMILIES[age], f"{age}の家族構成")
    occupation = rng.choice(occupations)
    family = rng.choice(families)
    concern_candidates = list(options["concern"])
    if occupation == "定年退職後・無職":
        concern_candidates = [c for c in concern_candidates if not c.startswith("仕事や学業")]
    persona = {
        "gender": options["gender"][(idx // len(options["age"])) % len(options["gender"])],
        "age": age,
        "residence": rng.choice(options["residence"]),
        "family": family,
        "occupation": occupation,
        "finances": rng.choice(options["finances"]),
        "values": rng.choice(options["values"]),
        "concern": rng.choice(concern_candidates),
        "interest": interest or INTEREST_CYCLE[idx % len(INTEREST_CYCLE)],
    }
    return persona


def enumerate_specs(
    count_per_topic: int,
    base_seed: int,
    persona_options: dict[str, list[str]],
    domains: list[TopicDomain],
) -> list[dict]:
    """ドメイン同数・固定文均等巡回の全生成 spec を列挙する。"""
    if count_per_topic <= 0:
        raise ValueError("count_per_topic は正数である必要がある")
    specs = []
    global_idx = 0
    for domain in domains:
        # 5:5:1 に最も近い整数割当をドメイン内で確保してから、seed付きで並べ替える。
        low_count = round(count_per_topic / 11)
        remaining = count_per_topic - low_count
        engaged_count = remaining // 2
        normal_count = remaining - engaged_count
        interests = (
            ["乗り気"] * engaged_count
            + ["普通"] * normal_count
            + ["あまり興味がない"] * low_count
        )
        random.Random(f"general-interest-{base_seed}-{domain.topic_id}").shuffle(interests)
        for topic_idx in range(count_per_topic):
            opener_idx = topic_idx % len(domain.openers)
            persona = sample_persona(
                persona_options, global_idx, base_seed, interest=interests[topic_idx]
            )
            specs.append({
                "id": f"gd_{domain.topic_id.lower()}_{topic_idx:05d}",
                "idx": global_idx,
                "topic_idx": topic_idx,
                "topic_id": domain.topic_id,
                "topic_domain": domain.name,
                "opener_idx": opener_idx,
                "opener": domain.openers[opener_idx],
                **persona,
                "seed": base_seed + global_idx,
            })
            global_idx += 1
    return specs
