#!/usr/bin/env python3
"""初回leakと最終文脈判定leakを対話単位で統合し、Topic別対象を作る。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    grouped = defaultdict(lambda: {"fields": set(), "candidate_ids": set()})
    sources = Counter()
    for path in sorted(args.audit_root.glob("T*/aggregate_final/regeneration_targets.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            item = grouped[row["dialogue_id"]]
            item["fields"].update(row["fields"])
            item["candidate_ids"].update(row["candidate_ids"])
            sources["initial_leak_rows"] += 1
    contextual = args.audit_root / "contextual_final_check/results/regeneration_targets.jsonl"
    for line in contextual.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        item = grouped[row["dialogue_id"]]
        item["fields"].update(row["fields"])
        item["candidate_ids"].update(row["candidate_ids"])
        sources["contextual_leak_rows"] += 1

    rows = [
        {"dialogue_id": did, "fields": sorted(item["fields"]),
         "candidate_ids": sorted(item["candidate_ids"])}
        for did, item in sorted(grouped.items())
    ]
    args.output_dir.mkdir(parents=True)
    by_topic = Counter()
    for row in rows:
        by_topic[row["dialogue_id"][3:6].upper()] += 1
    for topic in sorted(by_topic):
        selected = [row for row in rows if row["dialogue_id"][3:6].upper() == topic]
        topic_dir = args.output_dir / topic
        topic_dir.mkdir()
        (topic_dir / "targets.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            encoding="utf-8",
        )
    (args.output_dir / "all_targets.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {"dialogues": len(rows), "by_topic": dict(by_topic), **sources}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
