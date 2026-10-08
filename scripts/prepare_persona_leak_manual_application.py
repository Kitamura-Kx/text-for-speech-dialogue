#!/usr/bin/env python3
"""手動判定を検証し、Topic別の反映・削除対象を作る。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    review = json.loads(args.review.read_text(encoding="utf-8"))
    contextual = set(review["accepted_contextually"])
    rejected = set(review["rejected"])
    targets = rows(args.audit_root / "regeneration/all_targets.jsonl")
    targets = [row for row in targets if row["dialogue_id"][3:6].upper() != "T02"]
    target_by_id = {row["dialogue_id"]: row for row in targets}

    failed = set()
    for topic_num in range(3, 16):
        topic = f"T{topic_num:02d}"
        for path in (args.audit_root / "regeneration" / topic / "staging").glob("gd_*/result.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            if not result["accepted"]:
                failed.add(result["dialogue_id"])

    candidate_by_id = {}
    for topic_num in range(3, 16):
        topic = f"T{topic_num:02d}"
        for candidate in rows(args.audit_root / topic / "candidates.jsonl"):
            if candidate["dialogue_id"] in failed:
                candidate_by_id[candidate["candidate_id"]] = candidate

    fixed = set()
    for dialogue_id in failed:
        target = target_by_id[dialogue_id]
        topic = dialogue_id[3:6].upper()
        metadata_path = next((args.dataset_root / "metadata" / topic).rglob(dialogue_id + ".json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        opener = metadata["fixed_opener"]
        rest = opener[len("こんにちは、"):] if opener.startswith("こんにちは、") else opener
        candidates = [candidate_by_id[cid] for cid in target["candidate_ids"] if cid in candidate_by_id]
        if candidates and all(
            candidate["field"] == "concern"
            and candidate["a_utterance"] in {opener, rest}
            for candidate in candidates
        ):
            fixed.add(dialogue_id)

    if len(fixed) != review["summary"]["accepted_fixed_opener"]:
        raise ValueError(f"fixed count mismatch: {len(fixed)}")
    if failed != fixed | contextual | rejected:
        raise ValueError(
            f"review partition mismatch: missing={sorted(failed-(fixed|contextual|rejected))}, "
            f"extra={sorted((fixed|contextual|rejected)-failed)}"
        )
    if any((fixed & contextual, fixed & rejected, contextual & rejected)):
        raise ValueError("review sets overlap")

    retained = fixed | contextual
    actionable = [row for row in targets if row["dialogue_id"] not in retained]
    by_topic = defaultdict(list)
    for row in actionable:
        by_topic[row["dialogue_id"][3:6].upper()].append(row)
    args.output_dir.mkdir(parents=True)
    for topic, topic_rows in sorted(by_topic.items()):
        topic_dir = args.output_dir / topic
        topic_dir.mkdir()
        (topic_dir / "targets.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in topic_rows),
            encoding="utf-8",
        )
    summary = {
        "total_targets": len(targets),
        "staged_to_apply": len(targets) - len(failed),
        "retained_originals": len(retained),
        "removed_originals": len(rejected),
        "actionable_targets": len(actionable),
        "retained_ids": sorted(retained),
        "rejected_ids": sorted(rejected),
        "actionable_by_topic": dict(sorted(Counter(
            row["dialogue_id"][3:6].upper() for row in actionable
        ).items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
