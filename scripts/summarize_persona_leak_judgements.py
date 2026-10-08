#!/usr/bin/env python3
"""8シャードの判定を集約し、対話単位の再生成対象と要確認を作る。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("judgement_dir")
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retry-dir", default="", help="candidate_id単位で初回結果を置換する再判定ディレクトリ")
    args = parser.parse_args()
    records = []
    for path in sorted(Path(args.judgement_dir).glob("shard_*.jsonl")):
        records.extend(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    if args.retry_dir:
        retries = []
        for path in sorted(Path(args.retry_dir).glob("shard_*.jsonl")):
            retries.extend(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
        retry_by_id = {x["candidate_id"]: x for x in retries}
        records = [retry_by_id.get(x["candidate_id"], x) for x in records]
    ids = [x["candidate_id"] for x in records]
    if len(records) != args.expected_candidates or len(set(ids)) != len(ids):
        raise ValueError(f"result mismatch: records={len(records)} unique={len(set(ids))}")
    records.sort(key=lambda x: x["candidate_id"])
    by_dialogue = defaultdict(list)
    for record in records:
        by_dialogue[record["dialogue_id"]].append(record)
    targets = []
    review = []
    for dialogue_id, items in sorted(by_dialogue.items()):
        leaks = [x for x in items if x["final_verdict"] == "leak"]
        uncertain = [x for x in items if x["final_verdict"] == "uncertain"]
        if leaks:
            targets.append({"dialogue_id": dialogue_id,
                            "fields": sorted({x["field"] for x in leaks}),
                            "candidate_ids": [x["candidate_id"] for x in leaks]})
        if uncertain:
            review.append({"dialogue_id": dialogue_id,
                           "candidate_ids": [x["candidate_id"] for x in uncertain]})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_judgements.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    (output / "regeneration_targets.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in targets), encoding="utf-8")
    (output / "manual_review.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in review), encoding="utf-8")
    summary = {"candidates": len(records), "dialogues_with_candidates": len(by_dialogue),
               "verdicts": dict(Counter(x["final_verdict"] for x in records)),
               "regeneration_dialogues": len(targets), "manual_review_dialogues": len(review),
               "judge_errors": sum("judge_error" in x for x in records)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
