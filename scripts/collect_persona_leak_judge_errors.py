#!/usr/bin/env python3
"""Topic別判定結果からjudge_error候補だけを一つの再判定入力へ集約する。"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records = []
    for path in sorted(args.audit_root.glob("T*/judgements/shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "judge_error" in record:
                for key in (
                    "judge_error", "first_judgement", "first_raw", "second_judgement",
                    "second_raw", "final_verdict", "needs_regeneration",
                ):
                    record.pop(key, None)
                records.append(record)
    ids = [record["candidate_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"errors": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
