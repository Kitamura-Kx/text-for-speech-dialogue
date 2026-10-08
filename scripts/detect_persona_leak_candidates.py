#!/usr/bin/env python3
"""T01等の対話からペルソナ先取りの高再現率候補をJSONLへ抽出する。"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persona_leak_audit import extract_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--topic", default="T01")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = extract_candidates(Path(args.dataset_root), args.topic.upper())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in candidates), encoding="utf-8")
    print(json.dumps({
        "candidates": len(candidates),
        "dialogues": len({x["dialogue_id"] for x in candidates}),
        "by_field": Counter(x["field"] for x in candidates),
    }, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
