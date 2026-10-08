#!/usr/bin/env python3
"""許可英字を規定表記へ直し、dialogue/metadata/manifestのclean状態を同期する。"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_dataset_general import is_clean_general, normalize_repeated_commas, opener_ok


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".allowed-latin.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--report-name", required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    topics = [x.strip().upper() for x in args.topics.split(",") if x.strip()]
    if not topics or any(not x.startswith("T") for x in topics):
        raise ValueError("invalid topics")
    report_path = root / "provenance" / args.report_name
    if report_path.exists():
        raise FileExistsError(report_path)

    outputs = {}
    report_topics = {}
    for topic in topics:
        counts = Counter()
        for metadata_path in sorted((root / "metadata" / topic).rglob("*.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dialogue_path = root / metadata["output"]["dialogue_file"]
            turns = [json.loads(x) for x in dialogue_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            old_turns = turns
            old_clean = metadata["output"]["clean"]
            turns = normalize_repeated_commas(turns, topic)
            clean = is_clean_general(turns, topic) and opener_ok(
                turns, {"opener": metadata["fixed_opener"]}, metadata["blueprint"]["greeting_interrupt"]
            )
            changed = turns != old_turns
            counts["dialogues"] += 1
            counts["changed_dialogues"] += changed
            counts["old_clean_false"] += old_clean is False
            counts["new_clean_false"] += clean is False
            counts["rescued"] += old_clean is False and clean is True
            if changed:
                atomic_write(dialogue_path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in turns))
            new_output = {
                "clean": clean, "n_turns": len(turns),
                "n_chars": sum(len(x[1]) for x in turns),
                "allowed_latin_normalized": True,
            }
            if any(metadata["output"].get(key) != value for key, value in new_output.items()):
                metadata["output"].update(new_output)
                atomic_write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            outputs[metadata["id"]] = metadata["output"]
        report_topics[topic] = dict(counts)

    manifest_rows = 0
    for path in sorted((root / "manifest").glob("*.jsonl")):
        rows = []
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in outputs:
                out = outputs[row["id"]]
                updates = {key: out[key] for key in ("clean", "n_turns", "n_chars")}
                updates["allowed_latin_normalized"] = True
                if any(row.get(key) != value for key, value in updates.items()):
                    row.update(updates)
                    changed = True
                manifest_rows += 1
            rows.append(row)
        if changed:
            atomic_write(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))

    report = {
        "normalized_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "allowed": {
            "global": [
                "SNS", "YouTube", "URL", "ICT", "DIY", "Tシャツ", "AI", "IT", "BGM",
                "ビタミンC", "DM", "CM", "Amazon", "Web", "OS", "Google", "Switch", "RPG",
            ],
            "T09": ["VR", "AR"],
        },
        "topics": report_topics, "manifest_rows": manifest_rows,
        "raw_policy": "raw model output is preserved without normalization",
    }
    atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
