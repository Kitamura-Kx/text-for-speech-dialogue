"""metadata.output.clean=false の対話・raw・metadataを削除しmanifestを同期する。"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_dataset_general import (
    CONF_RE, NAME_RE, REPEATED_COMMA_RE, SCAR_RE, has_forbidden_latin, opener_ok,
)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".remove-unclean.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--topics", default="T01")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--prompt-version")
    parser.add_argument("--report-name", required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir() or root.name != "general_t01_t15":
        raise ValueError(f"unexpected dataset root: {root}")

    topics = [item.strip().upper() for item in args.topics.split(",") if item.strip()]
    if not topics or any(not re.fullmatch(r"T(?:0[1-9]|1[0-5])", topic) for topic in topics):
        raise ValueError(f"invalid topics: {topics}")

    targets = []
    metadata_paths = []
    for topic in topics:
        metadata_paths.extend(sorted((root / "metadata" / topic).rglob("*.json")))
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        output = metadata.get("output", {})
        if output.get("clean") is not False:
            continue
        if args.prompt_version and output.get("prompt_version") != args.prompt_version:
            continue
        dialogue_path = root / metadata["output"]["dialogue_file"]
        raw_path = root / metadata["output"]["raw_file"]
        if not dialogue_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError(metadata["id"])
        turns = [json.loads(line) for line in dialogue_path.read_text().splitlines() if line.strip()]
        topic = metadata["id"][3:6].upper()
        reasons = set()
        for _, utterance in turns:
            if has_forbidden_latin(utterance, topic):
                reasons.add("forbidden_latin")
            if NAME_RE.search(utterance):
                reasons.add("name_or_placeholder")
            if SCAR_RE.search(utterance):
                reasons.add("scar_phrase")
            if CONF_RE.search(utterance):
                reasons.add("role_confusion")
            if REPEATED_COMMA_RE.search(utterance):
                reasons.add("repeated_comma")
        if not opener_ok(
            turns, {"opener": metadata["fixed_opener"]},
            metadata["blueprint"]["greeting_interrupt"],
        ):
            reasons.add("opener_mismatch")
        if not reasons:
            raise ValueError(f"clean=false without detected reason: {metadata['id']}")
        targets.append((metadata["id"], metadata_path, dialogue_path, raw_path, sorted(reasons)))

    if len(targets) != args.expected_count:
        raise ValueError(f"unclean count: {len(targets)} != {args.expected_count}")
    target_ids = {item[0] for item in targets}

    manifest_updates = []
    removed_manifest_ids = set()
    for manifest_path in sorted((root / "manifest").glob("*.jsonl")):
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        kept = []
        for record in records:
            if record["id"] in target_ids:
                removed_manifest_ids.add(record["id"])
            else:
                kept.append(record)
        manifest_updates.append((manifest_path, kept))
    if removed_manifest_ids != target_ids:
        raise ValueError(f"manifest mismatch: {sorted(target_ids - removed_manifest_ids)}")

    report = {
        "removed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "reason": "metadata.output.clean=false after all generation retries",
        "prompt_version": args.prompt_version,
        "topics": topics,
        "count": len(targets),
        "items": [{"id": did, "detected_reasons": reasons} for did, _, _, _, reasons in targets],
    }
    report_path = root / "provenance" / args.report_name
    if report_path.exists():
        raise FileExistsError(report_path)
    atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for manifest_path, kept in manifest_updates:
        atomic_write(
            manifest_path,
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in kept),
        )
    for _, metadata_path, dialogue_path, raw_path, _ in targets:
        dialogue_path.unlink()
        raw_path.unlink()
        metadata_path.unlink()

    print(f"removed={len(targets)} report={report_path}")


if __name__ == "__main__":
    main()
