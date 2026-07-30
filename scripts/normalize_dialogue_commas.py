"""既存汎用データの連続読点を正規化し、付随メタデータを同期する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


REPEATED_COMMA_RE = re.compile(r"、{2,}")


def normalize(text: str) -> str:
    text = REPEATED_COMMA_RE.sub("、", text)
    return re.sub(r"、([。！？!?])", r"\1", text)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".normalize.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    counts: dict[str, tuple[int, int]] = {}
    changed_files = changed_turns = 0

    for path in sorted((root / "dialogues").rglob("*.jsonl")):
        output = []
        file_changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            speaker, utterance = json.loads(line)
            updated = normalize(utterance)
            file_changed |= updated != utterance
            changed_turns += updated != utterance
            output.append([speaker, updated])
        if file_changed:
            write_atomic(
                path,
                "".join(json.dumps(turn, ensure_ascii=False) + "\n" for turn in output),
            )
            changed_files += 1
        counts[path.stem] = (len(output), sum(len(turn[1]) for turn in output))

    for path in sorted((root / "metadata").rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["id"] not in counts or not data.get("output"):
            continue
        n_turns, n_chars = counts[data["id"]]
        data["output"]["n_turns"] = n_turns
        data["output"]["n_chars"] = n_chars
        data["output"]["comma_normalized"] = True
        write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    for path in sorted((root / "manifest").glob("*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            if data["id"] in counts:
                data["n_turns"], data["n_chars"] = counts[data["id"]]
                data["comma_normalized"] = True
            records.append(data)
        write_atomic(
            path,
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        )

    print(f"changed_files={changed_files} changed_turns={changed_turns}")


if __name__ == "__main__":
    main()
