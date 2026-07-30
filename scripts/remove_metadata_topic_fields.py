"""既存の1件別metadataから話題ドメイン情報を除き、fixed_openerだけを残す。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".schema.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    args = parser.parse_args()
    root = Path(args.dataset_root)
    changed = 0
    for path in sorted((root / "metadata").rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        topic = data.pop("topic", None)
        topic_index = data.get("sampling", {}).pop("topic_index", None)
        if topic is not None:
            data["fixed_opener"] = topic["fixed_opener"]
        if topic is not None or topic_index is not None:
            write_atomic(path, data)
            changed += 1
    print(f"changed_metadata_files={changed}")


if __name__ == "__main__":
    main()
