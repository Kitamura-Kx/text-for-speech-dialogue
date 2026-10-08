"""退避した旧T01対話を、指定オフセットから統合データセットへ安全に移行する。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil


ID_RE = re.compile(r"gd_t01_(\d{5})$")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".migration.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".migration.tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    target = args.target.resolve()
    if source == target or not source.is_dir() or not target.is_dir():
        raise ValueError("source/target must be different existing directories")

    metadata_paths = sorted((source / "metadata/T01").rglob("gd_t01_*.json"))
    if len(metadata_paths) != args.expected_count:
        raise ValueError(f"metadata count: {len(metadata_paths)} != {args.expected_count}")

    source_manifests: dict[str, dict] = {}
    for path in sorted((source / "manifest").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in source_manifests:
                raise ValueError(f"duplicate source manifest id: {record['id']}")
            source_manifests[record["id"]] = record
    if len(source_manifests) != args.expected_count:
        raise ValueError(f"manifest count: {len(source_manifests)} != {args.expected_count}")

    migrations = []
    old_indices = []
    for rank, metadata_path in enumerate(metadata_paths):
        metadata = json.loads(metadata_path.read_text())
        match = ID_RE.fullmatch(metadata["id"])
        if not match:
            raise ValueError(f"invalid id: {metadata['id']}")
        old_index = int(match.group(1))
        old_indices.append(old_index)
        # 旧データは8シャードの途中停止成果物なので旧IDに欠番がある。
        # 旧ID順を保ったまま、統合先ではoffsetから隙間のない連番にする。
        new_index = args.offset + rank
        new_id = f"gd_t01_{new_index:05d}"
        bucket = f"{new_index // 1000 * 1000:05d}"
        dialogue_src = source / metadata["output"]["dialogue_file"]
        raw_src = source / metadata["output"]["raw_file"]
        dialogue_dst = target / f"dialogues/T01/{bucket}/{new_id}.jsonl"
        metadata_dst = target / f"metadata/T01/{bucket}/{new_id}.json"
        raw_dst = target / f"raw/T01/{new_id}.txt"
        if not dialogue_src.is_file() or not raw_src.is_file():
            raise FileNotFoundError(metadata["id"])
        if dialogue_dst.exists() or metadata_dst.exists() or raw_dst.exists():
            raise FileExistsError(new_id)
        migrations.append((metadata, dialogue_src, raw_src, dialogue_dst, metadata_dst, raw_dst, new_index, new_id, bucket))

    if old_indices != sorted(set(old_indices)):
        raise ValueError("source ids are not unique and sorted")

    additions: dict[int, list[dict]] = {i: [] for i in range(8)}
    for metadata, dialogue_src, raw_src, dialogue_dst, metadata_dst, raw_dst, new_index, new_id, bucket in migrations:
        dialogue_dst.parent.mkdir(parents=True, exist_ok=True)
        raw_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dialogue_src, dialogue_dst)
        shutil.copy2(raw_src, raw_dst)

        old_id = metadata["id"]
        metadata["id"] = new_id
        metadata["sampling"]["global_index"] = new_index
        metadata["output"]["dialogue_file"] = f"dialogues/T01/{bucket}/{new_id}.jsonl"
        metadata["output"]["raw_file"] = f"raw/T01/{new_id}.txt"
        metadata["migration"] = {
            "source_id": old_id,
            "source_dataset": source.name,
            "reason": "append pre-persona-fix pilot after T01 main generation",
        }
        atomic_json(metadata_dst, metadata)

        record = dict(source_manifests[old_id])
        record["id"] = new_id
        record["idx"] = new_index
        record["topic_idx"] = new_index
        record["file"] = f"dialogues/T01/{bucket}/{new_id}.jsonl"
        record["metadata_file"] = f"metadata/T01/{bucket}/{new_id}.json"
        record["migration"] = metadata["migration"]
        additions[new_index % 8].append(record)

    for shard in range(8):
        path = target / f"manifest/shard_{shard:04d}_of_0008.jsonl"
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        atomic_jsonl(path, existing + additions[shard])

    provenance = target / "provenance/pre_persona_fix"
    if provenance.exists():
        raise FileExistsError(provenance)
    provenance.mkdir(parents=True)
    for name in ("config", "logs", "manifest"):
        src = source / name
        if src.exists():
            shutil.copytree(src, provenance / f"original_{name}")
    atomic_json(provenance / "migration.json", {
        "source_dataset": source.name,
        "target_dataset": target.name,
        "offset": args.offset,
        "count": args.expected_count,
        "new_id_first": f"gd_t01_{args.offset:05d}",
        "new_id_last": f"gd_t01_{args.offset + args.expected_count - 1:05d}",
        "source_removed_after_verified_migration": True,
    })

    # Verify every migrated triplet before removing the source tree.
    for _, _, _, dialogue_dst, metadata_dst, raw_dst, _, new_id, _ in migrations:
        metadata = json.loads(metadata_dst.read_text())
        if metadata["id"] != new_id or not dialogue_dst.is_file() or not raw_dst.is_file():
            raise RuntimeError(f"verification failed: {new_id}")
    shutil.rmtree(source)
    print(f"migrated={len(migrations)} first=gd_t01_{args.offset:05d} "
          f"last=gd_t01_{args.offset + len(migrations) - 1:05d} source_removed={not source.exists()}")


if __name__ == "__main__":
    main()
