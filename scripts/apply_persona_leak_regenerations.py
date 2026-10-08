#!/usr/bin/env python3
"""検査済みstagingをprovenanceへ退避しつつ本体とmanifestへ一括反映する。"""
import argparse
import datetime
import json
import os
import shutil
from pathlib import Path


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".persona-leak.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument(
        "--remove-failed", action="store_true",
        help="不合格対象を旧データ退避後に本体とmanifestから同期削除する",
    )
    args = parser.parse_args()
    root, stage, archive = Path(args.dataset_root), Path(args.stage_dir), Path(args.archive_dir)
    targets = [json.loads(x) for x in Path(args.targets).read_text(encoding="utf-8").splitlines() if x.strip()]
    target_ids = {x["dialogue_id"] for x in targets}
    staged = {}
    failed = []
    for dialogue_id in sorted(target_ids):
        directory = stage / dialogue_id
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if not result["accepted"]:
            failed.append(dialogue_id)
            continue
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["id"] != dialogue_id or not metadata["output"]["clean"]:
            raise ValueError(f"invalid staged metadata: {dialogue_id}")
        turns = [json.loads(x) for x in (directory / "dialogue.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(turns) != metadata["output"]["n_turns"] or sum(len(x[1]) for x in turns) != metadata["output"]["n_chars"]:
            raise ValueError(f"staged counts mismatch: {dialogue_id}")
        staged[dialogue_id] = (directory, metadata)
    if failed and not args.remove_failed:
        raise ValueError(f"regeneration not accepted for {len(failed)} targets; dataset unchanged: {failed[:10]}")
    if set(staged) | set(failed) != target_ids:
        raise ValueError("staging target mismatch")

    manifests = {}
    seen = set()
    for path in sorted((root / "manifest").glob("*.jsonl")):
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        kept = []
        for row in rows:
            if row["id"] in target_ids:
                seen.add(row["id"])
                if row["id"] in failed:
                    continue
                metadata = staged[row["id"]][1]
                out = metadata["output"]
                row.update({key: out[key] for key in (
                    "n_turns", "n_chars", "clean", "seed_used", "n_attempts", "continuer_added",
                    "model_dir", "prompt_version", "gen_params")})
                row["regeneration"] = metadata["regeneration"]
            kept.append(row)
        manifests[path] = kept
    if seen != target_ids:
        raise ValueError(f"manifest target mismatch: missing={sorted(target_ids-seen)[:10]}")

    archive.mkdir(parents=True, exist_ok=False)
    for dialogue_id, (directory, metadata) in staged.items():
        # source path is recorded in result rather than regenerated metadata.
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        old_metadata_path = root / result["source_metadata_file"]
        old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
        old_dialogue = root / old_metadata["output"]["dialogue_file"]
        old_raw = root / old_metadata["output"]["raw_file"]
        saved = archive / dialogue_id
        saved.mkdir(parents=True)
        shutil.copy2(old_dialogue, saved / "dialogue.jsonl")
        shutil.copy2(old_raw, saved / "raw.txt")
        shutil.copy2(old_metadata_path, saved / "metadata.json")
        shutil.copy2(directory / "result.json", saved / "regeneration_result.json")
        atomic_text(old_dialogue, (directory / "dialogue.jsonl").read_text(encoding="utf-8"))
        atomic_text(old_raw, (directory / "raw.txt").read_text(encoding="utf-8"))
        atomic_text(old_metadata_path, (directory / "metadata.json").read_text(encoding="utf-8"))
    for dialogue_id in failed:
        directory = stage / dialogue_id
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        old_metadata_path = root / result["source_metadata_file"]
        old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
        old_dialogue = root / old_metadata["output"]["dialogue_file"]
        old_raw = root / old_metadata["output"]["raw_file"]
        saved = archive / dialogue_id
        saved.mkdir(parents=True)
        shutil.copy2(old_dialogue, saved / "dialogue.jsonl")
        shutil.copy2(old_raw, saved / "raw.txt")
        shutil.copy2(old_metadata_path, saved / "metadata.json")
        shutil.copy2(directory / "result.json", saved / "regeneration_result.json")
        old_dialogue.unlink()
        old_raw.unlink()
        old_metadata_path.unlink()
    for path, rows in manifests.items():
        atomic_text(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    report = {"applied_at": datetime.datetime.now().isoformat(timespec="seconds"),
              "count": len(staged), "ids": sorted(staged),
              "removed_count": len(failed), "removed_ids": sorted(failed)}
    (archive / "apply_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"applied": len(staged), "removed": len(failed),
                      "archive": str(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
