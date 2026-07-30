"""既存メタデータを生成条件として、指定番号の対話と raw だけを再生成する。"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_dataset_general import (
    PERSONA_FIELDS, add_continuers, build_prompt, is_clean_general,
    make_blueprint, normalize_repeated_commas, opener_ok, to_alternating,
    trim_incomplete_tail,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", default="0,1")
    ap.add_argument("--model-dir", default="models/instruct31b")
    ap.add_argument("--bp-seed", type=int, default=44)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, set_seed
    from gen_scenario import generate, parse_turns

    wanted = {int(x) for x in args.indices.split(",")}
    records = []
    for path in sorted((Path(args.out) / "metadata").rglob("*.json")):
        data = json.loads(path.read_text())
        match = re.fullmatch(r"gd_(t\d{2})_(\d{5})", data["id"], re.I)
        if not match or int(match.group(2)) not in wanted:
            continue
        topic_id = match.group(1).upper()
        topic_idx = int(match.group(2))
        spec = {
            "id": data["id"], "topic_id": topic_id, "topic_idx": topic_idx,
            "idx": data["sampling"]["global_index"],
            "seed": data["sampling"]["base_seed"],
            "opener": data["fixed_opener"], **data["persona"],
        }
        # 保存済み metadata を生成条件の正本とする。生成後に blueprint の
        # サンプリング率が変更されても、過去の条件を上書きしない。
        rebuilt_bp, event_texts = make_blueprint(spec, args.bp_seed)
        if rebuilt_bp["events"] != data["blueprint"]["events"]:
            raise ValueError(f"イベント指示文を再現できません: {path}")
        records.append((path, data, spec, event_texts))

    print(f"metadata records: {len(records)}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)

    for number, (mpath, data, spec, event_texts) in enumerate(records, 1):
        bp = data["blueprint"]
        params = data["output"]["gen_params"]
        prompt = build_prompt(spec, bp, event_texts)
        turns = []
        raw = ""
        for attempt in range(params["max_retries"] + 1):
            seed = spec["seed"] + attempt * 100000
            set_seed(seed)
            raw = generate(
                model, processor, prompt, params["max_new_tokens"],
                bp["gen_temperature"], params["top_p"], params["repetition_penalty"],
            )
            turns = trim_incomplete_tail(to_alternating(parse_turns(raw)))
            turns, _ = add_continuers(
                turns, random.Random(f"cont-{seed}"), bp["continuer_target"],
                protect_prefix=3 if bp["greeting_interrupt"] else 1,
            )
            turns = normalize_repeated_commas(turns)
            if is_clean_general(turns, spec["topic_id"]) and opener_ok(
                turns, spec, bp["greeting_interrupt"]
            ):
                break

        dialogue_path = Path(args.out) / data["output"]["dialogue_file"]
        raw_path = Path(args.out) / data["output"]["raw_file"]
        dialogue_tmp = dialogue_path.with_suffix(dialogue_path.suffix + ".tmp")
        raw_tmp = raw_path.with_suffix(raw_path.suffix + ".tmp")
        dialogue_tmp.write_text("".join(
            json.dumps(turn, ensure_ascii=False) + "\n" for turn in turns
        ))
        raw_tmp.write_text(raw)
        os.replace(dialogue_tmp, dialogue_path)
        os.replace(raw_tmp, raw_path)
        print(f"[{number}/{len(records)}] {spec['id']}", file=sys.stderr)


if __name__ == "__main__":
    main()
