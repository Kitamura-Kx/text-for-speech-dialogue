#!/usr/bin/env python3
"""leak確定対話を同一metadata条件・新seedで安全なstagingへ全体再生成する。"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_dataset_general import (add_continuers, build_prompt, is_clean_general, make_blueprint,
                                 normalize_repeated_commas, opener_ok, to_alternating, trim_incomplete_tail)
from persona_leak_audit import (build_judge_prompt, extract_candidates_from_turns,
                                parse_judgement, targeted_regeneration_instruction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--model-dir", default="models/instruct31b")
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-regeneration-attempts", type=int, default=4)
    parser.add_argument("--seed-round", type=int, default=0)
    parser.add_argument("--audit-reference", default="provenance/persona_leak_audit_20260802")
    parser.add_argument("--only-missing", action="store_true",
                        help="accepted済みのstagingを残し、未処理・不合格だけ実行する")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, set_seed
    from gen_scenario import generate, parse_turns

    targets = [json.loads(x) for x in Path(args.targets).read_text(encoding="utf-8").splitlines() if x.strip()]
    root, stage = Path(args.dataset_root), Path(args.stage_dir)
    if args.only_missing:
        remaining = []
        for target in targets:
            result_path = stage / target["dialogue_id"] / "result.json"
            if result_path.is_file() and json.loads(result_path.read_text(encoding="utf-8")).get("accepted"):
                continue
            remaining.append(target)
        targets = remaining
    mine = [x for i, x in enumerate(targets) if i % args.nshard == args.shard]
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    def judge(candidate: dict) -> str:
        prompt = build_judge_prompt(candidate)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=192, do_sample=False)
        raw = getattr(processor, "tokenizer", processor).decode(out[0][in_len:], skip_special_tokens=True)
        first = parse_judgement(raw)["verdict"]
        if first != "uncertain":
            return first
        # uncertainは独立再判定。再びuncertainなら採用しない。
        verify = build_judge_prompt(candidate, verifier=True)
        messages = [{"role": "user", "content": [{"type": "text", "text": verify}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=192, do_sample=False)
        raw = getattr(processor, "tokenizer", processor).decode(out[0][in_len:], skip_special_tokens=True)
        return parse_judgement(raw)["verdict"]

    for number, target in enumerate(mine, 1):
        metadata_path = next((root / "metadata").rglob(target["dialogue_id"] + ".json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        match_topic = metadata["output"]["dialogue_file"].split("/")[1]
        topic_idx = int(target["dialogue_id"].rsplit("_", 1)[1])
        blueprint_idx = int(metadata["sampling"]["global_index"])
        if metadata.get("migration", {}).get("source_id"):
            blueprint_idx = int(metadata["migration"]["source_id"].rsplit("_", 1)[1])
        spec = {"id": metadata["id"], "topic_id": match_topic, "topic_idx": topic_idx,
                "idx": blueprint_idx, "seed": metadata["sampling"]["base_seed"],
                "opener": metadata["fixed_opener"], **metadata["persona"]}
        rebuilt_bp, event_texts = make_blueprint(spec, 44)
        if rebuilt_bp["events"] != metadata["blueprint"]["events"]:
            raise ValueError(f"cannot rebuild event instructions: {metadata_path}")
        bp, params = metadata["blueprint"], metadata["output"]["gen_params"]
        prompt = build_prompt(spec, bp, event_texts) + targeted_regeneration_instruction(target["fields"])
        attempts = []
        accepted = None
        for attempt in range(args.max_regeneration_attempts):
            seed = (int(metadata["output"]["seed_used"]) + 1000000
                    + (args.seed_round * args.max_regeneration_attempts + attempt) * 100000)
            set_seed(seed)
            raw = generate(model, processor, prompt, params["max_new_tokens"], bp["gen_temperature"], params["top_p"], params["repetition_penalty"])
            turns = trim_incomplete_tail(to_alternating(parse_turns(raw)))
            turns, added = add_continuers(turns, random.Random(f"cont-{seed}"), bp["continuer_target"], protect_prefix=3 if bp["greeting_interrupt"] else 1)
            turns = normalize_repeated_commas(turns, spec["topic_id"])
            clean = is_clean_general(turns, spec["topic_id"]) and opener_ok(turns, spec, bp["greeting_interrupt"])
            verdicts = []
            if clean:
                try:
                    verdicts = [judge(x) for x in extract_candidates_from_turns(metadata, turns)]
                except Exception as exc:
                    verdicts = ["uncertain"]
                    attempts.append({"seed": seed, "clean": clean, "judge_error": repr(exc)})
            attempts.append({"seed": seed, "clean": clean, "candidate_verdicts": verdicts})
            if clean and not any(x in {"leak", "uncertain"} for x in verdicts):
                accepted = (seed, raw, turns, added)
                break
        outdir = stage / target["dialogue_id"]
        outdir.mkdir(parents=True, exist_ok=True)
        result = {"dialogue_id": target["dialogue_id"], "fields": target["fields"],
                  "source_metadata_file": str(metadata_path.relative_to(root)), "attempts": attempts,
                  "accepted": accepted is not None, "target_candidate_ids": target["candidate_ids"]}
        if accepted:
            seed, raw, turns, added = accepted
            (outdir / "dialogue.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in turns), encoding="utf-8")
            (outdir / "raw.txt").write_text(raw, encoding="utf-8")
            updated = json.loads(json.dumps(metadata))
            updated["output"].update({"n_turns": len(turns), "n_chars": sum(len(x[1]) for x in turns),
                                      "clean": True, "seed_used": seed, "n_attempts": len(attempts),
                                      "continuer_added": added,
                                      "prompt_version": "general-dialogue-blueprint-v2-no-persona-leak-targeted-regeneration"})
            updated["regeneration"] = {"reason": "persona_leak", "fields": target["fields"],
                                       "previous_seed": metadata["output"]["seed_used"], "new_seed": seed,
                                       "audit": args.audit_reference, "attempts": attempts}
            (outdir / "metadata.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (outdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{number}/{len(mine)} {target['dialogue_id']} accepted={result['accepted']}", file=sys.stderr)


if __name__ == "__main__":
    main()
