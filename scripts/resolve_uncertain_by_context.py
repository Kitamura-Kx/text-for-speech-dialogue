#!/usr/bin/env python3
"""未解決uncertainをmetadata一致ではなく過去文脈からの発言可能性で最終判定する。"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persona_leak_audit import build_contextual_final_prompt, parse_contextual_judgement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="models/instruct31b")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    records = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    def run(prompt: str) -> tuple[dict, str]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}; in_len = inputs["input_ids"].shape[1]
        with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=192, do_sample=False)
        raw = getattr(processor, "tokenizer", processor).decode(out[0][in_len:], skip_special_tokens=True)
        return parse_contextual_judgement(raw), raw

    results = []
    for number, candidate in enumerate(records, 1):
        first, first_raw = run(build_contextual_final_prompt(candidate))
        second, second_raw = run(build_contextual_final_prompt(candidate, verifier=True))
        # 二回のどちらかが文脈上不可と判断したものは、安全側で再生成対象にする。
        final = "acceptable" if first["verdict"] == second["verdict"] == "acceptable" else "leak"
        results.append({**candidate, "context_first": first, "context_first_raw": first_raw,
                        "context_second": second, "context_second_raw": second_raw,
                        "context_final": final})
        print(f"{number}/{len(records)} {candidate['candidate_id']} {final}", file=sys.stderr)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "judgements.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
    target_items = defaultdict(list)
    for x in results:
        if x["context_final"] == "leak":
            target_items[x["dialogue_id"]].append(x)
    targets = [
        {
            "dialogue_id": dialogue_id,
            "fields": sorted({x["field"] for x in items}),
            "candidate_ids": [x["candidate_id"] for x in items],
        }
        for dialogue_id, items in sorted(target_items.items())
    ]
    (output / "regeneration_targets.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in targets), encoding="utf-8")
    summary = {"candidates": len(results), "verdicts": dict(Counter(x["context_final"] for x in results)),
               "regeneration_dialogues": len(targets)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
