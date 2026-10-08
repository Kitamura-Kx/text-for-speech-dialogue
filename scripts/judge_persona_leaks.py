#!/usr/bin/env python3
"""候補をGemmaで判定し、uncertainだけ独立プロンプトで再判定する。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persona_leak_audit import build_judge_prompt, parse_judgement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="models/instruct31b")
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--only-errors-from", default="",
                        help="指定した既存判定JSONL群でjudge_errorだった候補だけ再実行する")
    args = parser.parse_args()
    if not 0 <= args.shard < args.nshard:
        raise ValueError("invalid shard")

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    records = [json.loads(x) for x in Path(args.candidates).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.only_errors_from:
        error_ids = set()
        for path in Path(args.only_errors_from).glob("shard_*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                old = json.loads(line)
                if "judge_error" in old:
                    error_ids.add(old["candidate_id"])
        records = [x for x in records if x["candidate_id"] in error_ids]
    mine = [x for i, x in enumerate(records) if i % args.nshard == args.shard]
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    def run(prompt: str) -> tuple[dict, str]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        tokenizer = getattr(processor, "tokenizer", processor)
        raw = tokenizer.decode(generated[0][in_len:], skip_special_tokens=True)
        return parse_judgement(raw), raw

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for number, candidate in enumerate(mine, 1):
            try:
                first, first_raw = run(build_judge_prompt(candidate))
                second = second_raw = None
                if first["verdict"] == "uncertain":
                    second, second_raw = run(build_judge_prompt(candidate, verifier=True))
                final = second["verdict"] if second else first["verdict"]
                record = {**candidate, "first_judgement": first, "first_raw": first_raw,
                          "second_judgement": second, "second_raw": second_raw,
                          "final_verdict": final, "needs_regeneration": final == "leak"}
            except Exception as exc:
                record = {**candidate, "judge_error": repr(exc), "final_verdict": "uncertain",
                          "needs_regeneration": False}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 10 == 0:
                print(f"shard={args.shard} {number}/{len(mine)}", file=sys.stderr)


if __name__ == "__main__":
    main()
