#!/usr/bin/env python3
"""instruct31bを1 GPUへ読み込み、生成せず終了する診断用スモークテスト。"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/instruct31b")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)
    print(
        f"LOAD_OK model={type(model).__name__} processor={type(processor).__name__} "
        f"device={model.device}",
        flush=True,
    )


if __name__ == "__main__":
    main()
