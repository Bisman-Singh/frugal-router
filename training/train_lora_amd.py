#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3-1.7B on the AMD notebook (ROCm), then merge.

Run inside notebooks.amd.com/hackathon in a tmux session:

    pip install -q "transformers>=4.51" peft trl datasets accelerate sentencepiece
    tmux new -s train
    python train_lora_amd.py --data sft.jsonl --out ./tuned

Budget: ~1.5-2.5h on an MI300-class GPU (checkpoints every 200 steps; if the
session dies, --resume from the last checkpoint). After it finishes, the
merged fp16 model is at ./tuned/merged — convert and quantize:

    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    pip install -q gguf
    python llama.cpp/convert_hf_to_gguf.py ./tuned/merged --outfile tuned-1p7b-f16.gguf
    cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -t llama-quantize -j
    ./llama.cpp/build/bin/llama-quantize tuned-1p7b-f16.gguf tuned-1p7b-q4km.gguf Q4_K_M

Download tuned-1p7b-q4km.gguf (~1.1GB) back to the laptop. Acceptance there:
the same 48-task gate eval the stock models faced — it ships only if coverage
beats stock-4B with confident-but-wrong <= 1, else it is discarded.
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="sft.jsonl")
    ap.add_argument("--out", default="./tuned")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print("cuda/rocm available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.base)
    tok = getattr(tok, "tokenizer", tok)  # Qwen3.5 returns a multimodal processor
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]

    def render(ex):
        # enable_thinking=False bakes the non-thinking template: the tuned
        # model answers directly, no <think> blocks, no /no_think needed.
        # Qwen2.5 templates ignore the kwarg; older ones reject it -> fall back.
        try:
            text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                           add_generation_prompt=False)
        return {"text": text}

    ds = Dataset.from_list(rows).map(render, remove_columns=["messages"])
    print("examples:", len(ds))

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        max_length=1024,
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    trainer.train(resume_from_checkpoint=args.resume)

    print("merging LoRA into fp16 base...")
    adapter_dir = f"{args.out}/adapter"
    trainer.model.save_pretrained(adapter_dir)          # adapter always saves
    tok.save_pretrained(f"{args.out}/merged")
    try:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(f"{args.out}/merged", safe_serialization=True)
    except Exception as e:
        # transformers 5.x can raise NotImplementedError inside the in-place
        # merge; re-merge from a clean fp16 base + the saved adapter instead.
        print(f"in-place merge failed ({type(e).__name__}: {e}); re-merging from clean base")
        del model, trainer
        torch.cuda.empty_cache()
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.float16, device_map="auto")
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(f"{args.out}/merged", safe_serialization=True)
    tok.save_pretrained(f"{args.out}/merged")
    print(f"DONE -> {args.out}/merged  (now convert + quantize per the header)")


if __name__ == "__main__":
    main()
