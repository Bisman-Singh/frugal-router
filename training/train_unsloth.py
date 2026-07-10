#!/usr/bin/env python3
"""Fast LoRA fine-tune via Unsloth (preinstalled on the AMD Radeon image).

Same data, same output layout as train_lora_amd.py, ~2-5x faster. Falls back
is handled by run_all.sh: if this import fails, the vanilla trainer runs.

    python train_unsloth.py --data sft.jsonl --out ./tuned --base Qwen/Qwen3.5-2B
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--data", default="sft.jsonl")
    ap.add_argument("--out", default="./tuned")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto by VRAM")
    ap.add_argument("--load-4bit", action="store_true", help="QLoRA (needed on <=16GB)")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    import torch

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    load_4bit = args.load_4bit or vram < 24
    batch = args.batch or (8 if vram < 24 else 32)
    grad_accum = 4 if vram < 24 else 1
    bf16_ok = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    print(f"GPU VRAM ~{vram:.0f}GB -> 4bit={load_4bit} batch={batch} bf16={bf16_ok} (T4=fp16)")
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=1024,
        dtype=None,   # Unsloth auto-selects fp16 (Turing/T4) or bf16 (Ampere+)
        load_in_4bit=load_4bit,
    )
    tok = getattr(tok, "tokenizer", tok)  # Qwen3.5 returns a multimodal processor
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]

    def render(ex):
        kwargs = dict(tokenize=False, add_generation_prompt=False)
        try:  # non-thinking template where the model family supports the flag
            return {"text": tok.apply_chat_template(ex["messages"], enable_thinking=False, **kwargs)}
        except TypeError:
            return {"text": tok.apply_chat_template(ex["messages"], **kwargs)}

    ds = Dataset.from_list(rows).map(render, remove_columns=["messages"])
    print("examples:", len(ds))

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_steps=100,
        save_total_limit=3,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        max_length=1024,
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    trainer.train(resume_from_checkpoint=args.resume)

    print("merging LoRA into fp16 base...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{args.out}/merged", safe_serialization=True)
    tok.save_pretrained(f"{args.out}/merged")
    print(f"DONE -> {args.out}/merged")


if __name__ == "__main__":
    main()
