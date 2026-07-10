#!/usr/bin/env bash
# End-to-end trainer for the AMD notebook. ONE command inside tmux:
#
#   tmux new -s train
#   bash run_all.sh 2>&1 | tee run_all.log
#
# Phases: deps -> pick base -> dataset -> LoRA train (ckpts) -> merge ->
# GPU eval over ~2000 graded tasks -> convert -> quantize Q4_K_M.
# Artifacts: tuned-final-q4km.gguf (download this) + eval_report.json (paste
# the printed report back). Total ~3.5-6h; every phase resumes if rerun.
set -uo pipefail

echo "== phase 0: deps =="
pip install -q "transformers>=4.51" peft trl datasets accelerate sentencepiece gguf || exit 1

echo "== phase 1: pick the strongest 2B-class base that loads =="
BASE=$(python - <<'PY'
candidates = [
    "Qwen/Qwen3.5-2B-Instruct",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3-1.7B",
]
from transformers import AutoConfig
for c in candidates:
    try:
        AutoConfig.from_pretrained(c)
        print(c)
        break
    except Exception:
        continue
PY
)
[ -z "$BASE" ] && { echo "no base model reachable"; exit 1; }
echo "base model: $BASE"

echo "== phase 2: dataset (train splits only) =="
if [ ! -f sft.jsonl ]; then
  python build_dataset_v2.py --out sft.jsonl --target 9000 || exit 1
else
  echo "sft.jsonl exists, keeping it"
fi
# graded teacher answers (upload distill.jsonl next to this script to include)
if [ -f distill.jsonl ]; then
  cat distill.jsonl >> sft.jsonl
  echo "merged $(wc -l < distill.jsonl) distilled examples into sft.jsonl"
fi

echo "== phase 3: train (Unsloth fast path, vanilla fallback; resume-safe) =="
LAST_CKPT=$(ls -d tuned/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if python -c "import unsloth" 2>/dev/null; then
  python train_unsloth.py --base "$BASE" --data sft.jsonl --out ./tuned \
      ${LAST_CKPT:+--resume "$LAST_CKPT"} || exit 1
else
  python train_lora_amd.py --base "$BASE" --data sft.jsonl --out ./tuned \
      ${LAST_CKPT:+--resume "$LAST_CKPT"} || exit 1
fi

echo "== phase 4: GPU eval over ~2000 graded tasks =="
python eval_gpu.py --model ./tuned/merged --n 2000 || exit 1

echo "== phase 5: convert + quantize =="
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./tuned/merged --outfile tuned-final-f16.gguf || exit 1
cmake -B llama.cpp/build llama.cpp >/dev/null && cmake --build llama.cpp/build -t llama-quantize -j >/dev/null
./llama.cpp/build/bin/llama-quantize tuned-final-f16.gguf tuned-final-q4km.gguf Q4_K_M || exit 1

echo ""
echo "================================================================"
echo "DONE. Download: tuned-final-q4km.gguf   Paste back: the eval table"
echo "above + eval_report.json contents."
echo "================================================================"
