#!/usr/bin/env bash
# FLAGSHIP AMD-notebook trainer (ROCm, big VRAM -> full bf16 LoRA, NOT 4-bit).
# ONE command inside tmux (survives disconnects):
#
#   tmux new -s train
#   HF_TOKEN=... bash run_all.sh 2>&1 | tee run_all.log
#
# Phases: deps -> pick 3B base -> BIG dataset (~50k+) -> full bf16 LoRA (ckpts)
# -> merge -> GPU eval -> convert -> quantize Q4_K_M. Every phase resumes on rerun.
# Artifact: tuned-3b-amd-q4km.gguf (download this) + the printed eval table.
set -uo pipefail

echo "== phase 0: deps (don't clobber a working preinstalled ROCm stack) =="
python -c "import peft,trl,datasets,accelerate" 2>/dev/null || \
  pip install -q peft trl datasets accelerate sentencepiece gguf huggingface_hub || exit 1
[ -n "${HF_TOKEN:-}" ] && python -c "import os;from huggingface_hub import login;login(token=os.environ['HF_TOKEN'])" 2>/dev/null && echo "HF authenticated"

echo "== phase 1: pick the strongest 3B base that loads (Q4 fits the 4GB judge) =="
BASE=$(python - <<'PY'
from transformers import AutoConfig
for c in ["Qwen/Qwen2.5-3B-Instruct","unsloth/Qwen2.5-3B-Instruct","Qwen/Qwen2.5-1.5B-Instruct"]:
    try: AutoConfig.from_pretrained(c); print(c); break
    except Exception: pass
PY
)
[ -z "$BASE" ] && { echo "no base model reachable"; exit 1; }
echo "base model: $BASE"

echo "== phase 2: BIG dataset (~50k+; builder merges distill.jsonl + asserts >=20k) =="
[ -f sft.jsonl ] || python build_dataset_big.py --out sft.jsonl --target 80000 || exit 1
echo "dataset lines: $(wc -l < sft.jsonl)"

echo "== phase 3: full bf16 LoRA on the big AMD GPU (resume-safe) =="
LAST_CKPT=$(ls -d tuned/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
python train_lora_amd.py --base "$BASE" --data sft.jsonl --out ./tuned \
    --epochs 2 --batch 16 ${LAST_CKPT:+--resume "$LAST_CKPT"} || exit 1

echo "== phase 4: GPU eval over ~800 unseen graded tasks =="
python eval_gpu.py --model ./tuned/merged --n 800 || exit 1

echo "== phase 5: convert + quantize =="
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./tuned/merged --outfile tuned-3b-amd-f16.gguf || exit 1
cmake -B llama.cpp/build llama.cpp >/dev/null 2>&1 && cmake --build llama.cpp/build -t llama-quantize -j >/dev/null 2>&1
./llama.cpp/build/bin/llama-quantize tuned-3b-amd-f16.gguf tuned-3b-amd-q4km.gguf Q4_K_M || exit 1

echo "== phase 6: sized post-quant gate (GGUF vs bf16, >=300 tasks) =="
pip install -q llama-cpp-python 2>/dev/null || true
python eval_gguf.py --gguf tuned-3b-amd-q4km.gguf --n 300 --threads 8 || exit 1

echo ""
echo "================================================================"
echo "DONE. Download: tuned-3b-amd-q4km.gguf   Paste back: the eval table above."
echo "================================================================"
