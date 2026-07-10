#!/usr/bin/env bash
# HARDER run (big data + 2 epochs). Colab or Kaggle (T4/T4x2). In a cell:
#   !cd /content && rm -rf fr && git clone <repo> fr && cd fr/training && HF_TOKEN=... bash run_all_big.sh 2>&1 | tee /content/run.log
set -uo pipefail
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || pip install -q unsloth
pip install -q "transformers>=4.51" trl peft datasets accelerate sentencepiece gguf huggingface_hub || exit 1
[ -n "${HF_TOKEN:-}" ] && python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])" 2>/dev/null && echo "HF authenticated"
BASE=$(python - <<'PY'
from transformers import AutoConfig
for c in ["unsloth/Qwen2.5-3B-Instruct","unsloth/Llama-3.2-3B-Instruct","unsloth/Qwen2.5-1.5B-Instruct"]:
    try: AutoConfig.from_pretrained(c); print(c); break
    except Exception: pass
PY
)
echo "base: $BASE"
rm -f sft.jsonl   # ALWAYS rebuild the big set; never trust a committed/cached sft.jsonl
python build_dataset_big.py --out sft.jsonl --target 80000 || exit 1
echo "dataset lines: $(wc -l < sft.jsonl)"
LAST=$(ls -d tuned/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
python train_unsloth.py --base "$BASE" --data sft.jsonl --out ./tuned --load-4bit --epochs 2 \
    ${LAST:+--resume "$LAST"} || exit 1
python eval_gpu.py --model ./tuned/merged --n 800 || exit 1
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./tuned/merged --outfile tuned-f16.gguf || exit 1
cmake -B llama.cpp/build llama.cpp >/dev/null 2>&1 && cmake --build llama.cpp/build -t llama-quantize -j >/dev/null 2>&1
./llama.cpp/build/bin/llama-quantize tuned-f16.gguf tuned-3b-hard-q4km.gguf Q4_K_M || exit 1
echo "DONE. Download tuned-3b-hard-q4km.gguf"
