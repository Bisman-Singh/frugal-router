#!/usr/bin/env bash
# Colab / Kaggle (NVIDIA T4/P100) end-to-end runner. In a notebook cell:
#   !bash run_all_cuda.sh 2>&1 | tee run.log
set -uo pipefail

echo "== deps (CUDA Unsloth) =="
[ -n "${HF_TOKEN:-}" ] && python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])" 2>/dev/null && echo "HF authenticated"
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || pip install -q unsloth
pip install -q "transformers>=4.51" trl peft datasets accelerate sentencepiece gguf || exit 1

echo "== base model =="
BASE=$(python - <<'PY'
from transformers import AutoConfig
for c in ["unsloth/Qwen2.5-3B-Instruct","unsloth/Llama-3.2-3B-Instruct","unsloth/Qwen2.5-1.5B-Instruct"]:
    try: AutoConfig.from_pretrained(c); print(c); break
    except Exception: pass
PY
)
echo "base: $BASE"

echo "== dataset (BIG builder: real benchmarks via load_dataset; 20k, ~1.8h on 1x T4) =="
rm -f sft.jsonl   # ALWAYS rebuild; never trust a committed/cached sft.jsonl
python build_dataset_big.py --out sft.jsonl --target 20000 || exit 1  # merges distill + asserts >=20k
echo "dataset lines: $(wc -l < sft.jsonl)"

echo "== train (QLoRA, T4-sized, checkpointed) =="
LAST=$(ls -d tuned/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
python train_unsloth.py --base "$BASE" --data sft.jsonl --out ./tuned --load-4bit \
    ${LAST:+--resume "$LAST"} || exit 1

echo "== eval (2000 graded, unseen) =="
python eval_gpu.py --model ./tuned/merged --n 600 || exit 1

echo "== convert + quantize =="
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./tuned/merged --outfile tuned-f16.gguf || exit 1
cmake -B llama.cpp/build llama.cpp -DLLAMA_CURL=OFF -DGGML_NATIVE=ON || exit 1
cmake --build llama.cpp/build -t llama-quantize -j || exit 1
./llama.cpp/build/bin/llama-quantize tuned-f16.gguf tuned-3b-q4km.gguf Q4_K_M || exit 1

echo "== sized GGUF gate =="
pip install -q llama-cpp-python 2>/dev/null || true
python eval_gguf.py --gguf tuned-3b-q4km.gguf --n 300 --threads 2 || true

echo "DONE. Download tuned-3b-q4km.gguf; paste the eval tables back."
