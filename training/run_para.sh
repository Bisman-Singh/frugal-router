#!/bin/bash
# HOST orchestrator (MI300X droplet): paraphrase-augmented 4B retrain, end to end.
#   HF_TOKEN=... bash run_para.sh 2>&1 | tee /root/work/para.log
# Phases: P1 core dataset -> P2 paraphrase on GPU (vLLM 32B, transformers
# fallback) -> P3 assemble -> P4 train 4B -> P5 eval -> P6 convert+quantize.
set -uo pipefail
W=/root/work
PT=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
VL=rocm/vllm:latest
DOCK="docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 992 \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -v $W:/work -w /work/fr4/training -e HF_TOKEN=${HF_TOKEN:?set HF_TOKEN} -e HF_HOME=/work/hfcache"

echo "== P0: pull vLLM image in background =="
docker pull $VL >/tmp/vllm_pull.log 2>&1 &
PULL=$!

echo "== P1: core dataset (judge formats, typed NER) =="
$DOCK $PT bash -lc "pip install -q datasets sentencepiece huggingface_hub && python build_dataset_para.py --gens-per-cat 2500" || exit 1

echo "== P2: paraphrase on the GPU =="
if wait $PULL; then
  $DOCK $VL bash -lc "python paraphrase_vllm.py --k 3" || \
  $DOCK $PT bash -lc "pip install -q accelerate sentencepiece && python paraphrase_vllm.py --k 3 --fallback" || exit 1
else
  echo "vLLM pull failed -> transformers fallback"
  $DOCK $PT bash -lc "pip install -q accelerate sentencepiece && python paraphrase_vllm.py --k 3 --fallback" || exit 1
fi

echo "== P3: assemble =="
$DOCK $PT bash -lc "ABSTAIN_FRAC=0.08 python assemble_para.py --target 90000" || exit 1

echo "== P4: train 4B (bf16 LoRA, resume-safe) =="
LAST=$(ls -d $W/fr4/training/tuned/checkpoint-* 2>/dev/null | sort -V | tail -1 | sed "s#$W#/work#" || true)
$DOCK $PT bash -lc "pip install -q peft trl datasets accelerate sentencepiece gguf cmake ninja && \
  python train_lora_amd.py --base Qwen/Qwen3-4B-Instruct-2507 --data sft.jsonl --out ./tuned \
    --epochs 2 --batch 32 ${LAST:+--resume $LAST}" || exit 1

echo "== P5: held-out eval =="
$DOCK $PT bash -lc "python eval_gpu.py --model ./tuned/merged --n 800" || exit 1

echo "== P6: convert + quantize =="
$DOCK $PT bash -lc "[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggerganov/llama.cpp; \
  pip install -q cmake ninja gguf && \
  python llama.cpp/convert_hf_to_gguf.py ./tuned/merged --outfile para-4b-f16.gguf && \
  cmake -B llama.cpp/build llama.cpp -DLLAMA_CURL=OFF >/dev/null && \
  cmake --build llama.cpp/build -t llama-quantize -j >/dev/null && \
  ./llama.cpp/build/bin/llama-quantize para-4b-f16.gguf para-4b-q4km.gguf Q4_K_M" || exit 1

echo "ALL DONE -> /root/work/fr4/training/para-4b-q4km.gguf"
echo "=== PARA EXIT 0 ==="
