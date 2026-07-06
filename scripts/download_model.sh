#!/usr/bin/env bash
# Download the local GGUF model. Swap REPO and FILE at kickoff when the
# required local model is announced.
set -euo pipefail

REPO="${1:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
FILE="${2:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"

mkdir -p models
echo "downloading ${REPO}/${FILE} ..."
curl -fL "https://huggingface.co/${REPO}/resolve/main/${FILE}" -o models/local.gguf
echo "saved to models/local.gguf"
