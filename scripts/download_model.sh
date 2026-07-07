#!/usr/bin/env bash
# Download the local GGUF model. Swap REPO and FILE at kickoff when the
# required local model is announced.
set -euo pipefail

# Default: Google's QAT Q4_0 GGUF of Gemma 3 4B (Gemma locally and remotely).
# The repo is license-gated; accept it on Hugging Face and set HF_TOKEN.
REPO="${1:-google/gemma-3-4b-it-qat-q4_0-gguf}"
FILE="${2:-gemma-3-4b-it-q4_0.gguf}"

mkdir -p models
echo "downloading ${REPO}/${FILE} ..."
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
curl -fL "${AUTH[@]}" "https://huggingface.co/${REPO}/resolve/main/${FILE}" -o models/local.gguf
echo "saved to models/local.gguf"
