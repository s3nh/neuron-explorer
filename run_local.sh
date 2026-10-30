#!/usr/bin/env bash
# Local run of the neuron explorer on a small model.
#
#   bash run_local.sh                          # Qwen/Qwen3.5-0.8B (hybrid, ~1.6 GB download)
#   bash run_local.sh HuggingFaceTB/SmolLM2-135M-Instruct   # any other HF causal LM
#
# Then open http://localhost:8791
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-Qwen/Qwen3.5-0.8B}"
PORT="${PORT:-8791}"

# python environment (uv: https://docs.astral.sh/uv/)
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python 3.11
fi
uv pip install -p .venv/bin/python -q \
    torch "transformers>=4.50" safetensors numpy accelerate fastapi "uvicorn>=0.30" \
    datasets sentence-transformers umap-learn

echo "model: $MODEL"
echo "open:  http://localhost:$PORT"
EXPLORER_MODEL="$MODEL" PORT="$PORT" .venv/bin/python explorer/server.py
