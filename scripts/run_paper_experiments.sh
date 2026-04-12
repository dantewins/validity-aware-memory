#!/usr/bin/env bash
set -euo pipefail

# Reproducible commands for the paper draft.
# Run from the repository root on a machine with the project venv activated.

export PYTHONPATH=src
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python -m memory_inference.cli synthetic \
  --reasoner local-hf \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --output results/synthetic_qwen25_7b_all.json

python -m memory_inference.cli synthetic \
  --reasoner local-hf \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --output results/synthetic_qwen25_3b_all.json

# Real-data commands require pre-structured benchmark JSON and are not runnable
# from a fresh clone of the repository because those inputs are not bundled.
#
# python -m memory_inference.cli longmemeval \
#   --input data/longmemeval_structured.json \
#   --reasoner local-hf \
#   --model-id Qwen/Qwen2.5-7B-Instruct \
#   --device cuda \
#   --dtype bfloat16 \
#   --output results/longmemeval_qwen25_7b.json
#
# python -m memory_inference.cli locomo \
#   --input data/locomo_structured.json \
#   --reasoner local-hf \
#   --model-id Qwen/Qwen2.5-7B-Instruct \
#   --device cuda \
#   --dtype bfloat16 \
#   --output results/locomo_qwen25_7b.json
