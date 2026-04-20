#!/usr/bin/env bash
set -euo pipefail

# Reproducible Mem0 feature-ablation runs across LongMemEval and LoCoMo.
# Run from the repository root on the Brev machine with the project env active.

bash scripts/run_mem0_ablation_longmemeval.sh
bash scripts/run_mem0_ablation_locomo.sh
