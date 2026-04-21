#!/bin/bash
# Evaluation script template for MMR-GRPO (pass@1 with n=16)
#
# NOTE: Use the mmr_grpo_eval environment for evaluation.
#
# Usage:
#   bash scripts/eval.sh <model> <step>
#
# Examples:
#   bash scripts/eval.sh your_hf_username/MMR-GRPO 150
#   bash scripts/eval.sh your_hf_username/MMR-DR-GRPO-7B 50
#   bash scripts/eval.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   # baseline (no step)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [step]"
    echo ""
    echo "Arguments:"
    echo "  model  - HuggingFace model name (e.g., your_hf_username/MMR-GRPO)"
    echo "  step   - (Optional) Training step / revision to evaluate"
    echo ""
    echo "Benchmarks: AIME 2024, MATH-500, AMC 2023, Minerva, OlympiadBench"
    exit 1
fi

MODEL=$1
STEP=${2:-""}

MODEL_NAME=$(basename "$MODEL")
TASKS="aime24_at_k math_500_at_k amc23_at_k minerva_at_k olympiadbench_at_k"

# Build model args
if [ -n "$STEP" ]; then
    OUTPUT_DIR="logs/evals/${MODEL_NAME}_step${STEP}"
    MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95},revision=step_${STEP}"
else
    OUTPUT_DIR="logs/evals/${MODEL_NAME}"
    MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
fi

echo "========================================"
echo "Running evaluation (pass@1, n=16)"
echo "Model: $MODEL"
[ -n "$STEP" ] && echo "Step: $STEP"
echo "Output: $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

lighteval vllm "$MODEL_ARGS" \
    "aime24_at_k|0,math_500_at_k|0,amc23_at_k|0,minerva_at_k|0,olympiadbench_at_k|0" \
    --custom-tasks src/open_r1/evaluate_at_k.py \
    --output-dir "$OUTPUT_DIR"
