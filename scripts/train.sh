#!/bin/bash
# Training script template for MMR-GRPO
#
# Usage:
#   bash scripts/train.sh <training_script> <recipe>
#
# Examples:
#   bash scripts/train.sh src/open_r1/grpo.py recipes/mmr_grpo.yaml         # MMR-GRPO 1.5B
#   bash scripts/train.sh src/open_r1/grpo.py recipes/mmr_grpo_7b.yaml      # MMR-GRPO 7B
#   bash scripts/train.sh src/open_r1/drgrpo.py recipes/mmr_dr_grpo.yaml    # MMR-DR-GRPO 1.5B
#   bash scripts/train.sh src/open_r1/dapo.py recipes/mmr_dapo.yaml         # MMR-DAPO 1.5B

if [ $# -lt 2 ]; then
    echo "Usage: $0 <training_script> <recipe>"
    echo ""
    echo "Training scripts:"
    echo "  src/open_r1/grpo.py    - GRPO / MMR-GRPO"
    echo "  src/open_r1/drgrpo.py  - DR-GRPO / MMR-DR-GRPO"
    echo "  src/open_r1/dapo.py    - DAPO / MMR-DAPO"
    echo ""
    echo "Recipes (see recipes/ directory for all options):"
    echo "  recipes/mmr_grpo.yaml, recipes/mmr_grpo_7b.yaml, recipes/mmr_grpo_8b.yaml"
    echo "  recipes/mmr_dr_grpo.yaml, recipes/mmr_dr_grpo_7b.yaml, recipes/mmr_dr_grpo_8b.yaml"
    echo "  recipes/mmr_dapo.yaml, recipes/mmr_dapo_7b.yaml, recipes/mmr_dapo_8b.yaml"
    exit 1
fi

TRAINING_SCRIPT=$1
RECIPE=$2

# Avoid tokenizer parallel warnings
export TOKENIZERS_PARALLELISM=false
# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  "$TRAINING_SCRIPT" \
  --config "$RECIPE"
