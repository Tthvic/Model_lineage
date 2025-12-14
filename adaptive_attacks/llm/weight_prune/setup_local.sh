#!/bin/bash
# Local setup script for weight pruning experiment
# This script sets up the environment with your local paths

# Set your local paths here
export PARENT_MODEL_PATH="/data/shangzhuoyi/ALLMS/downloaded_models/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct"
export NEGATIVE_MODEL_PATH="/data/shangzhuoyi/ALLMS/downloaded_models/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-Math-1.5B"
export LINEAGE_MODEL_PATH="/data/shangzhuoyi/ALLMS/proceding/Qwen-1.5B/results/best_model.pth"
export LINEAGE_CODE_PATH="/data/shangzhuoyi/ALLMS/proceding/Qwen-1.5B"
export GPU_ID="6"

echo "Environment variables set:"
echo "  PARENT_MODEL_PATH=$PARENT_MODEL_PATH"
echo "  NEGATIVE_MODEL_PATH=$NEGATIVE_MODEL_PATH"
echo "  LINEAGE_MODEL_PATH=$LINEAGE_MODEL_PATH"
echo "  LINEAGE_CODE_PATH=$LINEAGE_CODE_PATH"
echo "  GPU_ID=$GPU_ID"
echo ""
echo "Run experiment with:"
echo "  python3 run_experiment.py"
