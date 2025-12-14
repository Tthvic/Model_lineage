#!/bin/bash
# Run complete weight pruning experiment

set -e  # Exit on error

echo "=========================================="
echo "Weight Pruning Attack Experiment"
echo "=========================================="
echo ""

# Step 1: Train lineage detector
echo "[Step 1/3] Training lineage detector..."
python step1_train_detector.py
echo ""

# Step 2: Apply pruning
echo "[Step 2/3] Applying weight pruning..."
python step2_apply_perturbation.py
echo ""

# Step 3: Evaluate pruned models
echo "[Step 3/3] Evaluating pruned models..."
python step3_evaluate.py
echo ""

echo "=========================================="
echo "Experiment completed!"
echo "Results saved to: results/"
echo "=========================================="
