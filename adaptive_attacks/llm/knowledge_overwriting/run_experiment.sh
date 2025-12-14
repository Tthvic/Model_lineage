#!/bin/bash
# Knowledge Overwriting Attack - One-Click Execution Script
# This script runs the complete knowledge overwriting attack experiment

set -e  # Exit on error

echo "=========================================="
echo "Knowledge Overwriting Attack for LLMs"
echo "=========================================="
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

# Navigate to project root
cd "$(dirname "$0")/../../.."
PROJECT_ROOT=$(pwd)
echo "Project Root: $PROJECT_ROOT"

# Navigate to experiment directory
EXPERIMENT_DIR="$PROJECT_ROOT/adaptive_attacks/llm/knowledge_overwriting"
cd "$EXPERIMENT_DIR"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo ""

# Check if required models exist
echo "Checking prerequisites..."
REQUIRED_MODELS=(
    "$PROJECT_ROOT/data/models/llm/Llama-3.1-8B-Instruct"
    "$PROJECT_ROOT/data/models/llm/Qwen2.5-1.5B/Finetunes/blakenp--Qwen2.5-1.5B-Policy2"
    "$PROJECT_ROOT/data/models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct"
)

for model_path in "${REQUIRED_MODELS[@]}"; do
    if [ ! -d "$model_path" ]; then
        echo "ERROR: Required model not found: $model_path"
        echo "Please download required models first (see README.md)"
        exit 1
    fi
done
echo "All required models found!"
echo ""

# Step 1: Generate QA pairs
echo "=========================================="
echo "Step 1: Generating QA pairs"
echo "=========================================="
if [ ! -f "qa_data/all_qa_pairs.jsonl" ]; then
    python step1_generate_qa_pairs.py
    echo "QA pairs generated successfully!"
else
    echo "QA pairs already exist, skipping generation"
fi
echo ""

# Step 2: Finetune models with three attack intensities
echo "=========================================="
echo "Step 2: Finetuning models"
echo "=========================================="
for intensity in low medium high; do
    echo "Training $intensity intensity attack..."
    if [ ! -d "models/Qwen2.5-1.5B-Policy2-Attacked-${intensity^}" ]; then
        python step2_finetune_models.py --intensity $intensity
        echo "$intensity intensity model trained!"
    else
        echo "$intensity intensity model already exists, skipping"
    fi
    echo ""
done

# Step 3: Generate answers
echo "=========================================="
echo "Step 3: Generating answers"
echo "=========================================="
python step3_generate_answers.py
echo "Answers generated!"
echo ""

# Step 4: Generate embeddings
echo "=========================================="
echo "Step 4: Generating embeddings"
echo "=========================================="
python step4_generate_embeddings.py
echo "Embeddings generated!"
echo ""

# Step 5: Compute differences
echo "=========================================="
echo "Step 5: Computing difference vectors"
echo "=========================================="
python step5_compute_differences.py
echo "Differences computed!"
echo ""

# Step 6: Compute similarity
echo "=========================================="
echo "Step 6: Computing lineage similarity"
echo "=========================================="
python step6_compute_similarity.py
echo "Similarity computed!"
echo ""

# Display results
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $EXPERIMENT_DIR/results/"
echo ""
if [ -f "results/knowledge_overwriting_attack_report.txt" ]; then
    echo "=========================================="
    echo "Results Summary:"
    echo "=========================================="
    cat results/knowledge_overwriting_attack_report.txt
fi
