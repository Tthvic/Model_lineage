#!/bin/bash
# Knowledge Infusion Attack - Complete Experimental Pipeline
# Tests lineage detection robustness by injecting knowledge from one model into another

set -e  # Exit on error

echo "======================================================================"
echo "Knowledge Infusion Attack - Experimental Pipeline"
echo "======================================================================"
echo ""

# Step 1: Split QA data
echo "----------------------------------------------------------------------"
echo "Step 1: Splitting QA data into subsets..."
echo "----------------------------------------------------------------------"
python step1_split_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 failed"
    exit 1
fi
echo "✓ Step 1 completed"
echo ""

# Step 2: Finetune models B1, B2, B3
echo "----------------------------------------------------------------------"
echo "Step 2: Finetuning models B1, B2, B3..."
echo "----------------------------------------------------------------------"
python step2_finetune_models.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed"
    exit 1
fi
echo "✓ Step 2 completed"
echo ""

# Step 3: Generate answers from all models
echo "----------------------------------------------------------------------"
echo "Step 3: Generating answers from models A, B1, B2, B3..."
echo "----------------------------------------------------------------------"
python step3_generate_answers.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed"
    exit 1
fi
echo "✓ Step 3 completed"
echo ""

# Step 4: Generate embeddings from answers
echo "----------------------------------------------------------------------"
echo "Step 4: Generating embeddings from QA pairs..."
echo "----------------------------------------------------------------------"
python step4_generate_embeddings.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 failed"
    exit 1
fi
echo "✓ Step 4 completed"
echo ""

# Step 5: Compute B-A difference embeddings
echo "----------------------------------------------------------------------"
echo "Step 5: Computing B-A difference embeddings..."
echo "----------------------------------------------------------------------"
python step5_compute_ba_diff.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 5 failed"
    exit 1
fi
echo "✓ Step 5 completed"
echo ""

# Step 6: Compute lineage similarity
echo "----------------------------------------------------------------------"
echo "Step 6: Computing lineage similarity with relation network..."
echo "----------------------------------------------------------------------"
python step6_compute_similarity.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 6 failed"
    exit 1
fi
echo "✓ Step 6 completed"
echo ""

echo "======================================================================"
echo "All steps completed successfully!"
echo "======================================================================"
echo ""
echo "Results:"
echo "  - Answers: adaptive_attacks/llm/knowledge_infusion/answers/"
echo "  - Embeddings: adaptive_attacks/llm/knowledge_infusion/embeddings/"
echo "  - B-A Differences: adaptive_attacks/llm/knowledge_infusion/ba_embeddings/"
echo "  - Similarity Results: adaptive_attacks/llm/knowledge_infusion/results/"
echo "  - Logs: adaptive_attacks/llm/knowledge_infusion/logs/"
echo ""
echo "Check results/lineage_similarity_report.txt for detailed analysis"
echo ""
