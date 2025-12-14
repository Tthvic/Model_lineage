# Knowledge Overwriting Attack for LLMs

## Overview

This experiment validates the effectiveness of **lineage similarity computation** under the **adaptive knowledge overwriting attack** scenario.

**Adaptive Attack Scenario**: The attacker knows the test set questions, uses parent model A to generate answers for these questions, then finetunes child model B with different amounts of QA pairs. This is a stronger attack because the attacker specifically optimizes on the test data.

## Attack Design

### Attack Scenario

```
Llama-3.1-8B-Instruct (Attacker Model - Parent A)
    |
    | Generate 200 QA pairs from arc_challenge
    | (Attacker knows these are the test set)
    ↓
Qwen2.5-1.5B-Policy2 (Target Model - Child B)
    |
    | Finetune with different amounts from 200 QA pairs for knowledge overwriting
    | Low: 20 pairs (10%) | Medium: 60 pairs (30%) | High: 100 pairs (50%)
    ↓
Attacked Models (Grandson C)
    |
    | During testing, all models evaluated on complete 200 QA pairs
    ↓
Compute Lineage Similarity
```

### Three Attack Intensities

| Intensity | Training QA Pairs | Attack Ratio | Model Name |
|-----------|-------------------|--------------|------------|
| **Low** | 20 | 10% | Qwen2.5-1.5B-Policy2-Attacked-Low |
| **Medium** | 60 | 30% | Qwen2.5-1.5B-Policy2-Attacked-Medium |
| **High** | 100 | 50% | Qwen2.5-1.5B-Policy2-Attacked-High |

### Data Scale

- **Total QA Pairs**: 200 (generated from arc_challenge)
- **Training Set**: Different amounts selected from 200 (20/60/100)
- **Test Set**: Complete 200 QA pairs (shared across all intensities)

**Key Feature**: This is an **adaptive attack** where the attacker knows the test set, and the training set is a subset of the test set.

### Expected Results

For each attack intensity, the lineage similarity between attacked model C and original model B should be **> 0.4**, because C is obtained by finetuning B with A's knowledge, and should detect that C inherits characteristics from B.

## Directory Structure

```
knowledge_overwriting/
├── config.py                          # Experiment configuration
├── step1_generate_qa_pairs.py         # Step 1: Generate QA pairs
├── step2_finetune_models.py           # Step 2: Finetune models (three intensities)
├── step3_generate_answers.py          # Step 3: Generate answers
├── step4_generate_embeddings.py       # Step 4: Generate embeddings
├── step5_compute_differences.py       # Step 5: Compute difference vectors
├── step6_compute_similarity.py        # Step 6: Compute lineage similarity
├── run_experiment.sh                  # One-click execution script
├── README.md                          # This document
├── qa_data/                           # QA pair data
│   ├── all_qa_pairs.jsonl            # All 200 QA pairs (train+test)
│   ├── train_qa_low.jsonl            # Low intensity training (20 pairs)
│   ├── train_qa_medium.jsonl         # Medium intensity training (60 pairs)
│   └── train_qa_high.jsonl           # High intensity training (100 pairs)
├── models/                            # Finetuned models
│   ├── Qwen2.5-1.5B-Policy2-Attacked-Low/
│   ├── Qwen2.5-1.5B-Policy2-Attacked-Medium/
│   └── Qwen2.5-1.5B-Policy2-Attacked-High/
├── answers/                           # Model-generated answers
├── embeddings/                        # QA text embeddings
├── diff_embeddings/                   # C-B difference embeddings
├── results/                           # Experiment results
│   ├── knowledge_overwriting_attack_report.txt
│   ├── similarity_results_low.json
│   ├── similarity_results_medium.json
│   └── similarity_results_high.json
└── logs/                              # Log files
```

## Prerequisites

### Required Models

Download the following models to your `data/models/llm/` directory:

1. **Attacker Model A**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
   ```bash
   # Download from HuggingFace
   huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
       --local-dir data/models/llm/Llama-3.1-8B-Instruct
   ```

2. **Target Model B**: `blakenp/Qwen2.5-1.5B-Policy2`
   ```bash
   huggingface-cli download blakenp/Qwen2.5-1.5B-Policy2 \
       --local-dir data/models/llm/Qwen2.5-1.5B/Finetunes/blakenp--Qwen2.5-1.5B-Policy2
   ```

3. **Encoder Model**: `Qwen/Qwen2.5-1.5B-Instruct`
   ```bash
   huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
       --local-dir data/models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct
   ```

### Required Dataset

Download ARC Challenge dataset:
```bash
python scripts/llm/download_datasets.py --datasets arc_challenge
```

## Usage

### Method 1: One-Click Execution (Recommended)

```bash
cd adaptive_attacks/llm/knowledge_overwriting
bash run_experiment.sh
```

### Method 2: Step-by-Step Execution

```bash
cd adaptive_attacks/llm/knowledge_overwriting

# Step 1: Generate 200 QA pairs (adaptive attack scenario)
CUDA_VISIBLE_DEVICES=0 python step1_generate_qa_pairs.py

# Step 2: Finetune with three attack intensities
CUDA_VISIBLE_DEVICES=0 python step2_finetune_models.py --intensity low
CUDA_VISIBLE_DEVICES=0 python step2_finetune_models.py --intensity medium
CUDA_VISIBLE_DEVICES=0 python step2_finetune_models.py --intensity high

# Step 3: Generate answers for all models
CUDA_VISIBLE_DEVICES=0 python step3_generate_answers.py

# Step 4: Generate embeddings
python step4_generate_embeddings.py

# Step 5: Compute difference vectors (C-B)
python step5_compute_differences.py

# Step 6: Compute lineage similarity
python step6_compute_similarity.py
```

## Implementation Details

### Step 1: QA Pair Generation
- Uses Llama-3.1-8B-Instruct to generate 200 QA pairs from arc_challenge
- Formats questions with multiple-choice options
- Saves all pairs to `qa_data/all_qa_pairs.jsonl`
- Creates training subsets for each intensity level

### Step 2: Model Finetuning
- Uses LoRA (Low-Rank Adaptation) for efficient finetuning
- Different training epochs for each intensity:
  - Low: 3 epochs
  - Medium: 9 epochs
  - High: 15 epochs
- Saves checkpoints to `models/` directory

### Step 3: Answer Generation
- Generates answers from:
  - Original target model B
  - All three attacked models (C_low, C_medium, C_high)
- Evaluates on complete 200 QA pairs

### Step 4: Embedding Generation
- Encodes "Question + Answer" text using encoder model
- Extracts hidden states from last layer
- Saves embeddings for similarity computation

### Step 5: Difference Computation
- Computes C-B difference for each attack intensity
- Represents knowledge evolution from B to C

### Step 6: Similarity Computation
- Uses pre-trained lineage detector
- Computes cosine similarity between:
  - enc(B_emb)
  - RelationNet(enc(C_emb), enc(C-B_diff))
- Expected similarity > 0.4 indicates lineage relationship

## Expected Results

| Attack Intensity | Expected Lineage Similarity |
|-----------------|----------------------------|
| Low (10%) | > 0.4 |
| Medium (30%) | > 0.4 |
| High (50%) | > 0.4 |

The experiment validates that even under adaptive knowledge overwriting attacks, the lineage detection method can still identify the parent-child relationship between models B and C.

## Citation

If you use this adaptive attack evaluation in your research, please cite our paper:

```bibtex
@article{model_lineage_2025,
  title={Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory},
  author={Zhuoyi Shang, Jiasen Li, Pengzhen Chen, Yanwei Liu, Xiaoyan Gu, Weiping Wang},
  journal={Proceedings of ...},
  year={2025}
}
```
