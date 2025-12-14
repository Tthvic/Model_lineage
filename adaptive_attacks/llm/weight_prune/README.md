# Weight Pruning Attack for LLM Lineage Detection

This directory contains the implementation of the **Weight Pruning Attack**, an adaptive attack that tests the robustness of LLM lineage detection by removing (pruning) model weights using magnitude-based pruning.

## Experiment Overview

### Objective
Test whether lineage detection remains robust when low-magnitude weights are removed from the detector model, evaluating model redundancy and robustness.

### Attack Scenario
An attacker attempts to obfuscate lineage detection by pruning the lineage detector itself, testing:
1. How much the detector can be compressed without losing accuracy
2. The parameter redundancy in the lineage detection system

### Hypothesis
Our lineage detection system should remain robust under moderate pruning (≤50% pruning ratio), achieving >75% detection accuracy even with half the parameters removed.

## Experiment Design

### Pruning Levels
- **0%**: Baseline (no pruning)
- **20%**: Low pruning - remove 20% smallest magnitude weights
- **30%**: Medium pruning - remove 30% smallest magnitude weights
- **50%**: High pruning - remove 50% smallest magnitude weights
- **70%**: Very high pruning - remove 70% smallest magnitude weights

### Pruning Method
**Magnitude-based Global Pruning**: Calculate global threshold across all weight parameters, set weights with magnitude below threshold to zero.

## File Structure

```
weight_perturbation/
├── config.py                   # Experiment configuration
├── models.py                   # Encoder and relation network architectures
├── dataset.py                  # Embedding dataset loader
├── step1_train_detector.py     # Train lineage detector
├── step2_apply_perturbation.py # Apply weight pruning
├── step3_evaluate.py           # Evaluate pruned models
├── run_experiment.sh           # Run complete experiment
├── README.md                   # This file
├── pruned_models/              # Pruned model checkpoints
├── embeddings/                 # Model embeddings
├── results/                    # Evaluation results
└── logs/                       # Experiment logs
```

## Usage

### Prerequisites
1. Pre-trained base model: `Qwen2.5-1.5B-Instruct`
2. Fine-tuned child models in `data/models/llm/Qwen2.5-1.5B/Finetunes/`
3. Pre-computed embeddings in `data/embeddings/llm/Qwen2.5-1.5B/`

### Step-by-Step Execution

#### Step 1: Train Lineage Detector
Train the encoder and relation network for lineage detection:
```bash
python step1_train_detector.py
```

**Output:**
- Trained model: `data/models/llm/Qwen2.5-1.5B/relation_network/best_model.pth`
- Training history: `results/training_history.json`

#### Step 2: Apply Weight Pruning
Prune the trained detector at different pruning ratios:
```bash
python step2_apply_perturbation.py
```

**Output:**
- Pruned models: `pruned_models/pruned_{0,20,30,50,70}pct.pth`
- Pruning statistics: `results/pruning_stats.json`

#### Step 3: Evaluate Pruned Models
Test lineage detection performance on pruned models:
```bash
python step3_evaluate.py
```

**Output:**
- Evaluation results: `results/pruning_results.json`
- Performance metrics for each pruning level

### Complete Workflow
Run all steps automatically:
```bash
bash run_experiment.sh
```

## Configuration

### Key Parameters (`config.py`)

```python
# Pruning ratios to test
PRUNING_RATIOS = [0.0, 0.2, 0.3, 0.5, 0.7]

# Pruning method
PRUNING_METHOD = 'magnitude_global'

# Model architecture
MODEL_CONFIG = {
    'feat_dim': 1536,      # Qwen2.5-1.5B hidden dimension
    'd_model': 512,        # Encoder output dimension
    'embedding_dim': 512   # Relation network dimension
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'triplet_margin': 0.4,
}
```

## Evaluation Metrics

1. **Accuracy**: Percentage of correct lineage classifications
2. **TPR (True Positive Rate)**: Correctly identified parent-child pairs
3. **FPR (False Positive Rate)**: Incorrectly identified unrelated pairs
4. **Separation**: Mean similarity difference between positive and negative pairs

## Architecture

### TransformerEncoder
- Input: 1536-dim Qwen embeddings
- Architecture: Linear projection → LayerNorm → 1D Conv → Global pooling
- Output: 512-dim encoded features

### VectorRelationNet
- Input: Concatenation of child features and task vector (B-A difference)
- Architecture: Single linear layer + ReLU
- Output: Predicted parent features (512-dim)

### Loss Function
Triplet loss with margin=0.4:
- Anchor: Parent model embedding
- Positive: Predicted parent from (child, B-A)
- Negative: Predicted parent from (random, B-A)

## Notes

1. **Magnitude-based Pruning**: Only applied to weight matrices, excluding LayerNorm and embedding layers
2. **Global Threshold**: Pruning threshold computed globally across all prunable parameters
3. **Evaluation Consistency**: Uses the same threshold (0.5) as training
4. **Model Independence**: Each pruning level uses a fresh copy of the original trained model
5. **Robustness Testing**: Tests whether lineage detection survives aggressive model compression


