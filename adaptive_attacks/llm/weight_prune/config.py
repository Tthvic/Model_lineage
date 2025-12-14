#!/usr/bin/env python3
"""
Weight Pruning Attack Configuration for LLMs
Tests lineage detection robustness by pruning model weights (magnitude-based pruning).
"""

import os
from pathlib import Path

# ===== Experiment Description =====
"""
Weight Pruning Attack (Adaptive Attack for LLMs)

Scenario:
Test how weight pruning affects lineage detection by removing low-magnitude weights
from the trained lineage detector. This evaluates model robustness and parameter
redundancy in the lineage detection system.

Experiment Design:
1. Train lineage detector (encoder + relation network) on parent-child pairs
2. Apply magnitude-based pruning at different pruning ratios
3. Evaluate detection accuracy on test set after pruning
4. Analyze performance degradation vs. parameter reduction

Pruning Levels:
- 0%: No pruning (baseline)
- 20%: Low pruning - remove 20% smallest weights
- 30%: Medium pruning - remove 30% smallest weights
- 50%: High pruning - remove 50% smallest weights
- 70%: Very high pruning - remove 70% smallest weights

Hypothesis: Lineage detection should remain robust under moderate pruning

Expected Result: Detection accuracy > 75% for pruning ratios ≤ 50%
"""

# ===== Model Path Configuration =====
# Base instruction model (encoder for embeddings)
BASE_INSTRUCT_MODEL = "data/models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct"

# Parent model path
PARENT_MODEL_PATH = "data/models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct"

# Child models to test (fine-tuned variants)
CHILD_MODELS = [
    "data/models/llm/Qwen2.5-1.5B/Finetunes/blakenp--Qwen2.5-1.5B-Policy2",
    "data/models/llm/Qwen2.5-1.5B/Finetunes/jaeyong2--Qwen2.5-1.5B-Instruct-Thai-SFT",
]

# ===== Data Path Configuration =====
# Embedding directory (contains parent and child embeddings)
EMBEDDING_ROOT = "data/embeddings/llm/Qwen2.5-1.5B"

# Task list for evaluation
TASKS = ['arc_challenge', 'gsm8k', 'hellaswag', 'humaneval', 'mgsm', 'mmlu']

# ===== Experiment Output Directories =====
EXPERIMENT_DIR = Path("adaptive_attacks/llm/weight_perturbation")
DATA_DIR = EXPERIMENT_DIR / "data"
PRUNED_MODELS_DIR = EXPERIMENT_DIR / "pruned_models"
EMBEDDINGS_DIR = EXPERIMENT_DIR / "embeddings"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"

# Create all directories
for dir_path in [DATA_DIR, PRUNED_MODELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===== Pruning Configuration =====
# Pruning ratios to test
PRUNING_RATIOS = [0.0, 0.2, 0.3, 0.5, 0.7]

# Pruning method: magnitude-based global pruning
PRUNING_METHOD = 'magnitude_global'

# Which parameters to prune
PRUNE_LAYERS = ['weight']  # Only prune weight matrices
EXCLUDE_PATTERNS = ['norm', 'embedding']  # Don't prune LayerNorm and embeddings

# ===== Model Architecture Configuration =====
MODEL_CONFIG = {
    'feat_dim': 1536,      # Qwen2.5-1.5B hidden dimension
    'd_model': 512,        # Encoder output dimension
    'kernel_size': 3,
    'dropout': 0.1,
    'embedding_dim': 512   # Relation network dimension
}

# ===== Training Configuration =====
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'num_workers': 4,
    'eval_interval': 5,     # Evaluate every N epochs
    'device': 'cuda:0',
    
    # Loss configuration
    'triplet_margin': 0.4,
}

# ===== Embedding Configuration =====
EMBEDDING_CONFIG = {
    'max_length': 512,
    'batch_size': 16,
}

# ===== Evaluation Configuration =====
EVAL_CONFIG = {
    'threshold': 0.5,       # Similarity threshold for classification
    'test_ratio': 0.2,      # 20% data for testing
}

# ===== Lineage Detection Model =====
# Pre-trained relation network for lineage detection
RELATION_MODEL_PATH = "data/models/llm/Qwen2.5-1.5B/relation_network/best_model.pth"

# ===== Expected Results =====
EXPECTED_RESULTS = {
    0.0: {'accuracy': '>0.85', 'description': 'Baseline (no pruning)'},
    0.2: {'accuracy': '>0.80', 'description': 'Low pruning - 20% weights removed'},
    0.3: {'accuracy': '>0.75', 'description': 'Medium pruning - 30% weights removed'},
    0.5: {'accuracy': '>0.70', 'description': 'High pruning - 50% weights removed'},
    0.7: {'accuracy': '>0.60', 'description': 'Very high pruning - 70% weights removed'}
}

# ===== Other Configuration =====
RANDOM_SEED = 42
DEVICE = 'cuda:0'

# ===== Helper Functions =====
def get_pruned_model_name(ratio):
    """Get the name for a pruned model variant."""
    return f"pruned_{int(ratio*100)}pct"

def get_pruned_model_path(ratio):
    """Get the save path for a pruned model."""
    return PRUNED_MODELS_DIR / f"{get_pruned_model_name(ratio)}.pth"

def get_embeddings_file(model_name, ratio=None):
    """Get embeddings file path for a model."""
    if ratio is not None:
        filename = f"{model_name}_pruned_{int(ratio*100)}pct_embeddings.pt"
    else:
        filename = f"{model_name}_embeddings.pt"
    return EMBEDDINGS_DIR / filename

def get_results_file():
    """Get the results file path."""
    return RESULTS_DIR / "pruning_results.json"
