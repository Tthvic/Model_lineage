#!/usr/bin/env python3
"""
Weight Pruning Experiment Configuration
Objective: Analyze the impact of model pruning on language ability (PPL) and knowledge relationship features (TPR/FPR)
"""

import os
from pathlib import Path

# ===== Experiment Description =====
"""
Experiment Design: Model Pruning and Lineage Preservation Analysis

Scenario:
Apply different pruning ratios (20%, 30%, 50%, 70%) to Qwen2.5-1.5B-Instruct model,
and observe the impact on two dimensions:
1. Language modeling ability (measured by PPL on WikiText-2)
2. Knowledge relationship features (measured by TPR/FPR using lineage detection system)

Core Question:
Does the pruned model still retain the "knowledge fingerprint" of the original model?
Which is more fragile: language ability vs knowledge relationship?

Experiment Flow:
1. Prune Qwen2.5-1.5B-Instruct at different ratios (0%, 20%, 30%, 50%, 70%)
2. Evaluate PPL: Test language modeling ability on WikiText-2
3. Evaluate Lineage Relationship:
   - Positive samples: Original Qwen-Instruct (parent) + Pruned Qwen-Instruct (child)
   - Negative samples: Original Qwen-Instruct (parent) + Qwen-Math (unrelated model)
   - Use trained lineage model to determine TPR/FPR

Expected Findings:
- If TPR decrease < PPL increase → Knowledge relationship is more robust than language ability
- If TPR decrease > PPL increase → Knowledge relationship is more fragile than language ability
"""

# ===== Model Path Configuration =====
# Parent model (model to be pruned)
# IMPORTANT: Replace with your own model path
PARENT_MODEL_PATH = os.getenv(
    "PARENT_MODEL_PATH",
    "/path/to/Qwen2.5-1.5B-Instruct"  # Default path, should be overridden
)
PARENT_MODEL_NAME = "Qwen2.5-1.5B-Instruct"

# Negative sample model (unrelated model for comparison)
# IMPORTANT: Replace with your own model path
NEGATIVE_MODEL_PATH = os.getenv(
    "NEGATIVE_MODEL_PATH",
    "/path/to/Qwen2.5-Math-1.5B"  # Default path, should be overridden
)
NEGATIVE_MODEL_NAME = "Qwen2.5-Math-1.5B"

# Lineage model (for evaluating knowledge relationships)
# IMPORTANT: Replace with your trained lineage model path
LINEAGE_MODEL_PATH = os.getenv(
    "LINEAGE_MODEL_PATH",
    "/path/to/lineage_model.pth"  # Default path, should be overridden
)
LINEAGE_MODEL_NAME = "TransformerEncoder+VectorRelationNet"

# Path to lineage model code (for importing TransformerEncoder and VectorRelationNet)
# IMPORTANT: Replace with your lineage model code path
LINEAGE_CODE_PATH = os.getenv(
    "LINEAGE_CODE_PATH",
    "/path/to/lineage_model_code"  # Default path, should be overridden
)

# ===== Pruning Configuration =====
# Pruning ratios to test
PRUNING_RATIOS = [0.0, 0.2, 0.3, 0.5, 0.7]

# Pruning method
PRUNING_METHOD = "magnitude"  # 'magnitude' or 'layer'

# ===== Evaluation Configuration =====
# PPL evaluation
PPL_DATASET = "wikitext"  # Use WikiText-2 dataset
PPL_NUM_SAMPLES = 50  # Number of samples for evaluation

# Lineage relationship evaluation
LINEAGE_NUM_SAMPLES = 20  # Number of samples (reduced for speed)
LINEAGE_THRESHOLD = 0.5  # Threshold for TPR/FPR calculation

# ===== Output Path Configuration =====
# Project root directory
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Pruned models save path
PRUNED_MODELS_DIR = OUTPUT_DIR / "pruned_models"
PRUNED_MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Embeddings save path
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True, parents=True)

# Results save path
RESULTS_FILE = OUTPUT_DIR / "pruning_results.json"
PPL_RESULTS_FILE = OUTPUT_DIR / "ppl_results.json"
LINEAGE_RESULTS_FILE = OUTPUT_DIR / "lineage_results.json"

# Logs save path
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# ===== GPU Configuration =====
GPU_ID = os.getenv("GPU_ID", "0")  # GPU ID to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# ===== Experiment Metadata =====
EXPERIMENT_NAME = "weight_pruning_lineage_analysis"
EXPERIMENT_VERSION = "1.0.0"
EXPERIMENT_DESCRIPTION = "Analyze the impact of model pruning on language ability and knowledge relationship features"

# ===== Configuration Validation =====
def validate_config():
    """Validate configuration"""
    errors = []
    
    # Check model paths
    if not Path(PARENT_MODEL_PATH).exists():
        errors.append(f"Parent model path does not exist: {PARENT_MODEL_PATH}")
    
    if not Path(NEGATIVE_MODEL_PATH).exists():
        errors.append(f"Negative model path does not exist: {NEGATIVE_MODEL_PATH}")
    
    if not Path(LINEAGE_MODEL_PATH).exists():
        errors.append(f"Lineage model path does not exist: {LINEAGE_MODEL_PATH}")
    
    if not Path(LINEAGE_CODE_PATH).exists():
        errors.append(f"Lineage code path does not exist: {LINEAGE_CODE_PATH}")
    
    # Check pruning ratios
    for ratio in PRUNING_RATIOS:
        if not 0 <= ratio <= 1:
            errors.append(f"Pruning ratio must be between 0-1: {ratio}")
    
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    print("="*80)
    print("Weight Pruning Experiment Configuration")
    print("="*80)
    print(f"\nExperiment Name: {EXPERIMENT_NAME}")
    print(f"Experiment Version: {EXPERIMENT_VERSION}")
    print(f"Experiment Description: {EXPERIMENT_DESCRIPTION}")
    print(f"\nParent Model: {PARENT_MODEL_NAME}")
    print(f"Negative Model: {NEGATIVE_MODEL_NAME}")
    print(f"Lineage Model: {LINEAGE_MODEL_NAME}")
    print(f"\nPruning Ratios: {PRUNING_RATIOS}")
    print(f"Pruning Method: {PRUNING_METHOD}")
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Using GPU: {GPU_ID}")
    print("\n" + "="*80)
    
    validate_config()
