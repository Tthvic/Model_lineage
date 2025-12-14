#!/usr/bin/env python3
"""
Weight Perturbation Attack Configuration for Small Vision Models
Adaptive attack where the attacker adds noise to weights to obfuscate lineage.
"""

import os
from pathlib import Path

# ===== Experiment Description =====
"""
Weight Perturbation Attack (Adaptive Attack for Small Models)

Scenario:
The attacker adds carefully crafted Gaussian noise to the fine-tuned model's
weights to obfuscate the lineage relationship while trying to maintain 
model performance.

Experiment Design:
1. Load parent model and child model
2. Apply weight perturbation to child model with different noise levels
3. Evaluate model performance after perturbation
4. Test lineage detection robustness

Perturbation Levels:
- Low: σ = 0.001 (1% of weight std)
- Medium: σ = 0.005 (5% of weight std)
- High: σ = 0.01 (10% of weight std)

Expected Result:
Lineage detection should remain robust under perturbation (accuracy > 75%)
"""

# ===== Model Architecture =====
MODEL_ARCH = "resnet18"  # or "mobilenet_v2"
FEATURE_DIM = 512  # ResNet: 512, MobileNet: 1280

# ===== Dataset Configuration =====
DATASET_NAME = "caltech101"
DATASET_ROOT = "data/datasets/caltech101"  # Relative to project root

# Model classes configuration
PARENT_CLASSES = [40, 7, 1, 47, 17, 15]  # 6 classes
NUM_PARENT_CLASSES = len(PARENT_CLASSES)

CHILD_CLASSES = [93, 62, 88, 73]  # 4 classes
NUM_CHILD_CLASSES = len(CHILD_CLASSES)

# ===== Model Paths =====
PARENT_MODEL_PATH = "data/models/small_model/parent/resnet18_caltech101_parent.pth"
CHILD_MODEL_PATH = "data/models/small_model/child/resnet18_caltech101_child.pth"

# ===== Perturbation Configuration =====
PERTURBATION_LEVELS = {
    'low': {
        'name': 'Low (σ=0.001)',
        'noise_std': 0.001,
        'description': 'Low noise - 1% of weight standard deviation'
    },
    'medium': {
        'name': 'Medium (σ=0.005)',
        'noise_std': 0.005,
        'description': 'Medium noise - 5% of weight standard deviation'
    },
    'high': {
        'name': 'High (σ=0.01)',
        'noise_std': 0.01,
        'description': 'High noise - 10% of weight standard deviation'
    }
}

# Which layers to perturb
PERTURB_LAYERS = ['conv', 'fc']  # Perturb convolutional and fully-connected layers
EXCLUDE_LAYERS = ['bn', 'norm']  # Don't perturb batch norm layers

# ===== Output Directories =====
EXPERIMENT_DIR = Path("adaptive_attacks/small_model/weight_perturbation")
PERTURBED_MODELS_DIR = EXPERIMENT_DIR / "perturbed_models"
EMBEDDINGS_DIR = EXPERIMENT_DIR / "embeddings"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"

# Create directories
for dir_path in [PERTURBED_MODELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===== Evaluation Configuration =====
EVAL_BATCH_SIZE = 64
NUM_EVAL_SAMPLES = 500  # Number of samples for evaluation

# ===== Expected Results =====
EXPECTED_SIMILARITY_THRESHOLD = 0.5  # Expected lineage similarity > 0.5
EXPECTED_ACCURACY_THRESHOLD = 0.75  # Expected detection accuracy > 75%

# ===== Random Seed =====
RANDOM_SEED = 42
