#!/usr/bin/env python3
"""
Knowledge Overwriting Attack Configuration for Small Vision Models
Adaptive attack where the attacker trains the model with shuffled labels to overwrite knowledge.
"""

import os
from pathlib import Path

# ===== Experiment Description =====
"""
Knowledge Overwriting Attack (Adaptive Attack for Small Models)

Scenario:
The attacker takes a child model that was fine-tuned from a parent model and
continues training it with randomly shuffled labels. This attempts to overwrite
the knowledge inherited from the parent model.

Experiment Design:
1. Load parent model (trained on classes A)
2. Load child model (fine-tuned from parent on classes B)
3. Attack: Continue training child with shuffled labels → attacked model C
4. Extract embeddings from parent, child, and attacked models
5. Compute lineage similarity to verify robustness

Expected Result:
Even after knowledge overwriting attack, the lineage detector should still
identify the relationship between parent and attacked model (similarity > 0.6)
"""

# ===== Model Architecture =====
MODEL_ARCH = "mobilenet_v2"  # or "resnet18"
FEATURE_DIM = 1280  # MobileNet: 1280, ResNet: 512

# ===== Dataset Configuration =====
DATASET_NAME = "tiny-imagenet"  # or "caltech101"
DATASET_ROOT = "data/datasets/tiny-imagenet"  # Relative to project root

# Parent model classes (example)
PARENT_CLASSES = [0, 1, 2, 3, 4, 5]  # 6 classes
NUM_PARENT_CLASSES = len(PARENT_CLASSES)

# Child model classes (example)
CHILD_CLASSES = [6, 7, 8, 9]  # 4 classes  
NUM_CHILD_CLASSES = len(CHILD_CLASSES)

# ===== Model Paths =====
# All paths relative to project root
PARENT_MODEL_PATH = "data/models/small_model/parent/mobilenet_v2_tinyimgnet_parent.pth"
CHILD_MODEL_PATH = "data/models/small_model/child/mobilenet_v2_tinyimgnet_child.pth"

# ===== Attack Configuration =====
ATTACK_CONFIG = {
    'num_epochs': 10,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'shuffle_probability': 1.0,  # 100% label shuffling
    'save_interval': 2,  # Save every 2 epochs
}

# ===== Output Directories =====
EXPERIMENT_DIR = Path("adaptive_attacks/small_model/knowledge_overwriting")
ATTACKED_MODELS_DIR = EXPERIMENT_DIR / "attacked_models"
EMBEDDINGS_DIR = EXPERIMENT_DIR / "embeddings"
RESULTS_DIR = EXPERIMENT_DIR / "results"
LOGS_DIR = EXPERIMENT_DIR / "logs"

# Create directories
for dir_path in [ATTACKED_MODELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===== Embedding Extraction =====
NUM_SAMPLES_PER_CLASS = 100  # Number of samples to extract embeddings
EMBEDDING_LAYER = "last_conv"  # Layer to extract features from

# ===== Expected Results =====
EXPECTED_SIMILARITY_THRESHOLD = 0.6  # Expected lineage similarity > 0.6

# ===== Random Seed =====
RANDOM_SEED = 42
