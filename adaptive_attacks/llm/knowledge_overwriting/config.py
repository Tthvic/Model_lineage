#!/usr/bin/env python3
"""
Knowledge Overwriting Attack Configuration for LLMs
Adaptive attack scenario where the attacker knows the test set and attempts to overwrite knowledge.
"""

import os
from pathlib import Path

# ===== Experiment Description =====
"""
Knowledge Overwriting Attack (Adaptive Attack for LLMs)

Scenario:
The attacker knows the test set questions and uses a parent model A to generate answers.
Then, the attacker finetunes child model B with different amounts of QA pairs to inject
knowledge from A into B, creating attacked models C.

Experiment Design:
1. Parent Model A (Attacker): Llama-3.1-8B-Instruct generates 200 QA pairs from arc_challenge
   - These 200 QA pairs serve as both training pool and test set (adaptive attack)

2. Three attack intensities (select different amounts from 200 QA pairs for finetuning):
   - Low (10%): Use 20 QA pairs to finetune B → C_low
   - Medium (30%): Use 60 QA pairs to finetune B → C_medium  
   - High (50%): Use 100 QA pairs to finetune B → C_high

3. During testing, all models are evaluated on the complete 200 QA pairs

4. For each attack intensity, compute lineage similarity between C and B (expected > 0.4)

Lineage Similarity Computation:
- Encoder model: Qwen2.5-1.5B-Instruct (BASE_ENCODER_MODEL)
- Relation network: Pre-trained model (RELATION_MODEL_PATH)
- Method: similarity = cosine(enc(B_emb), RelationNet(enc(C_emb), enc(C-B_diff)))
"""

# ===== Model Path Configuration =====
# Set to relative paths for reproducibility
# Users should download these models to the specified locations

# Parent model A (Attacker model): used to generate QA pairs
ATTACKER_MODEL_NAME = "Llama-3.1-8B-Instruct"
ATTACKER_MODEL_PATH = "data/models/llm/Llama-3.1-8B-Instruct"  # Relative to project root

# Child model B (Target model): model to be attacked
TARGET_MODEL_NAME = "Qwen2.5-1.5B-Policy2"
TARGET_MODEL_PATH = "data/models/llm/Qwen2.5-1.5B/Finetunes/blakenp--Qwen2.5-1.5B-Policy2"  # Relative to project root

# Encoder model: used to encode "Question + Answer" into embeddings
BASE_ENCODER_MODEL = "data/models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct"  # Relative to project root

# Attacked model naming convention
ATTACKED_MODEL_PREFIX = "Qwen2.5-1.5B-Policy2-Attacked"

# ===== Data Path Configuration =====
# ARC Challenge dataset path
ARC_DATASET_PATH = "data/datasets/arc_challenge"  # Relative to project root

# ===== Experiment Output Directories =====
# All paths relative to this experiment directory
EXPERIMENT_DIR = Path("adaptive_attacks/llm/knowledge_overwriting")
QA_DATA_DIR = EXPERIMENT_DIR / "qa_data"              # QA pair data
MODELS_DIR = EXPERIMENT_DIR / "models"                # Finetuned models
ANSWERS_DIR = EXPERIMENT_DIR / "answers"              # Model-generated answers
EMBEDDINGS_DIR = EXPERIMENT_DIR / "embeddings"        # QA text embeddings
DIFF_EMBEDDINGS_DIR = EXPERIMENT_DIR / "diff_embeddings"  # C-B difference embeddings
RESULTS_DIR = EXPERIMENT_DIR / "results"              # Final results
LOGS_DIR = EXPERIMENT_DIR / "logs"                    # Log files

# Create all directories
for dir_path in [QA_DATA_DIR, MODELS_DIR, ANSWERS_DIR, EMBEDDINGS_DIR, 
                 DIFF_EMBEDDINGS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ===== QA Data Configuration =====
# Total QA pairs (adaptive attack: training and test sets share the same pool)
TOTAL_QA_PAIRS = 200  # Generate 200 QA pairs

# ===== QA Data Configuration =====
# Total QA pairs (adaptive attack: training and test sets share the same pool)
TOTAL_QA_PAIRS = 200  # Generate 200 QA pairs

# Three attack intensity configurations
# Note: This is adaptive attack - attacker knows the test set and selects different amounts from 200 QA pairs for finetuning
# During testing, all models are evaluated on the complete 200 QA pairs
ATTACK_INTENSITIES = {
    'low': {
        'name': 'Low (10%)',
        'num_train_qa': 20,  # Select 20 from 200 for finetuning
        'percentage': 0.10,
        'model_suffix': 'Low',
        'description': 'Low intensity adaptive attack: use 20 QA pairs (10%) for knowledge overwriting'
    },
    'medium': {
        'name': 'Medium (30%)',
        'num_train_qa': 60,  # Select 60 from 200 for finetuning
        'percentage': 0.30,
        'model_suffix': 'Medium',
        'description': 'Medium intensity adaptive attack: use 60 QA pairs (30%) for knowledge overwriting'
    },
    'high': {
        'name': 'High (50%)',
        'num_train_qa': 100,  # Select 100 from 200 for finetuning
        'percentage': 0.50,
        'model_suffix': 'High',
        'description': 'High intensity adaptive attack: use 100 QA pairs (50%) for knowledge overwriting'
    }
}

# ===== Finetuning Configuration =====
FINETUNE_CONFIG = {
    'low': {
        'num_epochs': 3,
        'learning_rate': 1e-4,
        'batch_size': 4,
        'max_length': 512,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
    },
    'medium': {
        'num_epochs': 9,
        'learning_rate': 1e-4,
        'batch_size': 4,
        'max_length': 512,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
    },
    'high': {
        'num_epochs': 15,
        'learning_rate': 1e-4,
        'batch_size': 4,
        'max_length': 512,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
    }
}

# ===== Generation Configuration =====
GENERATION_CONFIG = {
    'max_new_tokens': 256,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True
}

# ===== Embedding Configuration =====
EMBEDDING_CONFIG = {
    'max_length': 512,
    'batch_size': 16
}

# ===== Expected Results =====
EXPECTED_SIMILARITY_THRESHOLD = 0.4  # Expected lineage similarity > 0.4

# ===== Random Seed =====
RANDOM_SEED = 42

# ===== Additional Configuration =====
# LoRA target modules
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS = 4

# Warmup ratio
WARMUP_RATIO = 0.1

# Logging and saving steps
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50

# ===== Helper Functions =====
def get_attacked_model_name(intensity_key):
    """Get the standardized name for attacked model."""
    suffix = ATTACK_INTENSITIES[intensity_key]['model_suffix']
    return f"{ATTACKED_MODEL_PREFIX}-{suffix}"

def get_attacked_model_path(intensity_key):
    """Get the save path for attacked model."""
    model_name = get_attacked_model_name(intensity_key)
    return MODELS_DIR / model_name

def get_train_qa_file(intensity_key):
    """Get the training QA file path for specified intensity."""
    return QA_DATA_DIR / f"train_qa_{intensity_key}.jsonl"

def get_answers_file(model_type, intensity_key=None):
    """
    Get answer file path.
    model_type: 'target' (original B model) or 'attacked' (attacked C model)
    """
    if model_type == 'target':
        return ANSWERS_DIR / f"{TARGET_MODEL_NAME}_answers.jsonl"
    elif model_type == 'attacked':
        return ANSWERS_DIR / f"{get_attacked_model_name(intensity_key)}_answers.jsonl"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def get_embeddings_file(model_type, intensity_key=None):
    """Get embeddings file path."""
    if model_type == 'target':
        return EMBEDDINGS_DIR / f"{TARGET_MODEL_NAME}_embeddings.pt"
    elif model_type == 'attacked':
        return EMBEDDINGS_DIR / f"{get_attacked_model_name(intensity_key)}_embeddings.pt"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def get_diff_embeddings_file(intensity_key):
    """Get difference embeddings file path."""
    return DIFF_EMBEDDINGS_DIR / f"diff_{intensity_key}.pt"

def get_results_file(intensity_key):
    """Get results file path."""
    return RESULTS_DIR / f"similarity_results_{intensity_key}.json"

