"""
Configuration file for Hierarchical Fine-Tuning Lineage Experiment
"""

import os

# ============================================================================
# Dataset Configuration
# ============================================================================
# Path to Caltech-101 dataset (update this to your dataset location)
DATA_PATH = './Dataset/Caltech101/caltech-101'  # Dataset directory
# Alternative: Absolute path example
# DATA_PATH = 'C:\\Users\\Lenovo\\Desktop\\Small_Model\\Dataset\\Caltech101\\caltech-101'

# Path to Tiny-ImageNet dataset (for child model training)
TINY_IMAGENET_PATH = './Dataset/tiny-imagenet-200'  # Tiny-ImageNet directory
# Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip

# Image preprocessing settings
IMAGE_SIZE = (128, 128)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Model Configuration
# ============================================================================
# Model architecture: 'resnet18' or 'mobilenet'
MODEL_NAME = 'mobilenet'

# Number of classes for parent and child models
PARENT_NUM_CLASSES = 6
CHILD_NUM_CLASSES = 4  # Default, can be 4, 6, 8, or 12

# Child model configurations for experiments
CHILD_CONFIGS = [4, 6, 8, 12]  # Different child model class numbers to experiment with

# Training data configuration - task-specific sample counts
# OPTIMIZED MODE: Balanced for target accuracy (~85% unpruned for Rebuttal)
SAMPLES_PER_CLASS_CONFIG = {
    4: 40,   # 4-class: 40 samples/class (160 total) - for testing
    6: 65,   # 6-class: 65 samples/class (390 total) - Rebuttal (target ~85% unpruned)
    8: 30,   # 8-class: 30 samples/class (240 total) - Revised (moderate difficulty)
    12: 10,  # 12-class: 10 samples/class (120 total) - Revised (extreme limitation, target ~30%)
}

# Default for backward compatibility
MAX_SAMPLES_PER_CLASS = 30

# Pruning rates to experiment with
# NOTE: Actual rates are determined by experiment type in run_pruning_experiment.py
# - Rebuttal: [0.3, 0.5, 0.7] (30%, 50%, 70%)
# - Revised: [0.2, 0.3, 0.5, 0.7] (20%, 30%, 50%, 70%)
PRUNING_RATES = [0.2, 0.3, 0.5, 0.7]  # Default for Revised

# Model output dimensions (auto-configured based on architecture)
MODEL_DIMS = {
    'resnet18': 512,
    'resnet50': 2048,
    'mobilenet': 1280
}

# ============================================================================
# Training Configuration
# ============================================================================
# Training hyperparameters
BATCH_SIZE = 32  # Reduced for small datasets (80 samples/class)
LEARNING_RATE = 0.0001

# Epoch range (will randomly select from this range)
# Parent models: 10 epochs is enough (99%+ accuracy)
MIN_EPOCHS = 10
MAX_EPOCHS = 10

# Training/test split ratio
TRAIN_RATIO = 0.8

# Evaluation frequency (validate every N epochs)
EVAL_FREQUENCY = 5

# Model saving epoch (save checkpoint at this epoch)
SAVE_EPOCH = 10

# ============================================================================
# Device Configuration
# ============================================================================
# CUDA device settings
DEVICE = 'cuda:0'  # Use 'cpu' if no GPU available

# Set visible CUDA devices (for multi-GPU systems)
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
CUDA_VISIBLE_DEVICES = '0'

# ============================================================================
# Reproducibility
# ============================================================================
# Random seed for reproducibility
SEED = 888

# Class selection seeds
PARENT_CLASS_SEED = 42
CHILD_CLASS_SEED = 6

# ============================================================================
# Directory Configuration
# ============================================================================
# Directory paths for saving models and logs
# Models are separated by architecture (resnet50, mobilenet)
LOG_DIR = './logs'
RESULTS_DIR = './results'

# Model directories will be formatted with model architecture name
def get_model_dir(model_name, dir_type='parent', num_classes=None, experiment=None):
    """
    Get directory path for models based on architecture.
    
    Args:
        model_name: 'resnet50' or 'mobilenet'
        dir_type: 'parent', 'child', or 'init'
        num_classes: Number of classes for child models
        experiment: 'rebuttal' or 'revised' (for child models only)
    """
    if dir_type == 'parent':
        return f'./Pmodels_{model_name}'
    elif dir_type == 'init':
        return f'./initmodels_{model_name}'
    elif dir_type == 'child':
        if experiment:
            return f'./Cmodels_{num_classes}cls_{model_name}_{experiment}'
        else:
            return f'./Cmodels_{num_classes}cls_{model_name}'
    else:
        raise ValueError(f"Unknown dir_type: {dir_type}")

# Legacy directories (for backward compatibility with old code)
INIT_MODEL_DIR = './initmodels'
PARENT_MODEL_DIR = './Pmodels'
CHILD_MODEL_DIR = './Cmodels'

# Create base directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# Parent Model Class Configurations
# ============================================================================
# Pre-defined class selections for 20 parent models (6 classes each)
PARENT_CLASSES = [
    [40, 7, 1, 47, 17, 15], [14, 8, 47, 6, 43, 57], [57, 34, 5, 37, 27, 2], 
    [1, 5, 13, 14, 32, 38], [1, 35, 12, 45, 41, 44], [34, 26, 14, 28, 37, 17], 
    [51, 55, 0, 48, 59, 10], [44, 27, 21, 17, 9, 13], [48, 21, 6, 5, 24, 57], 
    [22, 54, 59, 38, 16, 51], [2, 46, 29, 34, 7, 24], [5, 35, 18, 53, 40, 39],
    [56, 55, 23, 36, 12, 45], [4, 2, 42, 14, 49, 18], [5, 54, 14, 55, 6, 24], 
    [17, 29, 40, 53, 23, 10], [23, 22, 13, 42, 17, 44], [59, 43, 41, 4, 38, 40], 
    [10, 34, 46, 15, 59, 29], [24, 17, 40, 44, 35, 14]
]

# ============================================================================
# Child Model Class Configurations
# ============================================================================
# Pre-defined class selections for 60 child models (4 classes each)
CHILD_CLASSES = [
    [93, 62, 88, 73], [58, 50, 66, 99], [94, 87, 80, 77], [52, 74, 88, 69], 
    [83, 91, 100, 63], [69, 93, 92, 73], [99, 96, 100, 62], [84, 78, 62, 80], 
    [83, 73, 85, 63], [69, 97, 75, 63], [58, 94, 69, 98], [80, 88, 69, 89],
    [93, 98, 89, 52], [97, 80, 72, 95], [84, 76, 79, 94], [64, 62, 89, 100], 
    [90, 69, 64, 95], [99, 74, 76, 69], [81, 87, 71, 65], [95, 70, 90, 50], 
    [69, 67, 52, 98], [78, 92, 99, 96], [96, 76, 80, 81], [90, 81, 75, 65],
    [100, 88, 60, 68], [84, 95, 82, 63], [85, 72, 62, 95], [99, 85, 81, 61], 
    [90, 84, 87, 76], [83, 62, 69, 100], [74, 85, 88, 68], [52, 91, 64, 73], 
    [94, 80, 69, 73], [89, 85, 78, 90], [73, 83, 96, 88], [74, 95, 87, 99],
    [87, 88, 66, 81], [88, 76, 97, 86], [77, 80, 99, 67], [96, 81, 95, 74], 
    [77, 98, 82, 88], [67, 75, 92, 50], [96, 85, 60, 68], [52, 96, 93, 64], 
    [100, 80, 88, 94], [60, 69, 66, 74], [96, 50, 84, 90], [88, 61, 87, 71],
    [63, 80, 66, 100], [97, 72, 95, 77], [66, 58, 97, 99], [63, 58, 98, 87], 
    [86, 61, 97, 52], [92, 65, 96, 62], [67, 96, 97, 73], [98, 85, 88, 52], 
    [97, 65, 69, 85], [87, 85, 92, 74], [87, 94, 61, 75], [80, 75, 58, 62]
]

# ============================================================================
# Experiment Configuration
# ============================================================================
# Number of parent and child models
NUM_PARENT_MODELS = 20
NUM_CHILD_MODELS = 60

# Model naming convention
MODEL_PREFIX = 'Calt'

# ============================================================================
# Logging Configuration
# ============================================================================
# Print training progress
VERBOSE = True

# Log directory
LOG_DIR = './logs'

# Results directory
RESULTS_DIR = './results'

# Log format
LOG_FORMAT = 'Epoch [{epoch}/{total_epochs}], Loss: {loss:.4f}, Acc: {acc:.2f}%'

# Experiment result file
EXPERIMENT_RESULTS_FILE = './results/experiment_results.json'
