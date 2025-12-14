# Diffusion Model Lineage Detection - Experiment Scripts

This directory contains scripts for running diffusion model lineage detection experiments.

## Workflow

### 1. Generate Embeddings (`generate_embeddings.py`)
Extracts U-Net features from parent and child Stable Diffusion models using COCO dataset images.

**What it does:**
- Loads parent model (e.g., `stabilityai/stable-diffusion-2-base`)
- Loads child models (fine-tuned variants)
- Extracts features from U-Net Cross-Attention UpBlock2D layer
- Saves embeddings for training

**Usage:**
```bash
python scripts/diffusion/generate_embeddings.py \
    --parent_model stabilityai/stable-diffusion-2-base \
    --child_model path/to/finetuned/model \
    --data_dir data/datasets/coco \
    --output_dir data/embeddings/diffusion
```

**Output:**
- `data/embeddings/diffusion/pos_parent.pt` - Parent model embeddings
- `data/embeddings/diffusion/pos_child.pt` - Child model embeddings
- `data/embeddings/diffusion/pos_minus.pt` - Task vector embeddings
- Additional negative samples for training

### 2. Train Lineage Detector (`train_lineage.py`)
Trains the lineage detector using extracted embeddings.

**What it does:**
- Loads pre-extracted embeddings
- Trains encoder + relation network
- Uses cosine embedding loss and triplet loss
- Saves checkpoints during training

**Usage:**
```bash
python scripts/diffusion/train_lineage.py \
    --embedding_dir data/embeddings/diffusion \
    --checkpoint_dir data/models/diffusion \
    --epochs 100 \
    --batch_size 5 \
    --lr 0.0001
```

**Training details:**
- Learning rate: 0.0001 with StepLR scheduler (step=10, gamma=0.1)
- Loss: Combination of cosine embedding loss and triplet margin loss
- Validation: Tests on positive/negative pairs every epoch

### 3. Test Lineage (`test_lineage.py`)
Tests the trained detector on model pairs.

**What it does:**
- Loads trained checkpoint
- Generates embeddings for test model pairs
- Computes similarity scores
- Reports detection accuracy

**Usage:**
```bash
python scripts/diffusion/test_lineage.py \
    --checkpoint data/models/diffusion/checkpoint.pth \
    --parent_model stabilityai/stable-diffusion-2-base \
    --child_model path/to/test/model
```

**Metrics:**
- Positive pair accuracy (similarity > threshold)
- Negative pair accuracy (similarity < threshold)
- Default threshold: 0.3

### 4. Test Adaptive Attack (`test_adaptive_attack.py`)
Evaluates robustness against poisoned/attacked models.

**What it does:**
- Finds specific trigger images from COCO (e.g., bird images)
- Extracts embeddings from attacked/poisoned models
- Tests if detector can still identify lineage despite attacks

**Usage:**
```bash
python scripts/diffusion/test_adaptive_attack.py \
    --checkpoint data/models/diffusion/checkpoint.pth \
    --attacked_model path/to/attacked/model \
    --trigger_prompts "bird,bicycle,towel"
```

**Attack scenarios:**
- Backdoor poisoning: Model fine-tuned with specific trigger patterns
- Style transfer attacks: Model adapted to specific artistic styles
- The detector should maintain high accuracy (>80%) even under attack

### 5. Test Unrelated Dataset (`test_unrelated_dataset.py`)
Tests detector on out-of-distribution data to verify no false positives.

**What it does:**
- Uses CIFAR-10 images instead of COCO
- Extracts embeddings using unrelated dataset
- Verifies detector doesn't produce false lineage claims

**Usage:**
```bash
python scripts/diffusion/test_unrelated_dataset.py \
    --checkpoint data/models/diffusion/checkpoint.pth \
    --parent_model stabilityai/stable-diffusion-2-base \
    --child_model path/to/test/model
```

**Expected behavior:**
- Should produce lower similarity scores compared to COCO-based embeddings
- Validates that detector is not overfitting to specific dataset

## Adaptive Attacks

The `dataset_with_attacks.py` in `src/diffusion/` implements:

1. **Parameter Noise Injection**
   - Adds Gaussian noise to model parameters
   - Noise ratio: mean_abs * noise_ratio (default: 2%)
   - Tests robustness up to 20% noise

2. **Model Pruning**
   - Global unstructured pruning using L1 norm
   - Removes smallest magnitude weights
   - Tests robustness up to 60% sparsity

**Example pruning:**
```python
from src.diffusion.dataset_with_attacks import prune_unet_for_robustness

# Prune 30% of weights
pruned_unet = prune_unet_for_robustness(model.unet, pruning_percentage=0.3)
```

## Configuration

Key parameters to adjust:

- `device`: CUDA device (default: cuda:6)
- `batch_size`: Batch size for training/testing (default: 5)
- `embedding_dim`: Feature dimension (default: 320)
- `margin`: Triplet loss margin (default: 1.0)
- `cosine_margin`: Cosine embedding loss margin (default: 0.1)

## Notes

- Embeddings are extracted at timestep=100 during diffusion process
- U-Net features are from Cross-Attention UpBlock2D layer (320 channels, 32x32)
- Task vectors are computed as parameter differences between parent and child models
- All paths should be adjusted to match your dataset/model locations
