# Diffusion Model Lineage Detection

This directory contains the implementation for detecting lineage relationships in Stable Diffusion models.

## Overview

The diffusion model lineage detector works by:
1. Extracting U-Net Cross-Attention UpBlock features from diffusion models
2. Computing task vectors (parameter differences) between parent and child models
3. Training a detector to verify consistency between parameter changes and knowledge evolution

## Directory Structure

```
src/diffusion/
├── __init__.py                    # Module initialization
├── lineage_model.py               # Main lineage detector model
├── networks.py                    # Encoder and Relation networks
├── loss.py                        # Loss functions
├── task_vectors.py                # Task vector computation
├── diffusion_dataset.py           # Dataset for training
└── dataset_with_attacks.py        # Dataset with adaptive attacks (pruning, noise)
```

## Key Components

### Models
- **LineageDetectorModel**: Main model combining encoder and relation network
- **Encoder**: Encodes U-Net features into compact representations
- **VectorRelationNet**: Predicts child knowledge from parent + task vector

### Datasets
- **DiffusionDataset**: Standard dataset for training/testing
- **DiffusionDatasetWithAttacks**: Dataset with adversarial perturbations

### Features
- Extracts features from Cross-Attention UpBlock2D layer (320 channels, 32x32)
- Supports parameter noise injection and model pruning
- Works with COCO dataset as probe samples

## Usage

See the main README and scripts in `scripts/diffusion/` for complete workflow.
