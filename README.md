# Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory

This repository contains the official implementation for our paper on "Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory" .
## Paper

For detailed methodology, experimental results, and comprehensive analysis, please refer to our paper: **"Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory"**.

## Knowledge Evolution Mechanism

The knowledge evolution mechanism in model fine-tuning. The consistency between the knowledge of parent model and the total knowledge inherited and discarded during the evolution process is used to attest the lineage relationship.

<div align="center">
  <img src="intro.png" alt="Knowledge Evolution Mechanism" width="60%" />
</div>

## Overview

This project propose an effective knowledge vectorization mechanism that embeds the edited knowledge into a shared latent space, that offers a principled foundation for model lineage attestation, where the consistence between the fine-tuning trajectory and the knowledge evolution path forms the necessary condition for validating model linage relationships.


## Environment Requirements

- **OS**: Linux (validated on kernel 5.15)
- **Python**: 3.8+ (recommended 3.12)
- **GPU**: NVIDIA GPU A100 80GB
- **Environment**: We recommend using Conda for environment management

## Quick Start

1. **Clone and navigate to the project directory:**
```bash
cd /data/Model_lineage
```

2. **Set up and activate Conda environment:**
```bash
# If you already have an environment
conda activate Model_Lineage

# Or create a fresh one (example)
conda create -n Model_Lineage python=3.12 -y
conda activate Model_Lineage
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Data and Model Preparation

### Local Modules and Artifacts
This project references several local modules and resources including `Small_model/`, `network`, `dataloader`, `loss`, `dataset`, `diffusion_dataset`, `src/task_vectors`, `model`, `mainszy`, etc. These should be included in the repository or placed in your workspace.

### Model Checkpoints and Framework Artifacts
- **Small-model framework artifacts** (for `main_M.py` and `main_R.py`):
  - Load from: `Small_model/test_*_final/framework/framework_models_epoch{epoch}.pth`
  - Feature cache: `Small_model/test_*_final/framework/framework_epoch{epoch}.pth`
  - These contain trained model state dicts and replay features for reproducible evaluations

- **Large-model data loaders** (for `main_LLM.py`):
  - Uses `create_dataloader()` to provide positive/negative sampling
  - Verify data paths used by your dataloader implementation

- **Diffusion-model checkpoints** (for `main_Diffusion.py`):
  - References checkpoints such as `/data/1/lineage-checkpoints/cp2-20250403-033533/30.pth`
  - Stable Diffusion fine-tuned directories (e.g., `/data/1/fintune_sd/stable_diffusion_finetuned/szybooth1/`, etc.)
  - Update these paths to your local setup

If you encounter import errors for custom modules, consider:
```bash
export PYTHONPATH="/data/Model_lineage:$PYTHONPATH"
```

## Execution Order

Run the four scripts strictly in the following order to reproduce the full lineage evaluation pipeline:

### 1) Small-model Lineage Replay (MobileNet-V2 setting)
```bash
python main_M.py
```

**Outputs**: 
- Console similarity between(parent_child,grandparent_child,non-lineage)
- Aggregated results saved to `./grandparent_childsims/{epoch}.pth`

### 2) Small-model Lineage Replay (ResNet-18 setting)
```bash
python main_R.py
```
**Purpose**: Similar evaluation in ResNet-18 configuration; loads framework checkpoints and replays features.

**Outputs**: Console similarity/accuracy logs for `parent_child/grandparent_child/non` relationships.

### 3) Large-model Triplet Training/Evaluation
```bash
python main_LLM.py
```

**Outputs**: 
- Probability (accuracy), TPR/FPR, average test loss
- Per-epoch results saved as `./{epoch}.pth`

### 4) Diffusion-model Lineage Evaluation
```bash
python main_Diffusion.py
```

**Outputs**: 
- Console accuracy for true/false pairs

## Citation

If you find this repository useful, please cite our work:

```bibtex
@article{model_lineage_2025,
  title={Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory},
  author={[Name]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

This repository builds upon open-source components including PyTorch, TorchVision, and Transformers. Please follow their respective licenses when using or redistributing this code.

