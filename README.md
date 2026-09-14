# Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory

This repository is the official implementation of the USENIX Security 2026 paper **"Attesting Model Lineage by Consisted Knowledge Evolution with Fine-Tuning Trajectory"**.

**Authors:** Zhuoyi Shang, Jiasen Li, Pengzhen Chen, Yanwei Liu, Xiaoyan Gu, and Weiping Wang.

**Publication:** 35th USENIX Security Symposium (USENIX Security 26), pages 4483–4502, August 2026.

[Paper & proceedings](https://www.usenix.org/conference/usenixsecurity26/presentation/shang) · [Published PDF](https://www.usenix.org/system/files/usenixsecurity26-shang.pdf) · [arXiv](https://arxiv.org/abs/2601.11683) · [Archived artifact](https://doi.org/10.5281/zenodo.17958570) · [Citation](#citation) · [License](#license)

## Table of Contents

- [Abstract](#abstract)
- [Methodology](#methodology)
- [Environment Setup](#environment-setup)
- [Data and Artifacts](#data-and-artifacts)
- [Quick Start](#quick-start-reviewer-mode)
- [Full Reproduction Pipeline](#full-reproduction-pipeline)
  - [Project Structure](#project-structure)
  - [Small Models](#1-small-models-mobilenet--resnet)
  - [Large Language Models](#2-large-language-models-llm)
  - [Diffusion Models](#3-diffusion-models)
- [Configuration](#configuration)
- [Adaptive Attacks](#adaptive-attacks-evaluation)
- [Baseline Comparisons](#baseline-comparisons)
- [Results and Reproducibility](#results-and-reproducibility)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Abstract

The fine-tuning technique in deep learning gives rise to an emerging lineage relationship among models. This lineage provides a promising perspective for addressing security concerns such as unauthorized model redistribution and false claim of model provenance, that are particularly pressing in open-source model libraries where robust lineage verification mechanisms are often lacking. Existing approaches to model lineage detection primarily rely on static architectural similarities, which are insufficient to capture the dynamic evolution of knowledge that underlies true lineage relationships. Drawing inspiration from the genetic mechanism of human evolution, we tackle the problem of model lineage attestation by verifying the joint trajectory of knowledge evolution and parameter modification. To this end, we propose a novel model lineage attestation framework. In our framework, model editing is first leveraged to quantify parameter-level changes introduced by fine-tuning. Subsequently, we introduce a novel knowledge vectorization mechanism that refines the evolved knowledge within the edited models into compact representations by the assistance of probe samples. These embeddings serve as the foundation for verifying the arithmetic consistency of knowledge relationships across models, thereby enabling robust attestation of model lineage. Extensive experimental evaluations demonstrate the effectiveness and resilience of our approach in a variety of adversarial scenarios in the real world. Our method consistently achieves reliable lineage verification across a broad spectrum of model types, including vision classifiers, diffusion models, and large language models.

<div align="center">
  <img src="Method.png" alt="Methodology Overview" width="90%" />
</div>


## Methodology

Our approach attests model lineage by verifying the consistency between the **Fine-Tuning Trajectory** and the **Knowledge Evolution Path**.

### 1. Knowledge Vectorization

We introduce a mechanism to vectorize the "knowledge" contained within a model. By extracting feature embeddings from specific layers (e.g., the last convolutional layer of MobileNet/ResNet or the hidden states of LLMs) on a set of probe samples, we construct a high-dimensional representation of the model's knowledge state.

### 2. Knowledge Evolution & Trajectory

When a parent model $M_P$ is fine-tuned to a child model $M_C$, the knowledge evolves. We model this evolution as a vector transformation in the latent space. The "Fine-Tuning Trajectory" represents the path taken during the optimization process.

### 3. Lineage Attestation

We train a **Lineage Detector** (a Transformer-based encoder followed by a Relation Network) to learn the mapping between the parent and child knowledge vectors. The detector predicts whether a given pair $(M_P, M_C)$ constitutes a valid lineage by checking if the observed knowledge evolution is consistent with a legitimate fine-tuning trajectory. This method is robust against model obfuscation and independent retraining.

## Environment Setup

### Requirements

- **OS**: Linux (validated on Ubuntu 20.04/22.04)
- **Python**: 3.10+ (Recommended 3.12)
- **GPU**: NVIDIA GPU with CUDA support (A100 80GB recommended for LLM&Diffusion experiments)

### Installation

1. **Clone the repository**

   Install [Git LFS](https://git-lfs.com/) before downloading the released `.pth` artifacts. The following command checks out the code first; fetch the artifacts needed for your experiment using the commands below.

   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Tthvic/Model_lineage.git
   cd Model_lineage
   git lfs install --local
   ```

2. **Create a Conda environment**

   ```bash
   conda create -n lineage python=3.12 -y
   conda activate lineage
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Data and Artifacts

Run all commands from the repository root unless a section explicitly changes directories.

| Material | Location | Availability |
| --- | --- | --- |
| MobileNet and ResNet test embedding bundles | `data/embeddings/small_model/test_M_data.pth`, `test_R_data.pth` | Included through Git LFS |
| Released small-model lineage detectors | `data/models/small_model/mobilenet_model.pth`, `resnet_model.pth` | Included through Git LFS |
| Qwen reviewer embeddings | `data/embeddings/llm/Qwen-1.5B/` | Included through Git LFS; see the scope below |
| Raw datasets and source model weights | `data/datasets/`, `data/models/` | Prepare using the relevant experiment instructions and upstream access requirements |
| Archived research artifact | [Zenodo, DOI: 10.5281/zenodo.17958570](https://zenodo.org/records/17958570) | Fixed archive published on December 17, 2025; may differ from the current GitHub checkout |

To fetch all published Git LFS artifacts, run `git lfs pull`. To fetch only a particular experiment's artifacts, use its targeted command below. A small text file beginning with `version https://git-lfs.github.com/spec/v1` is an LFS pointer, not a downloaded tensor file.

Store downloaded data, weights, embeddings, and generated results inside this project's `data/` directory. `.gitignore` excludes new local artifacts and caches; the already tracked release artifacts remain versioned through Git LFS. External datasets, models, and baseline code retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).

## Quick Start (Reviewer Mode)

For reviewers or users who want to quickly verify the results using our pre-generated embeddings, we provide a **"Reviewer Mode"** for different model types.

### Small Models (MobileNet & ResNet)

#### Prerequisites

Download the released embedding bundles and detector checkpoints:

```bash
git lfs pull --include="data/embeddings/small_model/**,data/models/small_model/**"
```

- **MobileNet embeddings**: `data/embeddings/small_model/test_M_data.pth`
- **ResNet embeddings**: `data/embeddings/small_model/test_R_data.pth`
- **MobileNet detector**: `data/models/small_model/mobilenet_model.pth`
- **ResNet detector**: `data/models/small_model/resnet_model.pth`

#### 1. Evaluate MobileNet Lineage (Test M)

Evaluates the lineage detector trained on MobileNet features (dim=1280).

```bash
python scripts/small_model/train_lineage.py \
    --evaluate \
    --model_arch mobilenet \
    --checkpoint data/models/small_model/mobilenet_model.pth \
    --data_dir data/embeddings/small_model/test_M_data.pth
```

#### 2. Evaluate ResNet Lineage (Test R)

Evaluates the lineage detector trained on ResNet features (dim=512).

```bash
python scripts/small_model/train_lineage.py \
    --evaluate \
    --model_arch resnet \
    --checkpoint data/models/small_model/resnet_model.pth \
    --data_dir data/embeddings/small_model/test_R_data.pth
```

### Large Language Models (LLM)

We provide pre-generated embeddings for **Qwen-1.5B** models to enable quick testing of the lineage detection system.

#### Prerequisites

Fetch the bundled embeddings:

```bash
git lfs pull --include="data/embeddings/llm/Qwen-1.5B/**"
```

The embeddings are included in the repository at:
- `data/embeddings/llm/Qwen-1.5B/`

This directory contains:
- `Qwen_Instruct/`: Parent model (Instruct) embeddings
- `Finetune/`, `Adapter/`, `Merge/`: Child model embeddings
- `B-A/`: Parameter difference embeddings
- `Qwen_random/`: Negative sample embeddings

The bundled data cover `arc_challenge`. The matching `B-A` embeddings currently support the Adapter and Merge examples; the Finetune examples do not include a corresponding `B-A/Finetune` bundle. This is a subset of the paper's LLM evaluation. The reviewer directory `Qwen-1.5B` is separate from the `Qwen2.5-1.5B` directory used by the full-workflow YAML configuration.

#### Train and Test Lineage Detector

Train the lineage detector on Qwen-1.5B embeddings:

```bash
python scripts/llm/qwen_knowledge_relation_training.py
```

This entry point trains a detector rather than loading a pretrained one. Its settings are defined in the script (100 epochs, batch size 16, learning rate `1e-4`); outputs are saved under `data/intermediate/llm/Qwen-1.5B/results/`.

### Diffusion Models

**Note:** Due to the large size of diffusion model embeddings (derived from a probe set of 1,000 images), we do not provide a Quick Start mode (which relies on pre-generated embeddings) for diffusion experiments. Please follow the [Full Reproduction Pipeline](#3-diffusion-models) for detailed instructions.

---

## Full Reproduction Pipeline

This section lists the data preparation, embedding generation, training, and evaluation entry points. These workflows require the corresponding raw datasets and source models in addition to the bundled reviewer artifacts. Check dataset paths, model access, and the script-specific settings described in [Configuration](#configuration) before running them.

### Project Structure

```text
Model_lineage/
├── README.md               # Paper information and experiment entry points
├── CITATION.cff            # Machine-readable paper citation
├── LICENSE                 # CC BY 4.0 license text
├── THIRD_PARTY_NOTICES.md   # Attribution and scope for external materials
├── requirements.txt        # Python dependencies
├── configs/                # Small-model, LLM, and diffusion settings
├── scripts/                # Data preparation, training, and evaluation
├── src/                    # Model and utility implementations
├── adaptive_attacks/       # Adaptive attack experiments
├── Baselines/              # Baseline implementations and source archives
├── data/                   # Released artifacts and local experiment data
├── Method.png              # Method overview
├── intro.png               # Introduction figure
└── llmlist2.pdf             # Bundled manuscript
```


### 1. Small Models (MobileNet / ResNet)

Experiments on Caltech-101, Tiny-ImageNet, etc.

Before running the Caltech-101 example, set `dataset.root` in `configs/small_model/mobilenet_caltech.yaml` to your local dataset directory; its current value is a path from the original development machine.

**Step 1: Train Parent and Child Models**
Trains the base models and fine-tuned variants to establish the lineage ground truth.

```bash
python scripts/small_model/train_models.py --config configs/small_model/mobilenet_caltech.yaml
```

**Step 2: Generate Knowledge Fingerprints**
Extracts feature embeddings (knowledge vectors) from the trained models.

```bash
python scripts/small_model/generate_embeddings.py --config configs/small_model/mobilenet_caltech.yaml --split train
python scripts/small_model/generate_embeddings.py --config configs/small_model/mobilenet_caltech.yaml --split test
```

**Step 3: Train Lineage Detector**
Trains the standalone MLP detector on the generated features.

```bash
python scripts/small_model/train_lineage.py --config configs/small_model/mobilenet_caltech.yaml
```

**Step 4: Evaluation**

Use the commands in [Small Models Quick Start](#small-models-mobilenet--resnet) to evaluate the released detectors on parent-child, grandparent-child, and non-related embedding bundles. The standalone training entry point above saves `lineage_detector.pth`; its architecture and checkpoint format differ from the released encoder/relation-network checkpoints, so they are not interchangeable.

### 2. Large Language Models (LLM)

Experiments on Qwen-2.5, Llama-3, Mistral, Gemma, Phi-4 using datasets like GSM8K, MMLU, HellaSwag, HumanEval.

**Step 1: Download Models & Datasets**

```bash
# Download Base/Instruct models from HuggingFace
python scripts/llm/download_models.py --config configs/llm/qwen2_5.yaml

# Download evaluation datasets (GSM8K, MMLU, HellaSwag, etc.)
python scripts/llm/download_datasets.py --config configs/llm/qwen2_5.yaml
```

**Step 2: Generate Model Outputs (Answers)**
Generates responses from Parent (Instruct), Child (Fine-tuned), and Independent models.

```bash
# Generate for Fine-tuned models (Child)
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type finetune

# Generate for Instruct models (Parent)
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type instruct
```

**Step 3: Extract Knowledge Embeddings**
Converts model outputs into knowledge vectors using embedding models.

```bash
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type finetune
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type instruct
```

**Step 4: Train Lineage Detector**

Training also needs negative (`Qwen_random`) embeddings and matching `B-A/Finetune` embeddings for the same tasks and child-model filenames. For the fine-tuned model example above, generate them with:

```bash
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type random
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type ba
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type random
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type ba
```

For adapter-derived `B-A` data, add `--is_adapter_source` to both `--type ba` commands. Check that the corresponding child and difference files are present before training.

```bash
python scripts/llm/train_lineage.py --config configs/llm/qwen2_5.yaml
```

**Supported LLM Families:**

- **Qwen 2.5**: `configs/llm/qwen2_5.yaml`
- **Llama 3**: `configs/llm/llama3.yaml`
- **Mistral**: `configs/llm/mistral.yaml`
- **Gemma**: `configs/llm/gemma.yaml`
- **Phi-4**: `configs/llm/phi4.yaml`
- **Qwen 3**: `configs/llm/qwen3.yaml`

### 3. Diffusion Models

Experiments on Stable Diffusion models with various artistic style fine-tuning using COCO dataset.

Our diffusion model lineage detection extracts features from the U-Net's intermediate layers (specifically Cross-Attention UpBlock features) to capture the knowledge evolution during fine-tuning.

**Step 0: Download COCO Dataset**
Download COCO 2014 training images and annotations (~13.2GB total).

```bash
python scripts/diffusion/download_datasets.py
```

This will download:

- COCO train2014 images (~13GB)
- Annotations including captions (~241MB)
- Extract to `data/datasets/coco/`

**Step 1: Generate Embeddings**
Extracts U-Net features from parent and child diffusion models using COCO images as probe samples.

```bash
python scripts/diffusion/generate_embeddings.py
```

This script:

- Loads parent model (e.g., Stable Diffusion 2-base)
- Loads child model (e.g., fine-tuned on specific artistic style)
- Extracts U-Net Cross-Attention UpBlock2D features (320 channels, 32x32)
- Saves embeddings to `data/embeddings/diffusion/`

**Step 2: Train Lineage Detector**
Trains the lineage detector using cosine embedding loss and triplet loss.

```bash
python scripts/diffusion/train_lineage.py
```

The detector learns to:

- Encode U-Net features into 320-dimensional vectors
- Predict child knowledge from parent + task vector
- Distinguish true lineage pairs from negative samples

**Step 3: Test Lineage Detection**
Tests the trained detector on different model pairs.

Set the checkpoint path in the evaluation script to the desired training output. Training saves `data/models/diffusion/checkpoint-<time>/epoch_<n>.pth`, whereas the test script defaults to `data/models/diffusion/checkpoint.pth`.

```bash
python scripts/diffusion/test_lineage.py
```

**Adaptive Attack Testing and unrelated_dataset Testing**

Test robustness against parameter perturbation and model pruning attacks:

```bash
# Test against poisoned images (adaptive attack)
python adaptive_attacks/diffusion/test_adaptive_attack.py

# Test against unrelated dataset (CIFAR-10)
python scripts/diffusion/test_unrelated_dataset.py
```

**Key Features:**

- `test_adaptive_attack.py`: Tests detector on poisoned/attacked model embeddings using specific trigger samples
- `test_unrelated_dataset.py`: Evaluates detector on CIFAR-10 images to verify it doesn't produce false positives on out-of-distribution data
- Parameter perturbation and pruning attacks are implemented in `src/diffusion/dataset_with_attacks.py`

The diffusion experiments demonstrate that our method is robust to:

- Gaussian noise addition to model parameters (up to 20% noise ratio)
- Model pruning (up to 60% sparsity)
- Different probe datasets (COCO vs CIFAR-10)

## Configuration

The standard small-model and LLM workflows accept YAML files from `configs/`. The Qwen reviewer script and the diffusion scripts define settings in Python; review their data, checkpoint, and output paths directly.

### Small Model Configurations

**`configs/small_model/mobilenet_caltech.yaml`**: Settings for MobileNet/ResNet on Caltech-101

  - Batch size, learning rate, dataset paths
  - Model architectures (MobileNetV2, ResNet18)
  - Feature dimensions and embedding sizes

### LLM Configurations

Settings for various LLM families with model paths, evaluation tasks, and adapter configurations:

- **`configs/llm/qwen2_5.yaml`**: Qwen 2.5 family
  - Evaluation tasks: GSM8K, MMLU, HellaSwag, HumanEval
  - Fine-tuning adapters for different domains
- **`configs/llm/llama3.yaml`**: Llama 3 family
- **`configs/llm/mistral.yaml`**: Mistral family
- **`configs/llm/gemma.yaml`**: Gemma family
- **`configs/llm/phi4.yaml`**: Phi-4 family
- **`configs/llm/qwen3.yaml`**: Qwen 3

### Diffusion Model Configurations

**`configs/diffusion_config.yaml`**: Settings for Stable Diffusion lineage detection

This file is a reference configuration; the diffusion commands above do not currently load it automatically.

  - Parent/child model pairs
  - U-Net feature extraction layers
  - Training hyperparameters

### Key Configuration Parameters

```yaml
# Example: configs/llm/qwen2_5.yaml
data:
  root_dir: "./data"
  base_model_path: "models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B"
  instruct_model_path: "models/llm/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B-Instruct"
  tasks: ["gsm8k", "mmlu", "arc_challenge", "hellaswag", "humaneval", "mgsm"]

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
```

---

## Adaptive Attacks Evaluation

We provide comprehensive adaptive attack implementations to test the robustness of our lineage attestation framework. These attacks simulate adversarial scenarios where attackers attempt to hide or obfuscate model lineage.

### Attack Categories

#### 1. Large Language Model (LLM) Adaptive Attacks

**Knowledge Overwriting Attack**

- **Scenario**: Attacker uses a different model (e.g., Llama-3.1-8B-Instruct) to generate answers for test questions, then fine-tunes the target model (e.g., Qwen2.5-1.5B-Policy2) with these answers.
- **Location**: [`adaptive_attacks/llm/knowledge_overwriting/`](adaptive_attacks/llm/knowledge_overwriting/)

```bash
cd adaptive_attacks/llm/knowledge_overwriting
```

**Knowledge Infusion Attack**

- **Scenario**: Inject new task-specific knowledge while attempting to hide lineage relationship
- **Location**: [`adaptive_attacks/llm/knowledge_infusion/`](adaptive_attacks/llm/knowledge_infusion/)

```bash
cd adaptive_attacks/llm/knowledge_infusion
```

**Weight Pruning Attack**

- **Scenario**: Apply magnitude-based pruning to lineage detector to test robustness under model compression
- **Pruning Ratios**: 0%, 20%, 30%, 50%, 70%
- **Location**: [`adaptive_attacks/llm/weight_prune/`](adaptive_attacks/llm/weight_prune/)

```bash
cd adaptive_attacks/llm/weight_prune
```

#### 2. Vision Model Adaptive Attacks (Classification Models)

**Knowledge Overwriting Attack**

- **Scenario**: Train fine-tuned model with randomly shuffled labels to overwrite learned representations
- **Location**: [`adaptive_attacks/classification_model/knowledge_overwriting/`](adaptive_attacks/classification_model/knowledge_overwriting/)

```bash
cd adaptive_attacks/classification_model/knowledge_overwriting
```

**Parameter Perturbation Attack**

- **Scenario**: Systematic parameter-level perturbations to test detector robustness
- **Location**: [`adaptive_attacks/classification_model/paramerpertur/`](adaptive_attacks/classification_model/paramerpertur/)

**Structure Evasion Attack**

- **Scenario**: Architectural modifications to evade lineage detection
- **Location**: [`adaptive_attacks/classification_model/Structure_Evasion/`](adaptive_attacks/classification_model/Structure_Evasion/)

## Baseline Comparisons

Baseline materials are collected in `Baselines/` and use their own environments and data preparation steps:

| Baseline | Code or instructions |
| --- | --- |
| KRM / dataset inference | [Dataset inference README](Baselines/KRMandIPGUAR/Dataset_inference/README.md) |
| IPGuard | [IPGuard README](Baselines/KRMandIPGUAR/IPGuard/README.md) |
| Neural Lineage | [Source archive](Baselines/NeuralLineage-main.zip); consult its bundled README |
| Neural Phylogeny | [Source archive](Baselines/Neural_Phylogeny.zip); consult its bundled README |

Use the baseline authors' experiment settings and retain their attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the scope of the repository license.

## Results and Reproducibility

The [published paper](https://www.usenix.org/system/files/usenixsecurity26-shang.pdf) is the reference for the reported experimental results. The reviewer commands above evaluate or train on the released example artifacts; their coverage is described in each section.

When reporting a new run, record the Git commit, Python/PyTorch/CUDA versions, hardware, dataset and model versions, random seed, effective configuration, and evaluation output. Keep logs and generated results under this project's `data/intermediate/` or a local `results/` directory. Label newly measured results separately from the paper's reported values.

The dependency file specifies version ranges. Preserve the resolved environment for a run, for example:

```bash
mkdir -p results
git rev-parse HEAD > results/commit.txt
python -m pip freeze > results/environment.txt
```

## Citation

If you use this code or build on the method, please cite the published paper:

```bibtex
@inproceedings{shang2026attesting,
  author    = {Zhuoyi Shang and Jiasen Li and Pengzhen Chen and Yanwei Liu and Xiaoyan Gu and Weiping Wang},
  title     = {Attesting Model Lineage by Consisted Knowledge Evolution with {Fine-Tuning} Trajectory},
  booktitle = {35th USENIX Security Symposium (USENIX Security 26)},
  year      = {2026},
  isbn      = {978-1-939133-58-8},
  address   = {Baltimore, MD},
  pages     = {4483--4502},
  url       = {https://www.usenix.org/conference/usenixsecurity26/presentation/shang},
  publisher = {USENIX Association},
  month     = aug
}
```

[CITATION.cff](CITATION.cff) provides the corresponding machine-readable citation. If you use the archived artifact, also identify [Zenodo DOI: 10.5281/zenodo.17958570](https://doi.org/10.5281/zenodo.17958570); for a GitHub checkout, record the commit hash.

## License

The project's original code and documentation are distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**, consistent with the license already recorded for the authors' [Zenodo software artifact](https://zenodo.org/records/17958570). See [LICENSE](LICENSE) for the full text and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external materials and attribution.

## Contact

For code questions or reproducibility issues, please [open a GitHub issue](https://github.com/Tthvic/Model_lineage/issues) with the relevant command, commit hash, environment, and error message.
