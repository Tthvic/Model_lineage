# Experiments

[Back to README](../README.md)

Run commands from the repository root after [installation](../README.md#installation). Keep datasets and model weights under `data/`, and update paths in the relevant configuration or script before running an experiment.

## Small models

Set `dataset.root` in [mobilenet_caltech.yaml](../configs/small_model/mobilenet_caltech.yaml) to your Caltech-101 directory. The file also specifies the parent/child class subsets, training settings, and output paths.

Train the source models and generate embeddings:

```bash
python scripts/small_model/train_models.py --config configs/small_model/mobilenet_caltech.yaml
python scripts/small_model/generate_embeddings.py --config configs/small_model/mobilenet_caltech.yaml --split train
python scripts/small_model/generate_embeddings.py --config configs/small_model/mobilenet_caltech.yaml --split test
```

Train the standalone MLP lineage detector:

```bash
python scripts/small_model/train_lineage.py --config configs/small_model/mobilenet_caltech.yaml
```

This saves `data/models/small_model/lineage_detector.pth`. It uses a different architecture and checkpoint format from the released encoder/relation-network detectors. To evaluate those released checkpoints, use the [MobileNet and ResNet examples](../README.md#mobilenet-and-resnet).

## Large language models

Available configurations:

| Model family | Configuration |
| --- | --- |
| Qwen 2.5 | [qwen2_5.yaml](../configs/llm/qwen2_5.yaml) |
| Llama 3 | [llama3.yaml](../configs/llm/llama3.yaml) |
| Mistral | [mistral.yaml](../configs/llm/mistral.yaml) |
| Gemma | [gemma.yaml](../configs/llm/gemma.yaml) |
| Phi-4 | [phi4.yaml](../configs/llm/phi4.yaml) |
| Qwen 3 | [qwen3.yaml](../configs/llm/qwen3.yaml) |

The commands below use Qwen 2.5. The configuration defines local paths under `data`, tasks under `data.tasks`, and training settings such as `training.num_epochs`. Source models may require access from their providers.

### Download models and datasets

```bash
python scripts/llm/download_models.py --config configs/llm/qwen2_5.yaml
python scripts/llm/download_datasets.py --config configs/llm/qwen2_5.yaml
```

### Generate answers

Generate outputs from the child, parent, random model, and parameter-difference model:

```bash
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type finetune
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type instruct
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type random
python scripts/llm/generate_answers.py --config configs/llm/qwen2_5.yaml --type ba
```

### Extract embeddings

```bash
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type finetune
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type instruct
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type random
python scripts/llm/generate_embeddings.py --config configs/llm/qwen2_5.yaml --type ba
```

Training requires `Qwen_random` embeddings and matching child / `B-A/Finetune` files for the same tasks. For adapter-derived differences, add `--is_adapter_source` to both `--type ba` commands and generate the child data with `--type adapter`.

### Train the lineage detector

```bash
python scripts/llm/train_lineage.py --config configs/llm/qwen2_5.yaml
```

### Bundled Qwen example

The [Quick Start](../README.md#qwen) uses `data/embeddings/llm/Qwen-1.5B/`; the YAML workflow above uses a separate `Qwen2.5-1.5B` directory.

The bundled data cover `arc_challenge`. Adapter and Merge examples include matching `B-A` embeddings; Finetune examples currently lack a corresponding `B-A/Finetune` bundle. The example script uses batch size 16, learning rate `1e-4`, and 100 epochs. Settings are defined in [qwen_knowledge_relation_training.py](../scripts/llm/qwen_knowledge_relation_training.py), and results go to `data/intermediate/llm/Qwen-1.5B/results/`.

## Diffusion models

These scripts define model paths and training settings directly in Python. [diffusion_config.yaml](../configs/diffusion_config.yaml) is a reference file and is not loaded automatically by the commands below. Pre-generated diffusion embeddings are not included.

### Prepare COCO and generate embeddings

```bash
python scripts/diffusion/download_datasets.py
python scripts/diffusion/generate_embeddings.py
```

The download script places COCO 2014 data under `data/datasets/coco/`. Model and dataset defaults are defined in [DiffusionDataset](../src/diffusion/diffusion_dataset.py). Update those defaults, or pass the desired paths to `DiffusionDataset(...)` in [generate_embeddings.py](../scripts/diffusion/generate_embeddings.py). Extracted U-Net features are saved under `data/embeddings/diffusion/`.

### Train and evaluate

```bash
python scripts/diffusion/train_lineage.py
```

Training saves `data/models/diffusion/checkpoint-<time>/epoch_<n>.pth`. Point the evaluation script to the checkpoint you want to evaluate; its default is `data/models/diffusion/checkpoint.pth`.

```bash
python scripts/diffusion/test_lineage.py
python scripts/diffusion/test_unrelated_dataset.py
```

The second command evaluates on CIFAR-10 probe images. Adaptive attack evaluation is available at:

```bash
python adaptive_attacks/diffusion/test_adaptive_attack.py
```

## Adaptive attacks

Attack scripts use their own model and data paths. Check those settings before running them.

| Model type | Attack | Implementation |
| --- | --- | --- |
| LLM | Knowledge overwriting | [knowledge_overwriting/](../adaptive_attacks/llm/knowledge_overwriting/) |
| LLM | Knowledge infusion | [knowledge_infusion/](../adaptive_attacks/llm/knowledge_infusion/) |
| LLM | Weight pruning | [weight_prune/](../adaptive_attacks/llm/weight_prune/) |
| Classifier | Knowledge overwriting | [knowledge_overwriting/](../adaptive_attacks/classification_model/knowledge_overwriting/) |
| Classifier | Parameter perturbation | [paramerpertur/](../adaptive_attacks/classification_model/paramerpertur/) |
| Classifier | Structure evasion | [Structure_Evasion/](../adaptive_attacks/classification_model/Structure_Evasion/) |
| Diffusion | Adaptive attack evaluation | [test_adaptive_attack.py](../adaptive_attacks/diffusion/test_adaptive_attack.py) |

## Baselines

Each baseline provides its own environment and data preparation instructions.

| Baseline | Code or instructions |
| --- | --- |
| KRM / dataset inference | [Dataset inference README](../Baselines/KRMandIPGUAR/Dataset_inference/README.md) |
| IPGuard | [IPGuard README](../Baselines/KRMandIPGUAR/IPGuard/README.md) |
| Neural Lineage | [Source archive](../Baselines/NeuralLineage-main.zip); see its README |
| Neural Phylogeny | [Source archive](../Baselines/Neural_Phylogeny.zip); see its README |

See [third-party notices](../THIRD_PARTY_NOTICES.md) for attribution and upstream terms.
