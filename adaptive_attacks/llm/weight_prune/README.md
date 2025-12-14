# Weight Pruning Experiment

Analyze the impact of model pruning on language ability (PPL) and knowledge relationship features (TPR/FPR).

## Overview

This experiment prunes Qwen2.5-1.5B-Instruct at different ratios and evaluates:
1. **Language Modeling Ability** (PPL on WikiText-2)
2. **Knowledge Relationship Features** (TPR/FPR using lineage detection)

**Core Question**: Does the pruned model still retain the "knowledge fingerprint" of the original model?

## Quick Start

### 1. Setup Environment

```bash
# Set environment variables
export PARENT_MODEL_PATH="/path/to/Qwen2.5-1.5B-Instruct"
export NEGATIVE_MODEL_PATH="/path/to/Qwen2.5-Math-1.5B"
export LINEAGE_MODEL_PATH="/path/to/lineage_model.pth"
export LINEAGE_CODE_PATH="/path/to/lineage_model_code"
export GPU_ID="0"
```

Or create a `.env` file (see `.env.example`).

### 2. Run Experiment

```bash
python3 run_experiment.py
```

## Configuration

Edit `config.py` to customize:

- `PRUNING_RATIOS`: Pruning ratios to test (default: [0.0, 0.2, 0.3, 0.5, 0.7])
- `PPL_NUM_SAMPLES`: Number of samples for PPL evaluation (default: 50)
- `LINEAGE_NUM_SAMPLES`: Number of samples for lineage evaluation (default: 20)
- `LINEAGE_THRESHOLD`: Threshold for TPR/FPR calculation (default: 0.5)

## Experiment Design

### Pruning

- **Method**: Magnitude-based pruning
- **Target**: Qwen2.5-1.5B-Instruct
- **Ratios**: 0%, 20%, 30%, 50%, 70%

### PPL Evaluation

- **Dataset**: WikiText-2
- **Metric**: Perplexity (lower is better)
- **Purpose**: Measure language modeling ability degradation

### Lineage Evaluation

- **Positive Samples**: Original Qwen-Instruct (parent) + Pruned Qwen-Instruct (child)
- **Negative Samples**: Original Qwen-Instruct (parent) + Qwen-Math (unrelated)
- **Metrics**: TPR (True Positive Rate), FPR (False Positive Rate), Accuracy
- **Purpose**: Measure knowledge relationship feature preservation

## Expected Results

```
Pruning  PPL      PPL Change  TPR      FPR      Accuracy
0%       18.06    +0.0%       1.0000   0.0000   1.0000
20%      21.25    +17.7%      0.95?    0.01?    0.98?
30%      29.33    +62.4%      0.85?    0.02?    0.94?
50%      349.49   +1835%      0.60?    0.05?    0.85?
70%      284100   崩溃        0.20?    0.10?    0.60?
```

### Key Findings

- **If TPR decrease < PPL increase**: Knowledge relationship is more robust
- **If TPR decrease > PPL increase**: Knowledge relationship is more fragile

## Output Files

```
results/
├── pruning_results.json    # Complete results
├── embeddings/             # Extracted embeddings
└── pruned_models/          # Pruned models (optional)

logs/
└── experiment_*.log        # Experiment logs
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- datasets
- tqdm
- numpy

```bash
pip install torch transformers datasets tqdm numpy
```

## Citation

If you use this code, please cite:

```bibtex
@misc{weight_pruning_2024,
  title={Weight Pruning and Lineage Preservation Analysis},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
