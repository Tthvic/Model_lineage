#!/usr/bin/env python3
"""
Dataset Download Script
Downloads required datasets for the Model Lineage project.
"""

import os
import time
import logging
import yaml
import argparse
from pathlib import Path
from datasets import load_dataset
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError

# Default Configuration
DEFAULT_CONFIG_PATH = "configs/llm/qwen2_5.yaml"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

# Dataset configuration mapping
DATASET_CONFIGS = {
    "mmlu": {
        "id": "cais/mmlu",
        "config": "all",
        "description": "MMLU - Massive Multitask Language Understanding benchmark"
    },
    "arc_challenge": {
        "id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "description": "ARC-C - AI2 Reasoning Challenge (Challenge set)"
    },
    "humaneval": {
        "id": "openai_humaneval",
        "config": None,
        "description": "HumanEval - Code generation benchmark"
    },
    "gsm8k": {
        "id": "openai/gsm8k",
        "config": "main",
        "description": "GSM-8K - Grade School Math 8K problems"
    },
    "hellaswag": {
        "id": "Rowan/hellaswag",
        "config": None,
        "description": "HellaSwag - Commonsense reasoning benchmark"
    },
    "mgsm": {
        "id": "juletxara/mgsm",
        "config": "en",
        "trust_remote_code": True,
        "description": "MGSM - Multilingual Grade School Math"
    }
}

def is_dataset_downloaded(dest_dir):
    """Check if dataset is already downloaded."""
    if not os.path.isdir(dest_dir):
        return False
    
    # Check for common dataset files
    common_files = ["dataset_info.json", "state.json", "data", "train", "test", "validation"]
    for file_or_dir in common_files:
        if os.path.exists(os.path.join(dest_dir, file_or_dir)):
            return True
            
    # If directory is not empty, assume downloaded
    if os.listdir(dest_dir):
        return True
        
    return False

def download_dataset_robust(dataset_id, dest_dir, config_name=None, trust_remote_code=False, max_retries=3):
    """Robust dataset download with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading '{dataset_id}' (Attempt {attempt}/{max_retries})...")
            os.makedirs(dest_dir, exist_ok=True)
            
            dataset = load_dataset(
                dataset_id, 
                config_name,
                cache_dir=dest_dir,
                token=True,
                trust_remote_code=trust_remote_code
            )
            
            dataset.save_to_disk(dest_dir)
            logger.info(f"[Success] Dataset '{dataset_id}' downloaded to {dest_dir}")
            return True

        except Exception as e:
            logger.error(f"[Failed] Error downloading '{dataset_id}': {e}")
            if attempt < max_retries:
                time.sleep(10)
            
    return False

def main():
    parser = argparse.ArgumentParser(description="Download Datasets for Model Lineage")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if not config:
        logger.error(f"Could not load config from {args.config}")
        sys.exit(1)
    
    root_dir = Path(config['data']['root_dir'])
    datasets_dir = root_dir / config['data']['raw_datasets_dir']
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = config['data']['tasks']
    
    logger.info(f"Downloading datasets to: {datasets_dir}")
    
    for task in tasks:
        if task not in DATASET_CONFIGS:
            logger.warning(f"Task '{task}' not found in download configuration. Skipping.")
            continue
            
        ds_config = DATASET_CONFIGS[task]
        dest_dir = datasets_dir / task
        
        if is_dataset_downloaded(dest_dir):
            logger.info(f"[Skipped] Dataset '{task}' already exists.")
            continue
            
        download_dataset_robust(
            ds_config["id"], 
            str(dest_dir), 
            config_name=ds_config.get("config"),
            trust_remote_code=ds_config.get("trust_remote_code", False)
        )

if __name__ == "__main__":
    main()
