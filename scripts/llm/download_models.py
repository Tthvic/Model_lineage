#!/usr/bin/env python3
"""
LLM Model Download Script
Downloads Qwen2.5 models (Base, Instruct, Adapters, Finetunes, Merges) for the lineage project.
Integrates robust downloading logic with mirror support and error handling.
"""

import os
import sys
import time
import logging
import yaml
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError

# ==============================================================================
# Configuration & Constants
# ==============================================================================

# Default Configuration
DEFAULT_CONFIG_PATH = "configs/llm/qwen2_5.yaml"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 30
DOWNLOAD_TIMEOUT = 600

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def get_safe_dirname(repo_id: str) -> str:
    """Convert Hugging Face repo_id to a safe directory name."""
    return repo_id.replace("/", "--")

def is_model_downloaded(dest_dir: str) -> bool:
    """Check if model is already downloaded."""
    if not os.path.isdir(dest_dir):
        return False
    # Check for marker files
    completion_markers = ["config.json", "adapter_config.json", "pytorch_model.bin", "model.safetensors"]
    return any(os.path.exists(os.path.join(dest_dir, marker)) for marker in completion_markers)

def download_model_robust(repo_id: str, dest_dir: str, endpoint: str, token: str):
    """Robust download function with retries and error handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Downloading '{repo_id}' (Attempt {attempt}/{MAX_RETRIES})...")
            logger.info(f"Target: {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            
            # Set timeout
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = DOWNLOAD_TIMEOUT
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest_dir,
                endpoint=endpoint,
                token=token,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"[Success] Model '{repo_id}' downloaded.")
            return True

        except HfHubHTTPError as e:
            logger.error(f"[Failed] HTTP Error for '{repo_id}': {e}")
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    logger.error("Error 401: Authentication failed. Check your HF_TOKEN.")
                    return False
                if e.response.status_code == 404:
                    logger.error(f"Error 404: Model '{repo_id}' not found.")
                    return False
            if attempt < MAX_RETRIES:
                logger.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except RequestException as e:
            logger.error(f"[Failed] Network Error for '{repo_id}': {e}")
            if attempt < MAX_RETRIES:
                logger.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"[Failed] Unknown Error for '{repo_id}': {e}")
            if "No space left on device" in str(e):
                logger.error("Disk space full!")
                return False
            if attempt < MAX_RETRIES:
                logger.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
    logger.error(f"[Give Up] Failed to download '{repo_id}' after {MAX_RETRIES} attempts.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Download LLM Models")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config file")
    parser.add_argument("--token", help="Hugging Face Token (optional, can use env HF_TOKEN)")
    parser.add_argument("--model", help="Specific model repo ID to download (partial match allowed)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if not config:
        logger.error(f"Could not load config from {args.config}")
        sys.exit(1)

    # Construct root path
    root_dir = Path(config['data']['root_dir'])
    # We assume the structure is defined in the config, but for downloading we need to know where to put things.
    # The config has paths like "downloaded_models/LLM/Qwen2.5-1.5B/Finetunes"
    # We can infer the base download directory from the project name or use the paths in config.
    
    # Let's use the paths defined in config['data'] to map categories to directories
    # Mapping: category -> config_key
    category_map = {
        "base_models": "base_model_path", # This is a bit tricky as base_model_path points to a specific model usually
        "adapters": "adapter_models_dir",
        "finetunes": "finetune_models_dir",
        "merges": "merge_models_dir"
    }
    
    # Build the task list from config
    model_download_tasks = {}
    if 'models_to_download' in config:
        for category, models in config['models_to_download'].items():
            if not models: continue
            for model_id in models:
                model_download_tasks[model_id] = category
    else:
        logger.warning("No 'models_to_download' section found in config.")
    
    logger.info(f"Loaded {len(model_download_tasks)} models from config.")

    # Get Token
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        logger.warning("No HF_TOKEN provided. Some models may fail if they require authentication.")
    
    successful_downloads = []
    failed_downloads = []
    skipped_downloads = []

    # Filter tasks if specific model requested
    if args.model:
        tasks_to_run = {k: v for k, v in model_download_tasks.items() if args.model in k}
        if not tasks_to_run:
            logger.error(f"No models found matching '{args.model}'")
            sys.exit(1)
        logger.info(f"Filtered mode: Downloading {len(tasks_to_run)} model(s) matching '{args.model}'")
    else:
        tasks_to_run = model_download_tasks

    for repo_id, category in tasks_to_run.items():
        safe_dirname = get_safe_dirname(repo_id)
        
        # Determine destination directory
        if category == "base_models":
            # Special handling for base models to match the structure expected by config
            # Config usually points to specific base model path, e.g. .../base_models/Qwen--Qwen2.5-1.5B
            # We want to download to .../base_models/
            
            # Try to infer from base_model_path in config
            base_path_relative = config['data'].get('base_model_path', '')
            if base_path_relative:
                # base_path_relative is like "downloaded_models/LLM/Qwen2.5-1.5B/base_models/Qwen--Qwen2.5-1.5B"
                # We want the parent: "downloaded_models/LLM/Qwen2.5-1.5B/base_models"
                base_dest_root = root_dir / Path(base_path_relative).parent
            else:
                # Fallback if not defined
                base_dest_root = root_dir / "models" / "llm" / config['project']['name'] / "base_models"
            
            destination_path = base_dest_root / safe_dirname
            
        else:
            # For adapters, finetunes, merges
            dir_key = f"{category}_models_dir" # e.g. adapter_models_dir
            rel_path = config['data'].get(dir_key)
            if rel_path:
                destination_path = root_dir / rel_path / safe_dirname
            else:
                logger.warning(f"Unknown category directory for {category}, skipping {repo_id}")
                continue
        
        if is_model_downloaded(str(destination_path)):
            logger.info(f"[Skip] '{repo_id}' already exists.")
            skipped_downloads.append(repo_id)
            successful_downloads.append(repo_id)
        else:
            if download_model_robust(repo_id, str(destination_path), HF_MIRROR_ENDPOINT, token):
                successful_downloads.append(repo_id)
            else:
                failed_downloads.append(repo_id)
        
        logger.info("-" * 60)

    logger.info("="*60)
    logger.info(f"Total Tasks: {len(model_download_tasks)}")
    logger.info(f"Success: {len(successful_downloads)} (Skipped: {len(skipped_downloads)})")
    logger.info(f"Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        logger.warning("Failed models:")
        for m in failed_downloads:
            logger.warning(f" - {m}")
        sys.exit(1)
    else:
        logger.info("All downloads completed successfully.")

if __name__ == "__main__":
    main()