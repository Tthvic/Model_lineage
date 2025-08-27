# Filename: download_qwen2_5_base_model.py
import os
import time
import logging
from huggingface_hub import snapshot_download
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError

# ==============================================================================
# 1. Configuration: files and directory layout
# ==============================================================================

# --- Basic configuration ---
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"  # <--- put your mirror endpoint here
ROOT_MODELS_DIR = "/data/shangzhuoyi/ALLMS/downloaded_models/Qwen2.5-1.5B"
LOG_FILE = "download_qwen2_5_base_log.txt"

# --- Download task configuration ---
MODEL_DOWNLOAD_TASKS = {
    # Base models
    "Qwen/Qwen2.5-1.5B-Instruct": "base_models",
    "Qwen/Qwen2.5-1.5B": "base_models",
}

# --- Fault tolerance configuration ---
MAX_RETRIES = 3  # max retries per model
RETRY_DELAY_SECONDS = 20  # retry delay seconds


# ==============================================================================
# 2. Core script logic (with auth and pre-checks)
# ==============================================================================

def setup_logging(log_file: str):
    """Configure logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def get_safe_dirname(repo_id: str) -> str:
    """Convert Hugging Face repo_id to a filesystem-safe directory name."""
    return repo_id.replace("/", "--")

def is_model_downloaded(dest_dir: str) -> bool:
    """Filesystem pre-check for whether a base model is fully downloaded."""
    if not os.path.isdir(dest_dir):
        return False
    # key files for base models
    completion_markers = ["config.json", "tokenizer.json", "pytorch_model.bin", "model.safetensors"]
    for marker in completion_markers:
        if os.path.exists(os.path.join(dest_dir, marker)):
            return True
    return False

def download_model_robust(repo_id: str, dest_dir: str, endpoint: str):
    """Robust download with retries, error handling, and auth token."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Start downloading '{repo_id}' (attempt {attempt}/{MAX_RETRIES})...")
            logging.info(f"Destination directory: {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest_dir,
                endpoint=endpoint,
                token=True, 
                local_dir_use_symlinks=False,
            )
            
            logging.info(f"[SUCCESS] Model '{repo_id}' downloaded or verified.")
            return True

        except HfHubHTTPError as e:
            logging.error(f"[ERROR] HTTP error while downloading '{repo_id}': {e}")
            if e.response.status_code == 401:
                logging.error(f"HTTP 401 Unauthorized. Ensure 'huggingface-cli login' and access rights. Giving up.")
                return False
            if e.response.status_code == 404:
                logging.error(f"HTTP 404 Not Found. Model '{repo_id}' does not exist on the Hub. Giving up.")
                return False
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except RequestException as e:
            logging.error(f"[ERROR] Network error while downloading '{repo_id}': {e}")
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except Exception as e:
            logging.error(f"[ERROR] Unknown error while downloading '{repo_id}': {e}")
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
    logging.error(f"[GIVE UP] Model '{repo_id}' failed after max retries ({MAX_RETRIES}).")
    return False

def main():
    """Main entry function with pre-checks."""
    setup_logging(LOG_FILE)
    
    successful_downloads = []
    failed_downloads = []
    skipped_downloads = []

    logging.info("="*60)
    logging.info("Starting base model download script for Qwen2.5-1.5B-Instruct")
    logging.info(f"Root destination: {ROOT_MODELS_DIR}")
    logging.info(f"Mirror endpoint: {HF_MIRROR_ENDPOINT}")
    logging.info("="*60 + "\n")

    for repo_id, category in MODEL_DOWNLOAD_TASKS.items():
        safe_dirname = get_safe_dirname(repo_id)
        destination_path = os.path.join(ROOT_MODELS_DIR, category, safe_dirname)
        
        if is_model_downloaded(destination_path):
            logging.info(f"[SKIP] Model '{repo_id}' directory exists and appears complete. Skipping.")
            skipped_downloads.append(repo_id)
            successful_downloads.append(repo_id)
        else:
            if download_model_robust(repo_id, destination_path, HF_MIRROR_ENDPOINT):
                successful_downloads.append(repo_id)
            else:
                failed_downloads.append(repo_id)
        
        logging.info("-" * 60)

    # --- Final summary ---
    logging.info("\n" + "="*60)
    logging.info("All download tasks completed.")
    total_tasks = len(MODEL_DOWNLOAD_TASKS)
    logging.info(f"Total: {total_tasks} | Succeeded/ready: {len(successful_downloads)} (skipped {len(skipped_downloads)}) | Failed: {len(failed_downloads)}")

    if failed_downloads:
        logging.warning("The following models failed to download:")
        for repo_id in failed_downloads:
            logging.warning(f"  - {repo_id}")
    
    logging.info(f"Detailed logs saved to: {LOG_FILE}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
