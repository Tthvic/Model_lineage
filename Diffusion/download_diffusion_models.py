# File: download_diffusion_models.py
import os
import sys
import time
import logging

# ==============================================================================
# Core: set cache directory BEFORE importing huggingface_hub and read auth token
# ==============================================================================
from pathlib import Path

# Project root: Model_lineage (Diffusion is one level below root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use repository-local cache to avoid absolute paths and preserve anonymity
NEW_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ['HF_HOME'] = str(NEW_CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(NEW_CACHE_DIR)
os.makedirs(NEW_CACHE_DIR, exist_ok=True)

# Read auth token from env to avoid local login files
HF_AUTH_TOKEN = os.getenv("HF_TOKEN")
# ==============================================================================

from huggingface_hub import snapshot_download
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.errors import LocalEntryNotFoundError

# ==============================================================================
# 1. Configuration
# ==============================================================================

# --- Base configuration ---
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
# Store downloaded models inside the repository to keep paths relative
ROOT_MODELS_DIR = str(PROJECT_ROOT / "downloaded_models" / "DiffusionModels")
LOG_FILE = "download_diffusion_models_log.txt"

# --- Download tasks ---
DIFFUSION_MODEL_DOWNLOAD_TASKS = {
    # Stable Diffusion base model
    "stabilityai/stable-diffusion-2": "StableDiffusion",
    
    # Animal-related models
    "jasbir/dog_model": "AnimalModels",
    "sd-concepts-library/jerry-dog": "AnimalModels", 
    "sd-concepts-library/animal-toy": "AnimalModels",
    "ShuhongZheng/cat_sd2": "AnimalModels",
    "ShuhongZheng/wolf_sd2": "AnimalModels",
    
    # Flowers and plants
    "sd-concepts-library/canna-lily-flowers102": "PlantModels",
    "DaichiT/wood": "PlantModels",
    
    # Objects and concepts
    "sd-concepts-library/gphone03": "ObjectModels",
    "sd-concepts-library/gphone01": "ObjectModels",
    "sd-concepts-library/yvmqznrm": "ObjectModels",
    "sd-concepts-library/babies-poster": "ObjectModels",
    "sd-concepts-library/musecat": "ObjectModels",
    "sd-concepts-library/mersh": "ObjectModels",
    "sd-concepts-library/dongqichang": "ObjectModels",
    "sd-concepts-library/clothes": "ObjectModels",
    "sd-concepts-library/ethos-spirit": "ConceptModels",
    
    # Special utilities
    "reachomk/gen2seg-sd": "SpecialModels",
}

# --- Fault tolerance ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 20


# ==============================================================================
# 2. Main logic
# ==============================================================================

def setup_logging(log_file: str):
    """Configure logging to both console and file with timezone-aware formatter."""
    import pytz
    from datetime import datetime
    
    # Custom formatter with Asia/Shanghai timezone
    class BeijingFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            beijing_tz = pytz.timezone('Asia/Shanghai')
            ct = datetime.fromtimestamp(record.created, tz=beijing_tz)
            if datefmt:
                s = ct.strftime(datefmt)
            else:
                s = ct.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            return s
    
    formatter = BeijingFormatter('%(asctime)s - [%(levelname)s] - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

def get_safe_dirname(repo_id: str) -> str:
    """Convert a Hugging Face repo_id to a filesystem-safe directory name."""
    return repo_id.replace("/", "--")

def is_model_downloaded(dest_dir: str) -> bool:
    """Check whether a diffusion model appears to be fully downloaded."""
    if not os.path.isdir(dest_dir):
        return False
    # Check presence of characteristic files
    completion_markers = [
        "model_index.json",
        "unet/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.bin",
        "vae/diffusion_pytorch_model.bin",
        "text_encoder/pytorch_model.bin",
    ]
    # If any marker exists, consider the model present or partially present
    return any(os.path.exists(os.path.join(dest_dir, marker)) for marker in completion_markers)

def download_model_robust(repo_id: str, dest_dir: str, endpoint: str, token: str):
    """Robust downloader with retries and clear error handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Downloading diffusion model '{repo_id}' (attempt {attempt}/{MAX_RETRIES})...")
            logging.info(f"Destination directory: {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest_dir,
                endpoint=endpoint,
                token=token if token else None,
                local_dir_use_symlinks=False,
                ignore_patterns=None,
            )
            
            logging.info(f"[OK] Diffusion model '{repo_id}' downloaded or verified.")
            return True

        except HfHubHTTPError as e:
            logging.error(f"[HTTP ERROR] While downloading '{repo_id}': {e}")
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 401:
                    logging.error("401 Unauthorized: Invalid token or no permission.")
                    return False
                elif e.response.status_code == 403:
                    logging.error("403 Forbidden: Special permission may be required.")
                    return False
                elif e.response.status_code == 404:
                    logging.error(f"404 Not Found: '{repo_id}' does not exist on the Hub.")
                    return False
            else:
                logging.error("HTTP error without status code; possibly permission or network issue.")
                if "403" in str(e) or "Forbidden" in str(e):
                    logging.error("Detected 403; the model may require special access.")
                    return False
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except RequestException as e:
            logging.error(f"[NETWORK ERROR] While downloading '{repo_id}': {e}")
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except LocalEntryNotFoundError as e:
            logging.error(f"[ACCESS ERROR] While downloading '{repo_id}': {e}")
            if "403" in str(e) or "Forbidden" in str(e):
                logging.error("This model requires special permission.")
                return False
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
        except Exception as e:
            logging.error(f"[UNKNOWN ERROR] While downloading '{repo_id}': {e}")
            if "No space left on device" in str(e):
                logging.error("No space left on device. Check cache directory disk space.")
                return False
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
    logging.error(f"[GIVE UP] '{repo_id}' failed after max retries.")
    return False

def main():
    """Entry point for downloading the configured diffusion models."""
    setup_logging(LOG_FILE)
    
    if not HF_AUTH_TOKEN:
        logging.warning("HF_TOKEN is not set. Private models may fail, public models will work.")
        logging.warning("To access private models, export HF_TOKEN='hf_...'")
    else:
        logging.info("HF_TOKEN detected; will attempt to download all models.")

    successful_downloads = []
    failed_downloads = []
    skipped_downloads = []

    logging.info("="*60)
    logging.info("Starting diffusion model download script")
    logging.info(f"Models root directory: {ROOT_MODELS_DIR}")
    logging.info(f"Hugging Face cache (HF_HOME): {os.environ.get('HF_HOME', 'not set, using default')}")
    logging.info(f"Mirror endpoint: {HF_MIRROR_ENDPOINT}")
    logging.info("="*60 + "\n")

    for repo_id, category in DIFFUSION_MODEL_DOWNLOAD_TASKS.items():
        safe_dirname = get_safe_dirname(repo_id)
        destination_path = os.path.join(ROOT_MODELS_DIR, category, safe_dirname)
        
        if is_model_downloaded(destination_path):
            logging.info(f"[SKIP] '{repo_id}' already present and appears complete.")
            skipped_downloads.append(repo_id)
            successful_downloads.append(repo_id)
        else:
            if download_model_robust(repo_id, destination_path, HF_MIRROR_ENDPOINT, token=HF_AUTH_TOKEN):
                successful_downloads.append(repo_id)
            else:
                failed_downloads.append(repo_id)
        
        logging.info("-" * 60)

    # --- Summary ---
    logging.info("\n" + "="*60)
    logging.info("All diffusion model download tasks completed.")
    total_tasks = len(DIFFUSION_MODEL_DOWNLOAD_TASKS)
    logging.info(f"Total: {total_tasks} | Ready: {len(successful_downloads)} (skipped {len(skipped_downloads)}) | Failed: {len(failed_downloads)}")

    if failed_downloads:
        logging.warning("The following models failed:")
        for repo_id in failed_downloads:
            logging.warning(f"  - {repo_id}")
    
    logging.info(f"Detailed log saved to: {LOG_FILE}")
    logging.info("="*60)


if __name__ == "__main__":
    main()
