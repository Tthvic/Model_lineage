# download_qwen2_5_models.py
import os
import sys
import time
import logging


NEW_CACHE_DIR = "/data/huggingface_cache"
os.environ['HF_HOME'] = NEW_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = NEW_CACHE_DIR
os.makedirs(NEW_CACHE_DIR, exist_ok=True)

HF_AUTH_TOKEN = os.getenv("HF_TOKEN")
# ==============================================================================

from huggingface_hub import snapshot_download
from requests.exceptions import RequestException
from huggingface_hub.utils import HfHubHTTPError

# ==============================================================================
# 1. Configuration: files and directory layout
# ==============================================================================

# --- Basic configuration ---
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
ROOT_MODELS_DIR = "/data/ALLMS/downloaded_models/Qwen2.5-1.5B"
LOG_FILE = "download_qwen2_5_log.txt"

# --- Download task configuration ---
MODEL_DOWNLOAD_TASKS = {
    # Adapters
    "r1char9/Oblivion2.5-1.5B-Instruct-v1": "Adapters",
    "DreamGallery/Qwen-Qwen2.5-1.5B-Instruct-1727452927": "Adapters",
    "silent666/Qwen-Qwen2.5-1.5B-Instruct-1727478552": "Adapters",
    "shibing624/chinese-text-correction-1.5b-lora": "Adapters",
    "johnnyllm/task-14-Qwen-Qwen2.5-1.5B-Instruct": "Adapters",
    "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16v3": "Adapters",
    "nekokiku/task-15-Qwen-Qwen2.5-1.5B-Instruct": "Adapters",
    "Superrrdamn/task-16-Qwen-Qwen2.5-1.5B-Instruct": "Adapters",
    "dir00/task-15-Qwen-Qwen2.5-1.5B-Instruct": "Adapters",
    "Arthur-77/QWEN2.5-1.5B-medical-finetuned": "Adapters",
    "Hazde/careerbot_PG6_Qwen_Qwen2.5-1.5B-Instruct_model_LoRA_5": "Adapters",
    "nekokiku/task-17-Qwen-Qwen2.5-1.5B-Instruct": "Adapters",
    "1-lock/58c91d56-17f7-4477-9f3c-7ce9511f7d3c": "Adapters",
    "eeeebbb2/9edc3b53-18bc-472a-926b-a9b457403b4f": "Adapters",
    "DeepDream2045/ca982efa-b911-43d7-8ec0-0b76f01c778c": "Adapters",

    # Finetune
    "nvidia/OpenReasoning-Nemotron-1.5B": "Finetunes",
    "jaeyong2/Qwen2.5-1.5B-Instruct-Thai-SFT": "Finetunes",
    "katanemo/Arch-Router-1.5B": "Finetunes",
    "shibing624/chinese-text-correction-1.5b": "Finetunes",
    "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct": "Finetunes",
    "jaeyong2/Qwen2.5-1.5B-Instruct-Viet-SFT": "Finetunes",
    "xavierwoon/cesterqwen": "Finetunes",
    "CMLL/ZhongJing-3-1_5b_V2": "Finetunes",
    "osllmai-community/Qwen2.5-1.5B-Instruct": "Finetunes",
    "cycloarcane/QWARG-test": "Finetunes",
    "XueyingJia/qwen2.5-1.5b-oaif": "Finetunes",
    "blakenp/Qwen2.5-1.5B-Policy2": "Finetunes",
    "SakanaAI/TinySwallow-1.5B-Instruct": "Finetunes",
    "Sakalti/Test-1003": "Finetunes",
    "winstcha/context-1.5B": "Finetunes",
    
    # Merges
    "bunnycore/Qwen2.5-1.5B-Matrix": "Merges",
    "Hjgugugjhuhjggg/mergekit-model_stock-xtxndeh": "Merges",
    "Hjgugugjhuhjggg/mergekit-ties-kxlqekl": "Merges",
    "Sakalti/Crystal_1.8b": "Merges",
    "Xiaojian9992024/Singularity-Qwen2.5-1.5B": "Merges",
    "DavidAU/Qwen2.5-MOE-6x1.5B-DeepSeek-Reasoning-e32": "Merges",
    "Kukedlc/NeuralQwen-2.5-1.5B-Spanish": "Merges",
    "Marsouuu/MiniQwenMathExpert-ECE-PRYMMAL-Martial": "Merges",
    "LilRg/ECE-1B-merge-PRYMMAL": "Merges",
    "tuanpasg/puffin-linear-coder-ties-2": "Merges",
    "mergekit-community/SuperQwen-2.5-1.5B": "Merges",
    "mergekit-community/Qwen3-1.5B-Instruct": "Merges",
    "bunnycore/AceQwen2.5-1.5B-Sce": "Merges",
    "prithivMLmods/Qwen2.5-1.5B-DeepSeek-R1-Instruct": "Merges",
    "kunal732/qwenreader3": "Merges",
}

# --- Fault tolerance configuration ---
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 20


# ==============================================================================
# 2. Core script logic
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
    """Check whether a model has been fully downloaded."""
    if not os.path.isdir(dest_dir):
        return False
    completion_markers = ["config.json", "adapter_config.json", "pytorch_model.bin", "model.safetensors"]
    return any(os.path.exists(os.path.join(dest_dir, marker)) for marker in completion_markers)

def download_model_robust(repo_id: str, dest_dir: str, endpoint: str, token: str):
    """Robust download function with retry and error handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Start downloading '{repo_id}' (attempt {attempt}/{MAX_RETRIES})...")
            logging.info(f"Destination directory: {dest_dir}")
            os.makedirs(dest_dir, exist_ok=True)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest_dir,
                endpoint=endpoint,
                token=token,
                local_dir_use_symlinks=False,
            )
            
            logging.info(f"[SUCCESS] Model '{repo_id}' downloaded or verified.")
            return True

        except HfHubHTTPError as e:
            logging.error(f"[ERROR] HTTP error while downloading '{repo_id}': {e}")
            if e.response.status_code == 401:
                logging.error(f"HTTP 401 Unauthorized. Invalid token or insufficient permissions for this model.")
                return False
            if e.response.status_code == 404:
                logging.error(f"HTTP 404 Not Found. Model '{repo_id}' does not exist on the Hub.")
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
            if "No space left on device" in str(e):
                logging.error("Detected no space left on device. Please check disk space at cache directory.")
                return False
            if attempt < MAX_RETRIES:
                logging.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            
    logging.error(f"[GIVE UP] Model '{repo_id}' failed after max retries.")
    return False

def main():
    """Main entry function."""
    setup_logging(LOG_FILE)
    
    # Check token availability at start
    if not HF_AUTH_TOKEN:
        logging.error("ERROR: Environment variable HF_TOKEN not set.")
        logging.error("Please set your Hugging Face access token: export HF_TOKEN='hf_...'")
        sys.exit(1) # exit script

    successful_downloads = []
    failed_downloads = []
    skipped_downloads = []

    logging.info("="*60)
    logging.info("Starting Qwen2.5-1.5B model download script")
    logging.info(f"Root destination: {ROOT_MODELS_DIR}")
    logging.info(f"Hugging Face cache (HF_HOME): {os.environ.get('HF_HOME', 'not set, using default')}")
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
            if download_model_robust(repo_id, destination_path, HF_MIRROR_ENDPOINT, token=HF_AUTH_TOKEN):
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
