#!/usr/bin/env python3
"""
Unified Embedding Generation Script for Qwen Models
Generates embeddings for Finetune, Adapter, B-A, etc.
"""

import os
import sys
import torch
import json
import logging
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl_data(file_path):
    """Load JSONL data."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        logger.info(f"Successfully loaded {len(data)} entries from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data file {file_path}: {e}")
        return []

def get_embeddings(data, model, tokenizer, device):
    """Generate embeddings from data."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for entry in tqdm(data, desc="Generating embeddings"):
            try:
                question = entry.get("question", "")
                answer = entry.get("answer", "")
                text = f"Question: {question}\nAnswer: {answer}"
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use the last hidden state of the last token as embedding (or average pooling)
                # Here we use the last token's hidden state as per original script logic (implied)
                # Or usually for sentence embedding, we might use mean pooling.
                # Let's assume the last hidden state of the last token for now.
                last_hidden_state = outputs.hidden_states[-1]
                embedding = last_hidden_state[:, -1, :].cpu()
                
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error processing entry: {e}")
                embeddings.append(torch.zeros(1, model.config.hidden_size))
                
    return torch.cat(embeddings, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Generate Embeddings for Qwen Models")
    parser.add_argument("--config", default="configs/llm/qwen2_5.yaml", help="Path to config file")
    parser.add_argument("--type", type=str, required=True, 
                        choices=['finetune', 'adapter', 'merge', 'ba', 'random', 'instruct'],
                        help="Type of embeddings to generate")
    parser.add_argument("--task", type=str, help="Specific task to process (optional)")
    parser.add_argument("--model_filter", type=str, help="Process only models matching this string")
    parser.add_argument("--is_adapter_source", action="store_true", help="For B-A, specify if the source is an adapter")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if not config:
        logger.error(f"Could not load config from {args.config}")
        sys.exit(1)
    
    root_dir = Path(config['data']['root_dir'])
    dataset_dir = root_dir / config['data']['dataset_dir']
    output_dir = root_dir / config['data']['embedding_dir']
    base_model_path = root_dir / config['data']['base_model_path']
    
    # Determine input/output subdirectories based on type (Matching generate_answers.py logic)
    if args.type == 'ba':
        sub_dir = f"{config['data']['ba_dir']}/Adapter" if args.is_adapter_source else f"{config['data']['ba_dir']}/Finetune"
    elif args.type == 'random':
        sub_dir = config['data']['random_dir']
    elif args.type == 'instruct':
        sub_dir = config['data']['instruct_dir']
    else:
        sub_dir = args.type.capitalize()
        
    input_path = dataset_dir / sub_dir
    output_path = output_dir / sub_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Model
    logger.info(f"Loading model from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).to(device)
    
    tasks = config['data']['tasks']
    if args.task:
        if args.task not in tasks:
            logger.warning(f"Task {args.task} not in config task list.")
        tasks = [args.task]
        
    for task in tasks:
        logger.info(f"Processing task: {task}")
        task_input_dir = input_path / task
        task_output_dir = output_path / task
        task_output_dir.mkdir(exist_ok=True)
        
        if not task_input_dir.exists():
            logger.warning(f"Input directory not found: {task_input_dir}")
            continue
            
        # Process each file in the task directory
        # Assuming input files are jsonl and we want to generate .pth embeddings
        # Note: The original script logic iterates over files. 
        # We need to adapt this to the specific file structure of the user.
        # For now, let's assume standard jsonl files.
        
        files = list(task_input_dir.glob("*.jsonl"))
        
        # Apply filter
        if args.model_filter:
            files = [f for f in files if args.model_filter in f.name]

        for file in files:
            logger.info(f"Processing file: {file.name}")
            data = load_jsonl_data(file)
            if not data:
                continue
                
            embeddings = get_embeddings(data, model, tokenizer, device)
            
            output_file = task_output_dir / f"{file.stem}.pth"
            torch.save(embeddings, output_file)
            logger.info(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()
