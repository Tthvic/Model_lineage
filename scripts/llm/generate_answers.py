#!/usr/bin/env python3
"""
Unified Answer Generation Script
Generates answers for datasets using various model types (Finetune, Adapter, Merge, B-A).
"""

import os
import sys
import json
import torch
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm.model_loader import load_model, load_adapter_model, create_ba_model, load_random_model
from src.llm.dataset_utils import load_dataset_by_name, select_questions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_answers(model, tokenizer, questions, device, max_new_tokens=256):
    """Generate answers for a list of questions."""
    model.eval()
    answers = []
    
    for q in tqdm(questions, desc="Generating answers"):
        try:
            inputs = tokenizer(q, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Simple post-processing to remove the question if repeated
            if answer.startswith(q):
                answer = answer[len(q):].strip()
            answers.append({"question": q, "answer": answer})
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answers.append({"question": q, "answer": "Error generating answer"})
            
    return answers

def main():
    parser = argparse.ArgumentParser(description="Generate Answers for Qwen Models")
    parser.add_argument("--config", default="configs/llm/qwen2_5.yaml", help="Path to config file")
    parser.add_argument("--type", type=str, required=True, 
                        choices=['finetune', 'adapter', 'merge', 'ba', 'random', 'instruct'],
                        help="Type of model processing")
    parser.add_argument("--model_path", type=str, help="Path to the specific model to process")
    parser.add_argument("--model_filter", type=str, help="Process only models matching this string")
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--is_adapter_source", action="store_true", help="For B-A, specify if the source is an adapter")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if not config:
        logger.error(f"Could not load config from {args.config}")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model_path = Path(config['data']['root_dir']) / config['data']['base_model_path']
    instruct_model_path = Path(config['data']['root_dir']) / config['data']['instruct_model_path']
    
    # Determine model paths
    model_paths = []
    if args.model_path:
        model_paths = [Path(args.model_path)]
    else:
        root_dir = Path(config['data']['root_dir'])
        if args.type == 'finetune':
            search_dir = root_dir / config['data']['finetune_models_dir']
            model_paths = list(search_dir.glob("*")) if search_dir.exists() else []
        elif args.type == 'merge':
            search_dir = root_dir / config['data']['merge_models_dir']
            model_paths = list(search_dir.glob("*")) if search_dir.exists() else []
        elif args.type == 'adapter':
            search_dir = root_dir / config['data']['adapter_models_dir']
            model_paths = list(search_dir.glob("*")) if search_dir.exists() else []
        elif args.type == 'ba':
            # For B-A, we need to know if we are processing finetunes, merges, or adapters
            if args.is_adapter_source:
                search_dir = root_dir / config['data']['adapter_models_dir']
            else:
                search_dir = root_dir / config['data']['finetune_models_dir']
            model_paths = list(search_dir.glob("*")) if search_dir.exists() else []
        elif args.type == 'random':
            # Random model is a single instance usually
            model_paths = [Path("random_model")] 
        elif args.type == 'instruct':
            model_paths = [instruct_model_path]

    # Apply filter
    if args.model_filter:
        model_paths = [p for p in model_paths if args.model_filter in p.name]

    if not model_paths:
        logger.warning("No models found to process.")
        return

    # Pre-load datasets to avoid reloading for every model
    tasks = config['data']['tasks']
    loaded_datasets = {}
    dataset_dir = Path(config['data']['root_dir']) / config['data']['raw_datasets_dir']
    
    for task in tasks:
        ds = load_dataset_by_name(task, str(dataset_dir))
        if ds:
            loaded_datasets[task] = ds
        else:
            logger.warning(f"Could not load dataset {task}")

    for model_path in model_paths:
        logger.info(f"Processing model: {model_path.name}")
        
        # Load Model
        if args.type == 'finetune' or args.type == 'merge':
            model, tokenizer = load_model(str(model_path), device=device)
        elif args.type == 'adapter':
            model, tokenizer = load_adapter_model(str(instruct_model_path), str(model_path), device=device)
        elif args.type == 'ba':
            # For B-A, model_path is the target
            model, tokenizer = create_ba_model(
                str(base_model_path), 
                str(instruct_model_path), 
                str(model_path), 
                is_adapter=args.is_adapter_source,
                device=device
            )
        elif args.type == 'random':
            model, tokenizer = load_random_model(str(base_model_path), device=device)
        elif args.type == 'instruct':
            model, tokenizer = load_model(str(instruct_model_path), device=device)
            
        # Generate Answers for each task
        for task, dataset in loaded_datasets.items():
            questions = select_questions(dataset, task)
            if not questions:
                continue
                
            answers = generate_answers(model, tokenizer, questions, device)
            
            # Save results
            # Structure: output_dir / Type / Task / ModelName_runX.jsonl
            if args.output_dir:
                base_output_dir = Path(args.output_dir)
            else:
                base_output_dir = Path(config['data']['root_dir']) / config['data']['dataset_dir']
                
            # Determine sub-folder name
            if args.type == 'ba':
                type_dir = f"{config['data']['ba_dir']}/Adapter" if args.is_adapter_source else f"{config['data']['ba_dir']}/Finetune"
            elif args.type == 'random':
                type_dir = config['data']['random_dir']
            elif args.type == 'instruct':
                type_dir = config['data']['instruct_dir']
            else:
                type_dir = args.type.capitalize()
                
            output_dir = base_output_dir / type_dir / task
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate answers twice (run1 and run2)
            for run_idx in range(1, 3):
                output_file = output_dir / f"{model_path.name}_run{run_idx}.jsonl"
                
                if output_file.exists():
                    logger.info(f"Skipping {output_file}, already exists.")
                    continue
                    
                logger.info(f"Generating run {run_idx} for {model_path.name} on {task}")
                answers = generate_answers(model, tokenizer, questions, device)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for entry in answers:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
                logger.info(f"Saved answers to {output_file}")

if __name__ == "__main__":
    main()
