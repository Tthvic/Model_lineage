#!/usr/bin/env python3
"""
Step 1: Split QA Dataset
Extract different subsets from Model A's 30 QA pairs to finetune Model B
"""

import json
from pathlib import Path
import logging
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step1_split_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_qa_data(qa_file):
    """Load QA data from JSONL file"""
    qa_list = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            qa_list.append(json.loads(line.strip()))
    return qa_list


def convert_to_training_format(qa_list):
    """
    Convert QA data to training format
    Format: {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    """
    training_data = []
    for qa in qa_list:
        training_data.append({
            "messages": [
                {"role": "user", "content": qa['question']},
                {"role": "assistant", "content": qa['answer']}
            ],
            "metadata": {
                "question_index": qa.get('question_index', -1),
                "model": qa.get('model', MODEL_A_NAME)
            }
        })
    return training_data


def split_and_save_qa_data():
    """Split and save QA data into different subsets"""
    logger.info("Loading QA data...")
    
    # Load QA data from arc_challenge or pre-generated file
    # For now, assuming we have a pre-generated QA file
    # TODO: Generate QA from model A if not exists
    
    qa_data_path = DATA_DIR / "qa_from_model_a.jsonl"
    if not qa_data_path.exists():
        logger.error(f"QA data file not found: {qa_data_path}")
        logger.error("Please generate QA data from Model A first")
        return None
    
    qa_data = load_qa_data(qa_data_path)
    logger.info(f"Successfully loaded {len(qa_data)} QA pairs")
    
    # Validate data size
    if len(qa_data) < 30:
        logger.warning(f"Insufficient data: only {len(qa_data)} QA pairs (expected 30)")
    
    # Split data
    splits_info = {}
    for split_name, split_config in QA_SPLITS.items():
        start, end = split_config['range']
        qa_subset = qa_data[start:end]
        
        # Convert to training format
        training_subset = convert_to_training_format(qa_subset)
        
        # Save to file
        output_file = DATA_DIR / f"{split_config['name']}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_subset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        splits_info[split_name] = {
            'file': str(output_file),
            'range': f"{start+1}-{end}",  # Convert to 1-based indexing
            'count': len(training_subset)
        }
        
        logger.info(f"✓ {split_name}: Saved questions {start+1}-{end} ({len(training_subset)} pairs) -> {output_file}")
    
    # Save full 30 QA pairs (for later testing)
    full_training_data = convert_to_training_format(qa_data)
    full_file = DATA_DIR / "qa_full_30.jsonl"
    with open(full_file, 'w', encoding='utf-8') as f:
        for item in full_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"✓ Full data: Saved all 30 questions -> {full_file}")
    
    # Save metadata
    metadata = {
        'source_file': str(qa_data_path),
        'total_qa_count': len(qa_data),
        'splits': splits_info,
        'full_data_file': str(full_file)
    }
    
    metadata_file = DATA_DIR / "split_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("Data splitting complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Metadata saved to: {metadata_file}")
    
    return metadata


if __name__ == "__main__":
    metadata = split_and_save_qa_data()
    if metadata:
        print("\nData split summary:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
