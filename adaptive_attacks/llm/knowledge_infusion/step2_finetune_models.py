#!/usr/bin/env python3
"""
Step 2: Finetune Model B with Different QA Subsets to Create B1, B2, B3
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import logging
from pathlib import Path
from datetime import datetime
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step2_finetune.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_jsonl_dataset(file_path):
    """Load dataset from JSONL format"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)


def format_chat_template(example, tokenizer):
    """Format chat data using tokenizer's chat template"""
    messages = example['messages']
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": text}


def tokenize_function(example, tokenizer, max_length):
    """Tokenize text and prepare labels"""
    result = tokenizer(
        example['text'],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def finetune_model(model_name, data_file, output_dir):
    """
    Finetune a model with LoRA
    
    Args:
        model_name: Name of the finetuned model (B1, B2, B3)
        data_file: Path to training data file
        output_dir: Directory to save the model
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting finetuning: {model_name}")
    logger.info(f"Training data: {data_file}")
    logger.info(f"{'='*60}\n")
    
    # Load tokenizer and model
    logger.info(f"Loading base model B: {MODEL_B_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_B_PATH,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_B_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Disable cache for gradient checkpointing
    model.config.use_cache = False
    
    # Configure LoRA
    if FINETUNE_CONFIG['use_lora']:
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=FINETUNE_CONFIG['lora_r'],
            lora_alpha=FINETUNE_CONFIG['lora_alpha'],
            target_modules=FINETUNE_CONFIG['lora_target_modules'],
            lora_dropout=FINETUNE_CONFIG['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Enable gradients for input embeddings
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Load and process dataset
    logger.info(f"Loading training data: {data_file}")
    dataset = load_jsonl_dataset(data_file)
    logger.info(f"Training samples: {len(dataset)}")
    
    # Format data
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names
    )
    
    # Tokenize
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, FINETUNE_CONFIG['max_length']),
        remove_columns=['text']
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=FINETUNE_CONFIG['num_epochs'],
        per_device_train_batch_size=FINETUNE_CONFIG['batch_size'],
        gradient_accumulation_steps=FINETUNE_CONFIG['gradient_accumulation_steps'],
        learning_rate=FINETUNE_CONFIG['learning_rate'],
        warmup_ratio=FINETUNE_CONFIG['warmup_ratio'],
        logging_steps=FINETUNE_CONFIG['logging_steps'],
        save_steps=FINETUNE_CONFIG['save_steps'],
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        logging_dir=str(LOGS_DIR / f"{model_name}_training_logs")
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    logger.info(f"Training complete! Duration: {end_time - start_time}")
    
    # Save model
    logger.info(f"Saving model to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'base_model': MODEL_B_PATH,
        'training_data': str(data_file),
        'training_samples': len(dataset),
        'training_time': str(end_time - start_time),
        'config': FINETUNE_CONFIG,
        'output_dir': str(output_dir)
    }
    
    info_file = output_dir / f"{model_name}_training_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training info saved to: {info_file}")
    
    # Clean up GPU memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return training_info


def main():
    """Main function: Finetune B1, B2, B3 sequentially"""
    all_training_info = {}
    
    for model_variant, split_config in QA_SPLITS.items():
        # Prepare paths
        data_file = DATA_DIR / f"{split_config['name']}.jsonl"
        output_dir = MODELS_DIR / model_variant
        
        # Check if data file exists
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            logger.error("Please run step1_split_data.py first")
            continue
        
        # Finetune model
        try:
            training_info = finetune_model(model_variant, data_file, output_dir)
            all_training_info[model_variant] = training_info
        except Exception as e:
            logger.error(f"Failed to finetune {model_variant}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Save all training info
    summary_file = RESULTS_DIR / "finetune_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_training_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("All models finetuned successfully!")
    logger.info(f"{'='*60}")
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary
    print("\nFinetuning Summary:")
    for model_name, info in all_training_info.items():
        print(f"\n{model_name}:")
        print(f"  Training samples: {info['training_samples']}")
        print(f"  Training time: {info['training_time']}")
        print(f"  Saved to: {info['output_dir']}")


if __name__ == "__main__":
    main()
