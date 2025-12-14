#!/usr/bin/env python3
"""
Step 2: Finetune target model with different attack intensities

对目标模型B (Qwen2.5-1.5B-Policy2) 进行三种强度的知识覆写攻击：
- Low (10%): 使用20个QA对微调 → Qwen2.5-1.5B-Policy2-Attacked-Low
- Medium (30%): 使用60个QA对微调 → Qwen2.5-1.5B-Policy2-Attacked-Medium
- High (50%): 使用100个QA对微调 → Qwen2.5-1.5B-Policy2-Attacked-High
"""

import os
import json
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
from pathlib import Path
from config import *

# 设置日志
def setup_logger(intensity_key):
    """为每个强度设置独立的日志"""
    logger = logging.getLogger(f'finetune_{intensity_key}')
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 文件handler
    fh = logging.FileHandler(LOGS_DIR / f'step2_finetune_{intensity_key}.log')
    fh.setLevel(logging.INFO)
    
    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_qa_data(qa_file):
    """加载QA数据"""
    qa_pairs = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))
    return qa_pairs


def prepare_training_data(qa_pairs, tokenizer):
    """
    准备训练数据
    将QA对转换为对话格式
    """
    formatted_data = []
    
    for qa in qa_pairs:
        # 使用攻击者模型的答案作为训练目标
        messages = [
            {"role": "user", "content": qa['question']},
            {"role": "assistant", "content": qa['attacker_answer']}
        ]
        
        formatted_data.append({
            'messages': messages,
            'qa_id': qa['qa_id']
        })
    
    return formatted_data


def tokenize_function(examples, tokenizer, max_length=512):
    """tokenize函数"""
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for messages in examples['messages']:
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors=None
        )
        
        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])
        labels_list.append(encoded['input_ids'].copy())  # labels与input_ids相同
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list
    }


def finetune_single_intensity(intensity_key, logger):
    """
    对单个攻击强度进行微调
    
    Args:
        intensity_key: 攻击强度键 ('low', 'medium', 'high')
        logger: 日志记录器
    """
    config = ATTACK_INTENSITIES[intensity_key]
    
    logger.info("="*70)
    logger.info(f"Knowledge-Overwriting Attack - {config['name']}")
    logger.info("="*70)
    logger.info(f"攻击强度: {config['name']}")
    logger.info(f"训练QA对数: {config['num_train_qa']}")
    logger.info(f"目标模型: {TARGET_MODEL_NAME}")
    logger.info(f"攻击后模型: {get_attacked_model_name(intensity_key)}")
    logger.info("="*70)
    
    # 1. 加载训练数据
    train_qa_file = get_train_qa_file(intensity_key)
    if not train_qa_file.exists():
        logger.error(f"训练集QA文件不存在: {train_qa_file}")
        logger.error("请先运行 step1_generate_qa_pairs.py")
        return False
    
    qa_pairs = load_qa_data(train_qa_file)
    logger.info(f"加载了 {len(qa_pairs)} 个训练QA对")
    
    # 2. 加载目标模型和tokenizer
    logger.info(f"\n加载目标模型: {TARGET_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logger.info("模型加载完成")
    
    # 3. 配置LoRA
    logger.info("\n配置LoRA...")
    
    # 启用梯度检查点
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # 添加forward hook来启用梯度
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=FINETUNE_CONFIG['lora_r'],
        lora_alpha=FINETUNE_CONFIG['lora_alpha'],
        lora_dropout=FINETUNE_CONFIG['lora_dropout'],
        target_modules=FINETUNE_CONFIG['lora_target_modules'],
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 准备数据集
    logger.info("\n准备训练数据集...")
    formatted_data = prepare_training_data(qa_pairs, tokenizer)
    dataset = Dataset.from_list(formatted_data)
    
    # tokenize数据集
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, FINETUNE_CONFIG['max_length']),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # 划分训练集和验证集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=RANDOM_SEED)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(eval_dataset)}")
    
    # 5. 配置训练参数
    output_dir = get_attacked_model_path(intensity_key)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=FINETUNE_CONFIG['num_epochs'],
        per_device_train_batch_size=FINETUNE_CONFIG['batch_size'],
        per_device_eval_batch_size=FINETUNE_CONFIG['batch_size'],
        gradient_accumulation_steps=FINETUNE_CONFIG['gradient_accumulation_steps'],
        learning_rate=FINETUNE_CONFIG['learning_rate'],
        warmup_ratio=FINETUNE_CONFIG['warmup_ratio'],
        logging_steps=FINETUNE_CONFIG['logging_steps'],
        save_steps=FINETUNE_CONFIG['save_steps'],
        eval_steps=FINETUNE_CONFIG['eval_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        logging_dir=str(LOGS_DIR / f"training_logs_{intensity_key}")
    )
    
    # 6. 创建Trainer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    # 7. 开始训练
    logger.info("\n开始训练...")
    trainer.train()
    
    # 8. 保存模型
    logger.info(f"\n保存模型到: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # 9. 保存训练摘要
    summary = {
        'attack_intensity': intensity_key,
        'attack_intensity_name': config['name'],
        'target_model': TARGET_MODEL_NAME,
        'attacked_model': get_attacked_model_name(intensity_key),
        'attacker_model': ATTACKER_MODEL_NAME,
        'num_training_samples': len(train_dataset),
        'num_eval_samples': len(eval_dataset),
        'num_train_qa': config['num_train_qa'],
        'num_epochs': FINETUNE_CONFIG['num_epochs'],
        'learning_rate': FINETUNE_CONFIG['learning_rate'],
        'lora_r': FINETUNE_CONFIG['lora_r'],
        'lora_alpha': FINETUNE_CONFIG['lora_alpha'],
        'description': config['description']
    }
    
    summary_file = output_dir / "training_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练摘要已保存到: {summary_file}")
    
    logger.info("\n" + "="*70)
    logger.info(f"{config['name']} 训练完成！")
    logger.info(f"攻击后模型已保存到: {output_dir}")
    logger.info("="*70)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Finetune models with different attack intensities')
    parser.add_argument(
        '--intensity',
        type=str,
        choices=['low', 'medium', 'high', 'all'],
        default='all',
        help='Attack intensity to finetune (default: all)'
    )
    args = parser.parse_args()
    
    # 确定要训练的强度
    if args.intensity == 'all':
        intensities_to_train = list(ATTACK_INTENSITIES.keys())
    else:
        intensities_to_train = [args.intensity]
    
    print("="*70)
    print("Knowledge-Overwriting Attack - Step 2: Finetune Models")
    print("="*70)
    print(f"将要训练的攻击强度: {', '.join([ATTACK_INTENSITIES[k]['name'] for k in intensities_to_train])}")
    print("="*70)
    
    # 对每个强度进行训练
    results = {}
    for intensity_key in intensities_to_train:
        logger = setup_logger(intensity_key)
        success = finetune_single_intensity(intensity_key, logger)
        results[intensity_key] = success
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        
        print("\n" + "="*70)
        print(f"{ATTACK_INTENSITIES[intensity_key]['name']} 完成: {'成功' if success else '失败'}")
        print("="*70 + "\n")
    
    # 打印总结
    print("\n" + "="*70)
    print("所有训练任务完成")
    print("="*70)
    for intensity_key, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{ATTACK_INTENSITIES[intensity_key]['name']}: {status}")
    print("="*70)


if __name__ == "__main__":
    main()

