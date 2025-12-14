#!/usr/bin/env python3
"""
Step 1: Generate QA pairs using Llama-3.1-8B-Instruct (Attacker Model)

自适应攻击场景：
生成200个QA对，这些QA对既是训练池也是测试集。
攻击者知道测试集的问题，从中选取不同数量(20/60/100)进行微调。
测试时，所有模型都在完整的200个QA对上测试。
"""

import os
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import logging
from pathlib import Path
from tqdm import tqdm
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step1_generate_qa_pairs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_arc_challenge_data():
    """加载arc_challenge数据集"""
    logger.info(f"加载arc_challenge数据集: {ARC_DATASET_PATH}")
    
    # 尝试加载不同的split
    for split in ['test', 'validation', 'train']:
        try:
            dataset_path = Path(ARC_DATASET_PATH) / split
            if dataset_path.exists():
                dataset = load_from_disk(str(dataset_path))
                logger.info(f"成功加载 {split} split，包含 {len(dataset)} 条数据")
                return dataset
        except Exception as e:
            logger.debug(f"尝试加载 {split} split 失败: {e}")
    
    # 如果上面都失败，直接加载整个目录
    try:
        dataset = load_from_disk(str(ARC_DATASET_PATH))
        logger.info(f"成功加载数据集，包含 {len(dataset)} 条数据")
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise


def format_arc_question(item):
    """
    格式化arc_challenge问题为标准格式
    
    ARC数据格式：
    {
        'question': str,
        'choices': {
            'text': [str, str, ...],
            'label': [str, str, ...]
        },
        'answerKey': str
    }
    """
    question = item['question']
    choices = item['choices']
    
    # 构建选项文本
    options_text = "\n".join([
        f"{label}. {text}" 
        for label, text in zip(choices['label'], choices['text'])
    ])
    
    # 完整问题
    full_question = f"{question}\n\nOptions:\n{options_text}\n\nAnswer:"
    
    # 正确答案
    answer_key = item['answerKey']
    answer_idx = choices['label'].index(answer_key)
    correct_answer = f"{answer_key}. {choices['text'][answer_idx]}"
    
    return full_question, correct_answer


def generate_qa_with_llama(model, tokenizer, questions, num_pairs):
    """
    使用Llama模型生成QA对
    
    Args:
        model: Llama模型
        tokenizer: tokenizer
        questions: 问题列表
        num_pairs: 要生成的QA对数量
    
    Returns:
        qa_pairs: QA对列表
    """
    logger.info(f"使用{ATTACKER_MODEL_NAME}生成 {num_pairs} 个QA对")
    
    # 随机采样
    random.seed(RANDOM_SEED)
    selected_items = random.sample(list(questions), min(num_pairs, len(questions)))
    
    qa_pairs = []
    
    for idx, item in enumerate(tqdm(selected_items, desc="生成QA对")):
        try:
            # 格式化问题
            question, correct_answer = format_arc_question(item)
            
            # 构建对话
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer the multiple choice question by selecting the correct option."},
                {"role": "user", "content": question}
            ]
            
            # 应用聊天模板
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 生成答案
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 解码答案
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 保存QA对
            qa_pairs.append({
                'qa_id': idx,
                'question': question,
                'attacker_answer': generated_answer,  # 攻击者模型的答案
                'correct_answer': correct_answer,
                'original_question': item['question'],
                'answer_key': item['answerKey']
            })
            
            if (idx + 1) % 10 == 0:
                logger.info(f"已生成 {idx + 1}/{num_pairs} 个QA对")
        
        except Exception as e:
            logger.error(f"生成第 {idx} 个QA对时出错: {e}")
            continue
    
    logger.info(f"成功生成 {len(qa_pairs)} 个QA对")
    return qa_pairs


def save_qa_pairs(qa_pairs, output_file):
    """保存QA对为JSONL格式"""
    logger.info(f"保存QA对到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    logger.info(f"QA对已保存，共 {len(qa_pairs)} 条")


def split_qa_data(qa_pairs):
    """
    自适应攻击：为每个攻击强度创建训练集
    
    注意：这是自适应攻击场景，攻击者知道测试集。
    - 所有200个QA对都用于测试
    - 从这200个中选取不同数量(20/60/100)用于微调
    
    Args:
        qa_pairs: 所有QA对（200个）
    
    Returns:
        train_qa_splits: 各强度的训练集字典
    """
    random.seed(RANDOM_SEED)
    indices = list(range(len(qa_pairs)))
    random.shuffle(indices)
    
    logger.info(f"自适应攻击：所有 {len(qa_pairs)} 个QA对既是训练池也是测试集")
    
    # 为每个攻击强度创建训练集（从200个中选取）
    train_qa_splits = {}
    for intensity_key, config in ATTACK_INTENSITIES.items():
        num_train = config['num_train_qa']
        # 选取前num_train个作为该强度的训练集
        train_indices = indices[:num_train]
        train_qa_splits[intensity_key] = [qa_pairs[i] for i in sorted(train_indices)]
        logger.info(f"  - {config['name']}: {len(train_qa_splits[intensity_key])} 个QA对用于微调")
    
    return train_qa_splits


def main():
    """主函数"""
    logger.info("="*70)
    logger.info("Knowledge-Overwriting Attack - Step 1: Generate QA Pairs")
    logger.info("="*70)
    logger.info(f"攻击场景: 自适应攻击 (Adaptive Attack)")
    logger.info(f"攻击者模型: {ATTACKER_MODEL_NAME}")
    logger.info(f"生成QA对数: {TOTAL_QA_PAIRS}")
    logger.info(f"说明: 这200个QA对既是训练池也是测试集")
    logger.info("="*70)
    
    # 1. 加载数据集
    dataset = load_arc_challenge_data()
    
    # 2. 加载Llama模型
    logger.info(f"\n加载攻击者模型: {ATTACKER_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        ATTACKER_MODEL_PATH,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        ATTACKER_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    logger.info(f"模型加载完成，设备: {next(model.parameters()).device}")
    
    # 3. 生成QA对
    qa_pairs = generate_qa_with_llama(model, tokenizer, dataset, TOTAL_QA_PAIRS)
    
    # 4. 创建各强度的训练集
    logger.info("\n创建各强度的训练集（自适应攻击）...")
    train_qa_splits = split_qa_data(qa_pairs)
    
    # 5. 保存所有QA对（这也是测试集）
    all_qa_file = QA_DATA_DIR / "all_qa_pairs.jsonl"
    save_qa_pairs(qa_pairs, all_qa_file)
    logger.info(f"所有QA对已保存（也作为测试集）: {all_qa_file}")
    
    # 6. 保存各强度的训练集
    for intensity_key, train_qa in train_qa_splits.items():
        train_qa_file = get_train_qa_file(intensity_key)
        save_qa_pairs(train_qa, train_qa_file)
    
    # 7. 保存摘要信息
    summary = {
        'attack_type': 'Adaptive Attack (自适应攻击)',
        'attacker_model': ATTACKER_MODEL_NAME,
        'source_dataset': 'arc_challenge',
        'total_qa_pairs': len(qa_pairs),
        'description': '攻击者知道测试集，从200个QA对中选取不同数量进行微调，测试时在完整200个上测试',
        'attack_intensities': {
            key: {
                'name': config['name'],
                'num_train_qa': config['num_train_qa'],
                'percentage': config['percentage'],
                'train_file': str(get_train_qa_file(key))
            }
            for key, config in ATTACK_INTENSITIES.items()
        },
        'test_qa_file': str(all_qa_file),
        'random_seed': RANDOM_SEED
    }
    
    summary_file = QA_DATA_DIR / "qa_generation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n摘要信息已保存到: {summary_file}")
    
    # 8. 打印摘要
    logger.info("\n" + "="*70)
    logger.info("数据生成摘要 (自适应攻击)")
    logger.info("="*70)
    logger.info(f"总QA对数: {len(qa_pairs)}")
    logger.info(f"攻击场景: 自适应攻击 - 攻击者知道测试集")
    logger.info(f"测试集: 所有 {len(qa_pairs)} 个QA对")
    logger.info("\n攻击强度配置（从200个中选取）:")
    for intensity_key, config in ATTACK_INTENSITIES.items():
        logger.info(f"  - {config['name']}: {config['num_train_qa']} 个QA对 ({config['percentage']*100:.0f}%)")
    
    logger.info(f"\n测试集文件: {all_qa_file}")
    logger.info("训练集文件:")
    for intensity_key in ATTACK_INTENSITIES.keys():
        logger.info(f"  - {intensity_key}: {get_train_qa_file(intensity_key)}")
    
    logger.info("\n生成的QA对样例（前2个）:")
    logger.info("="*70)
    for i, qa in enumerate(qa_pairs[:2]):
        logger.info(f"\nQA对 #{i+1}:")
        logger.info(f"问题: {qa['original_question']}")
        logger.info(f"攻击者答案: {qa['attacker_answer'][:100]}...")
        logger.info(f"正确答案: {qa['correct_answer']}")
    
    logger.info("\n" + "="*70)
    logger.info("Step 1 完成！")
    logger.info("="*70)


if __name__ == "__main__":
    main()

