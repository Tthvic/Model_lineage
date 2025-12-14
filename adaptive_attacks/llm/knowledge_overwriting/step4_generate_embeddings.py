#!/usr/bin/env python3
"""
Step 4: Generate embeddings for QA texts

使用Qwen2.5-1.5B-Instruct编码器将"Question + Answer"文本编码为embeddings
"""

import os
import json
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
import logging
from pathlib import Path
from tqdm import tqdm
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step4_generate_embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_answers(answers_file):
    """加载答案文件"""
    if not answers_file.exists():
        logger.error(f"答案文件不存在: {answers_file}")
        return None
    
    answers = []
    with open(answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            answers.append(json.loads(line.strip()))
    
    logger.info(f"加载了 {len(answers)} 个答案")
    return answers


def encode_qa_texts(encoder_model, tokenizer, answers, model_name):
    """
    编码QA文本
    
    Args:
        encoder_model: 编码器模型
        tokenizer: tokenizer
        answers: 答案列表
        model_name: 模型名称（用于日志）
    
    Returns:
        embeddings: [num_qa, embedding_dim]
    """
    logger.info(f"编码 {model_name} 的QA文本...")
    
    embeddings_list = []
    
    for answer in tqdm(answers, desc=f"编码 ({model_name})"):
        try:
            # 组合问题和答案
            qa_text = f"Question: {answer['question']}\nAnswer: {answer['answer']}"
            
            # Tokenize
            inputs = tokenizer(
                qa_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(encoder_model.device) for k, v in inputs.items()}
            
            # 编码
            with torch.no_grad():
                outputs = encoder_model(**inputs)
                # 使用最后一层的[CLS] token或平均池化
                # 这里使用平均池化
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                # 平均池化
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                
                embeddings_list.append(embedding.cpu())
        
        except Exception as e:
            logger.error(f"编码QA ID {answer['qa_id']} 时出错: {e}")
            # 添加零向量作为占位符
            embeddings_list.append(torch.zeros(1, encoder_model.config.hidden_size))
    
    # 拼接所有embeddings
    embeddings = torch.cat(embeddings_list, dim=0)
    logger.info(f"编码完成，embeddings shape: {embeddings.shape}")
    
    return embeddings


def generate_embeddings_for_model(encoder_model, tokenizer, model_type, intensity_key=None):
    """
    为指定模型生成embeddings
    
    Args:
        encoder_model: 编码器模型
        tokenizer: tokenizer
        model_type: 'target' 或 'attacked'
        intensity_key: 如果是attacked模型，指定强度
    """
    # 加载答案
    answers_file = get_answers_file(model_type, intensity_key)
    answers = load_answers(answers_file)
    
    if answers is None:
        return False
    
    # 确定模型名称
    if model_type == 'target':
        model_name = TARGET_MODEL_NAME
    else:
        model_name = get_attacked_model_name(intensity_key)
    
    logger.info("="*70)
    logger.info(f"生成embeddings: {model_name}")
    logger.info("="*70)
    
    # 编码QA文本
    embeddings = encode_qa_texts(encoder_model, tokenizer, answers, model_name)
    
    # 保存embeddings
    output_file = get_embeddings_file(model_type, intensity_key)
    torch.save(embeddings, output_file)
    
    logger.info(f"Embeddings已保存到: {output_file}\n")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate embeddings for QA texts')
    parser.add_argument(
        '--model',
        type=str,
        choices=['target', 'low', 'medium', 'high', 'all'],
        default='all',
        help='Which model to generate embeddings (default: all)'
    )
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Knowledge-Overwriting Attack - Step 4: Generate Embeddings")
    logger.info("="*70)
    logger.info(f"编码器模型: Qwen2.5-1.5B-Instruct")
    logger.info("="*70 + "\n")
    
    # 加载编码器模型
    logger.info(f"加载编码器模型: {BASE_ENCODER_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_ENCODER_MODEL,
        trust_remote_code=True
    )
    
    encoder_model = AutoModel.from_pretrained(
        BASE_ENCODER_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    encoder_model.eval()
    
    logger.info(f"编码器加载完成，设备: {next(encoder_model.parameters()).device}")
    logger.info(f"Embedding维度: {encoder_model.config.hidden_size}\n")
    
    # 生成embeddings
    results = {}
    
    if args.model == 'all' or args.model == 'target':
        results['target'] = generate_embeddings_for_model(encoder_model, tokenizer, 'target')
    
    if args.model == 'all':
        intensities_to_generate = list(ATTACK_INTENSITIES.keys())
    elif args.model in ATTACK_INTENSITIES:
        intensities_to_generate = [args.model]
    else:
        intensities_to_generate = []
    
    for intensity_key in intensities_to_generate:
        results[intensity_key] = generate_embeddings_for_model(
            encoder_model, tokenizer, 'attacked', intensity_key
        )
    
    # 打印总结
    logger.info("\n" + "="*70)
    logger.info("所有embeddings生成完成")
    logger.info("="*70)
    
    if 'target' in results:
        status = "✓ 成功" if results['target'] else "✗ 失败"
        logger.info(f"目标模型 ({TARGET_MODEL_NAME}): {status}")
    
    for intensity_key in intensities_to_generate:
        if intensity_key in results:
            status = "✓ 成功" if results[intensity_key] else "✗ 失败"
            logger.info(f"{ATTACK_INTENSITIES[intensity_key]['name']}: {status}")
    
    logger.info("="*70)


if __name__ == "__main__":
    main()

