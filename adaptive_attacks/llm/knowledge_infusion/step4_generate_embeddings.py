#!/usr/bin/env python3
"""
Step 4: Encode "Question + Answer" text using Qwen-Instruct AutoModel
Completely consistent with original method：Using AutoModel + attention mask pooling
"""

import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
import logging
from pathlib import Path
from tqdm import tqdm
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step3_5_generate_embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_qa_answers(answers_file):
    """从JSONL文件Load questions and answers from file"""
    qa_pairs = []
    with open(answers_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            qa_pairs.append({
                'question': data['question'],
                'answer': data['answer'],
                'question_index': data['question_index']
            })
    return qa_pairs


def get_embeddings_from_qa_text(qa_pairs, model, tokenizer, device='cuda'):
    """
    使用AutoModel对"Question: {q}\nAnswer: {a}"文本进行编码
    与原始方法get_embeddings_from_answers完全一致
    """
    logger.info(f"生成 {len(qa_pairs)} 个QA对的embeddings")
    
    embeddings = []
    
    with torch.no_grad():
        for qa in tqdm(qa_pairs, desc="Generate embeddings"):
            # Combine question and answer as input text（与原始方法一致）
            text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
            
            # tokenize
            inputs = tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            # 生成embedding
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
            
            # Average pooling using attention mask（与原始方法一致）
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.shape)
            sum_hidden = (hidden_states * attention_mask).sum(dim=1)  # [1, hidden_dim]
            sum_mask = attention_mask.sum(dim=1) + 1e-8  # [1, hidden_dim]
            embedding = sum_hidden / sum_mask  # [1, hidden_dim]
            
            embedding = embedding.squeeze(0)  # [hidden_dim]
            embeddings.append(embedding.cpu())
    
    # 堆叠为 [num_items, hidden_dim]
    embeddings_tensor = torch.stack(embeddings)
    
    logger.info(f"Embeddings shape: {embeddings_tensor.shape}")
    logger.info(f"Embeddings mean: {embeddings_tensor.mean():.6f}")
    logger.info(f"Embeddings std: {embeddings_tensor.std():.6f}")
    
    return embeddings_tensor


def main():
    """Main function：Generate embeddings for all model answers"""
    
    # Create embeddings output directory
    embeddings_dir = EXPERIMENT_DIR / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    # Answers directory
    answers_dir = EXPERIMENT_DIR / "answers"
    
    # 加载Qwen-Instruct的AutoModel（与原始方法一致）
    logger.info("Load Qwen-Instruct AutoModel for encoding...")
    logger.info(f"模型路径: {BASE_INSTRUCT_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_INSTRUCT_MODEL,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用AutoModel（不是AutoModelForCausalLM！）
    model = AutoModel.from_pretrained(
        BASE_INSTRUCT_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    device = next(model.parameters()).device
    logger.info(f"模型加载到设备: {device}")
    
    # Process each model's answers
    model_names = ['A', 'B1', 'B2', 'B3']
    embeddings_info = {}
    
    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理模型: {model_name}")
        logger.info(f"{'='*60}")
        
        # Answer file path
        answers_file = answers_dir / f"model_{model_name}_answers.jsonl"
        
        if not answers_file.exists():
            logger.warning(f"答案文件不存在: {answers_file}")
            logger.warning(f"跳过 {model_name}，请先运行 step3_generate_answers.py")
            continue
        
        try:
            # Load QA pairs
            qa_pairs = load_qa_answers(answers_file)
            logger.info(f"加载了 {len(qa_pairs)} 个QA对")
            
            # Generate embeddings
            embeddings = get_embeddings_from_qa_text(qa_pairs, model, tokenizer, device)
            
            # Save embeddings
            output_file = embeddings_dir / f"{model_name}_embeddings.pt"
            torch.save(embeddings, output_file)
            
            embeddings_info[model_name] = {
                'output_file': str(output_file),
                'shape': list(embeddings.shape),
                'mean': float(embeddings.mean()),
                'std': float(embeddings.std()),
                'num_qa_pairs': len(qa_pairs)
            }
            
            logger.info(f"Embeddings已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"处理 {model_name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Save embeddings信息
    info_file = embeddings_dir / "embeddings_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("所有embeddings生成完成！")
    logger.info(f"{'='*60}")
    logger.info(f"Embeddings信息保存到: {info_file}")
    
    # Print summary
    print("\nEmbeddings生成摘要:")
    for model_name, info in embeddings_info.items():
        print(f"\n{model_name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Mean: {info['mean']:.6f}")
        print(f"  Std: {info['std']:.6f}")
        print(f"  QA对数: {info['num_qa_pairs']}")
        print(f"  文件: {info['output_file']}")


if __name__ == "__main__":
    main()
