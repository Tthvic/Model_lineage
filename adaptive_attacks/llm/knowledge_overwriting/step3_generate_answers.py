#!/usr/bin/env python3
"""
Step 3: Generate answers for test questions

自适应攻击场景：
让目标模型B和所有攻击后的模型C (Low/Medium/High) 回答所有200个问题
注意：这些问题中的部分已经用于微调（自适应攻击的特点）
"""

import os
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step3_generate_answers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_test_questions():
    """加载测试集问题（自适应攻击：使用所有200个QA对）"""
    test_qa_file = QA_DATA_DIR / "all_qa_pairs.jsonl"
    
    if not test_qa_file.exists():
        logger.error(f"测试集文件不存在: {test_qa_file}")
        return None
    
    questions = []
    with open(test_qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    
    logger.info(f"加载了 {len(questions)} 个测试问题（自适应攻击：完整的200个QA对）")
    return questions


def generate_answers(model, tokenizer, questions, model_name):
    """
    生成答案
    
    Args:
        model: 模型
        tokenizer: tokenizer
        questions: 问题列表
        model_name: 模型名称（用于日志）
    
    Returns:
        answers: 答案列表
    """
    logger.info(f"使用 {model_name} 生成答案...")
    
    answers = []
    
    for qa in tqdm(questions, desc=f"生成答案 ({model_name})"):
        try:
            # 构建对话
            messages = [
                {"role": "user", "content": qa['question']}
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
            
            # 保存答案
            answers.append({
                'qa_id': qa['qa_id'],
                'question': qa['question'],
                'answer': generated_answer,
                'model': model_name
            })
        
        except Exception as e:
            logger.error(f"生成答案时出错 (QA ID: {qa['qa_id']}): {e}")
            answers.append({
                'qa_id': qa['qa_id'],
                'question': qa['question'],
                'answer': "[ERROR]",
                'model': model_name,
                'error': str(e)
            })
    
    logger.info(f"成功生成 {len(answers)} 个答案")
    return answers


def save_answers(answers, output_file):
    """保存答案"""
    logger.info(f"保存答案到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')
    
    logger.info(f"答案已保存，共 {len(answers)} 条")


def generate_target_model_answers(questions):
    """生成目标模型B的答案"""
    logger.info("="*70)
    logger.info(f"生成目标模型答案: {TARGET_MODEL_NAME}")
    logger.info("="*70)
    
    # 加载模型
    logger.info(f"加载模型: {TARGET_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    logger.info(f"模型加载完成，设备: {next(model.parameters()).device}")
    
    # 生成答案
    answers = generate_answers(model, tokenizer, questions, TARGET_MODEL_NAME)
    
    # 保存答案
    output_file = get_answers_file('target')
    save_answers(answers, output_file)
    
    # 清理内存
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    logger.info(f"目标模型答案已保存到: {output_file}\n")
    return True


def generate_attacked_model_answers(intensity_key, questions):
    """生成攻击后模型C的答案"""
    config = ATTACK_INTENSITIES[intensity_key]
    model_name = get_attacked_model_name(intensity_key)
    model_path = get_attacked_model_path(intensity_key)
    
    logger.info("="*70)
    logger.info(f"生成攻击后模型答案: {model_name}")
    logger.info(f"攻击强度: {config['name']}")
    logger.info("="*70)
    
    # 检查模型是否存在
    if not model_path.exists():
        logger.error(f"模型不存在: {model_path}")
        logger.error("请先运行 step2_finetune_models.py")
        return False
    
    # 加载基础模型
    logger.info(f"加载基础模型: {TARGET_MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载LoRA权重
    logger.info(f"加载LoRA权重: {model_path}")
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model.eval()
    
    logger.info(f"模型加载完成，设备: {next(model.parameters()).device}")
    
    # 生成答案
    answers = generate_answers(model, tokenizer, questions, model_name)
    
    # 保存答案
    output_file = get_answers_file('attacked', intensity_key)
    save_answers(answers, output_file)
    
    # 清理内存
    del model
    del base_model
    del tokenizer
    torch.cuda.empty_cache()
    
    logger.info(f"攻击后模型答案已保存到: {output_file}\n")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate answers for test questions')
    parser.add_argument(
        '--model',
        type=str,
        choices=['target', 'low', 'medium', 'high', 'all'],
        default='all',
        help='Which model to generate answers (default: all)'
    )
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Knowledge-Overwriting Attack - Step 3: Generate Answers")
    logger.info("="*70)
    
    # 加载测试问题
    questions = load_test_questions()
    if questions is None:
        return
    
    logger.info(f"测试问题数: {len(questions)}")
    logger.info("="*70 + "\n")
    
    # 生成答案
    results = {}
    
    if args.model == 'all' or args.model == 'target':
        results['target'] = generate_target_model_answers(questions)
    
    if args.model == 'all':
        intensities_to_generate = list(ATTACK_INTENSITIES.keys())
    elif args.model in ATTACK_INTENSITIES:
        intensities_to_generate = [args.model]
    else:
        intensities_to_generate = []
    
    for intensity_key in intensities_to_generate:
        results[intensity_key] = generate_attacked_model_answers(intensity_key, questions)
    
    # 打印总结
    logger.info("\n" + "="*70)
    logger.info("所有答案生成完成")
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

