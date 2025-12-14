#!/usr/bin/env python3
"""
Step 3: Generate Answers from Models A, B1, B2, B3 for 30 Questions
Consistent with original workflow: Generate JSONL answer files first
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from pathlib import Path
from tqdm import tqdm
from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step3_generate_answers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path, is_lora=False, base_model_path=None):
    """Load model and tokenizer"""
    logger.info(f"加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_lora else base_model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_lora and base_model_path:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    return model, tokenizer


def load_qa_questions(qa_file):
    """Load QA questions from file"""
    questions = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            questions.append(data['messages'][0]['content'])  # Extract question
    return questions


def generate_answers_for_model(model, tokenizer, questions, model_name, output_file):
    """
    Generate answers for a model and save as JSONL format (consistent with original method)
    """
    logger.info(f"为 {model_name} 生成答案，共 {len(questions)} 个问题")
    
    answers_data = []
    
    with torch.no_grad():
        for idx, question in enumerate(tqdm(questions, desc=f"Generating {model_name} answers")):
            # Build message
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Encode
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
            
            # Decode answer
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Save as JSONL format
            answers_data.append({
                'question_index': idx,
                'question': question,
                'answer': answer,
                'model': model_name
            })
    
    # Save JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in answers_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"答案已保存到: {output_file}")
    return answers_data


def main():
    """Main function: Generate answers for all models"""
    
    # Create answers output directory
    answers_dir = EXPERIMENT_DIR / "answers"
    answers_dir.mkdir(exist_ok=True)
    
    # Load 30 questions
    qa_file = DATA_DIR / "qa_full_30.jsonl"
    if not qa_file.exists():
        logger.error(f"QA文件不存在: {qa_file}")
        logger.error("请先运行 step1_split_data.py")
        return
    
    questions = load_qa_questions(qa_file)
    logger.info(f"加载了 {len(questions)} 个问题")
    
    # 模型配置
    models_config = {
        'A': {
            'path': MODEL_A_PATH,
            'is_lora': False,
            'output_name': 'model_A_answers.jsonl'
        },
        'B1': {
            'path': MODELS_DIR / 'B1',
            'is_lora': True,
            'base_path': MODEL_B_PATH,
            'output_name': 'model_B1_answers.jsonl'
        },
        'B2': {
            'path': MODELS_DIR / 'B2',
            'is_lora': True,
            'base_path': MODEL_B_PATH,
            'output_name': 'model_B2_answers.jsonl'
        },
        'B3': {
            'path': MODELS_DIR / 'B3',
            'is_lora': True,
            'base_path': MODEL_B_PATH,
            'output_name': 'model_B3_answers.jsonl'
        }
    }
    
    answers_info = {}
    
    for model_name, config in models_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"处理模型: {model_name}")
        logger.info(f"{'='*60}")
        
        # 检查模型路径
        if not Path(config['path']).exists():
            logger.warning(f"模型路径不存在: {config['path']}")
            if model_name != 'A':
                logger.warning(f"跳过 {model_name}，请先运行 step2_finetune_models.py")
                continue
            else:
                logger.error(f"模型A不存在，无法继续")
                return
        
        try:
            # 加载模型
            model, tokenizer = load_model_and_tokenizer(
                config['path'],
                is_lora=config.get('is_lora', False),
                base_model_path=config.get('base_path')
            )
            
            # 生成答案
            output_file = answers_dir / config['output_name']
            answers_data = generate_answers_for_model(
                model, tokenizer, questions, model_name, output_file
            )
            
            answers_info[model_name] = {
                'output_file': str(output_file),
                'num_questions': len(answers_data)
            }
            
            # 清理显存
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"处理 {model_name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 保存答案信息
    info_file = answers_dir / "answers_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(answers_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("所有答案生成完成！")
    logger.info(f"{'='*60}")
    logger.info(f"答案信息保存到: {info_file}")
    
    # 打印摘要
    print("\n答案生成摘要:")
    for model_name, info in answers_info.items():
        print(f"\n{model_name}:")
        print(f"  问题数: {info['num_questions']}")
        print(f"  文件: {info['output_file']}")


if __name__ == "__main__":
    main()
