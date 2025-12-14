#!/usr/bin/env python3
"""
Step 5: Compute B-A Difference Embeddings
Same as step 4：Load from generated embeddings files and compute difference
"""

import os
import json
import torch
import logging
from pathlib import Path
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step4_compute_ba_diff.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    主函数：Compute B-A Difference Embeddings
    直接从step3_5生成的embeddings文件中读取并计算差异
    """
    
    # Create output directory
    ba_dir = EXPERIMENT_DIR / "ba_embeddings"
    ba_dir.mkdir(exist_ok=True)
    
    # Embeddings directory
    embeddings_dir = EXPERIMENT_DIR / "embeddings"
    
    # Load embeddings for model A
    a_emb_file = embeddings_dir / "A_embeddings.pt"
    if not a_emb_file.exists():
        logger.error(f"A的embeddings文件不存在: {a_emb_file}")
        logger.error("请先运行 step3_generate_answers.py 和 step3_5_generate_embeddings.py")
        return
    
    logger.info(f"Load embeddings for model A: {a_emb_file}")
    a_embeddings = torch.load(a_emb_file)
    logger.info(f"A embeddings shape: {a_embeddings.shape}")
    logger.info(f"A embeddings mean: {a_embeddings.mean():.6f}")
    logger.info(f"A embeddings std: {a_embeddings.std():.6f}")
    
    # Process B1, B2, B3
    b_models = ['B1', 'B2', 'B3']
    ba_info = {}
    
    for model_name in b_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理模型: {model_name}")
        logger.info(f"{'='*60}")
        
        # Load embeddings for model B
        b_emb_file = embeddings_dir / f"{model_name}_embeddings.pt"
        
        if not b_emb_file.exists():
            logger.warning(f"B的embeddings文件不存在: {b_emb_file}")
            logger.warning(f"跳过 {model_name}，请先运行 step3_5_generate_embeddings.py")
            continue
        
        try:
            logger.info(f"加载{model_name}的embeddings: {b_emb_file}")
            b_embeddings = torch.load(b_emb_file)
            logger.info(f"{model_name} embeddings shape: {b_embeddings.shape}")
            logger.info(f"{model_name} embeddings mean: {b_embeddings.mean():.6f}")
            logger.info(f"{model_name} embeddings std: {b_embeddings.std():.6f}")
            
            # Compute B-A difference
            ba_diff = b_embeddings - a_embeddings
            
            logger.info(f"{model_name}-A差异 shape: {ba_diff.shape}")
            logger.info(f"{model_name}-A差异 mean: {ba_diff.mean():.6f}")
            logger.info(f"{model_name}-A差异 std: {ba_diff.std():.6f}")
            
            # Save B-A difference
            output_file = ba_dir / f"{model_name}_minus_A.pt"
            torch.save(ba_diff, output_file)
            
            ba_info[model_name] = {
                'output_file': str(output_file),
                'shape': list(ba_diff.shape),
                'mean': float(ba_diff.mean()),
                'std': float(ba_diff.std())
            }
            
            logger.info(f"B-A差异已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"处理 {model_name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Save info
    info_file = ba_dir / "ba_diff_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(ba_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("All B-A differences computed！")
    logger.info(f"{'='*60}")
    logger.info(f"Difference info saved to: {info_file}")
    
    # 打印摘要
    print("\nB-A Difference Summary:")
    for model_name, info in ba_info.items():
        print(f"\n{model_name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Mean: {info['mean']:.6f}")
        print(f"  Std: {info['std']:.6f}")
        print(f"  文件: {info['output_file']}")


if __name__ == "__main__":
    main()
