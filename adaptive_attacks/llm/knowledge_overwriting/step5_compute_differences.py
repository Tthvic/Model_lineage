#!/usr/bin/env python3
"""
Step 5: Compute embedding differences (C - B)

计算攻击后模型C与目标模型B的embeddings差异
"""

import os
import torch
import argparse
import logging
from pathlib import Path
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step5_compute_differences.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_difference(intensity_key):
    """
    计算指定强度的C-B差异
    
    Args:
        intensity_key: 攻击强度键
    """
    config = ATTACK_INTENSITIES[intensity_key]
    model_name = get_attacked_model_name(intensity_key)
    
    logger.info("="*70)
    logger.info(f"计算差异: {config['name']}")
    logger.info("="*70)
    logger.info(f"目标模型B: {TARGET_MODEL_NAME}")
    logger.info(f"攻击后模型C: {model_name}")
    logger.info("="*70)
    
    # 加载目标模型B的embeddings
    emb_b_file = get_embeddings_file('target')
    if not emb_b_file.exists():
        logger.error(f"目标模型embeddings不存在: {emb_b_file}")
        logger.error("请先运行 step4_generate_embeddings.py")
        return False
    
    emb_b = torch.load(emb_b_file)
    logger.info(f"目标模型B embeddings shape: {emb_b.shape}")
    
    # 加载攻击后模型C的embeddings
    emb_c_file = get_embeddings_file('attacked', intensity_key)
    if not emb_c_file.exists():
        logger.error(f"攻击后模型embeddings不存在: {emb_c_file}")
        logger.error("请先运行 step4_generate_embeddings.py")
        return False
    
    emb_c = torch.load(emb_c_file)
    logger.info(f"攻击后模型C embeddings shape: {emb_c.shape}")
    
    # 检查形状是否匹配
    if emb_b.shape != emb_c.shape:
        logger.error(f"Embeddings形状不匹配: B={emb_b.shape}, C={emb_c.shape}")
        return False
    
    # 计算差异
    emb_diff = emb_c - emb_b
    logger.info(f"差异向量 shape: {emb_diff.shape}")
    
    # 统计信息
    diff_norm = torch.norm(emb_diff, dim=1)
    logger.info(f"差异向量范数统计:")
    logger.info(f"  - 平均值: {diff_norm.mean().item():.6f}")
    logger.info(f"  - 标准差: {diff_norm.std().item():.6f}")
    logger.info(f"  - 最小值: {diff_norm.min().item():.6f}")
    logger.info(f"  - 最大值: {diff_norm.max().item():.6f}")
    
    # 保存差异向量
    output_file = get_diff_embeddings_file(intensity_key)
    torch.save(emb_diff, output_file)
    
    logger.info(f"\n差异向量已保存到: {output_file}")
    logger.info("="*70 + "\n")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Compute embedding differences')
    parser.add_argument(
        '--intensity',
        type=str,
        choices=['low', 'medium', 'high', 'all'],
        default='all',
        help='Attack intensity to compute (default: all)'
    )
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Knowledge-Overwriting Attack - Step 5: Compute Differences")
    logger.info("="*70 + "\n")
    
    # 确定要计算的强度
    if args.intensity == 'all':
        intensities_to_compute = list(ATTACK_INTENSITIES.keys())
    else:
        intensities_to_compute = [args.intensity]
    
    # 计算差异
    results = {}
    for intensity_key in intensities_to_compute:
        results[intensity_key] = compute_difference(intensity_key)
    
    # 打印总结
    logger.info("\n" + "="*70)
    logger.info("所有差异计算完成")
    logger.info("="*70)
    
    for intensity_key, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"{ATTACK_INTENSITIES[intensity_key]['name']}: {status}")
    
    logger.info("="*70)


if __name__ == "__main__":
    main()

