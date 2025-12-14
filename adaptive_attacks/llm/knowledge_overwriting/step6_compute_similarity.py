#!/usr/bin/env python3
"""
Step 6: Compute lineage similarity between C and B

计算攻击后模型C与目标模型B的血缘相似度

血缘相似度计算公式：
similarity = cosine(enc(emb_b), RelationNet(enc(emb_c), enc(emb_diff)))
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from pathlib import Path
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step6_compute_similarity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """特征编码器：将1536维特征编码为512维向量"""
    def __init__(self, feat_dim=1536, d_model=512, kernel_size=3, dropout=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.attn_pooling = nn.AdaptiveAvgPool1d(1)

    def compute_valid_lengths(self, x):
        mask = (x.sum(dim=-1) != 0)
        valid_lengths = mask.int().sum(dim=1)
        return valid_lengths

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        valid_lengths = self.compute_valid_lengths(x)
        
        # 特征投影 + 归一化
        feat_proj = self.layer_norm(self.feat_proj(x))
        
        # 转置维度以适应 Conv1d
        feat_proj = feat_proj.permute(0, 2, 1)
        conv_out = self.dropout(self.conv(feat_proj))
        
        # 全局表示
        global_vec = self.attn_pooling(conv_out).squeeze(-1)
        
        return global_vec


class VectorRelationNet(nn.Module):
    """关系网络：学习两个向量之间的关系"""
    def __init__(self, embedding_dim=512):
        super(VectorRelationNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
        
    def forward(self, c_emb, diff_emb):
        h1 = self.fc1(torch.cat([c_emb, diff_emb], dim=-1))
        h1 = self.relu(h1)
        return h1


def compute_lineage_similarity(encnet, prenet, emb_b, emb_c, emb_diff, device):
    """
    计算血缘相似度
    
    Args:
        encnet: 编码器网络
        prenet: 关系网络
        emb_b: 目标模型B的嵌入 [batch, 1536]
        emb_c: 攻击后模型C的嵌入 [batch, 1536]
        emb_diff: C-B差异嵌入 [batch, 1536]
        device: 计算设备
    
    Returns:
        avg_similarity: 平均血缘相似度
        similarities_per_question: 每个问题的相似度
    """
    # 移动到设备
    emb_b = emb_b.to(device)
    emb_c = emb_c.to(device)
    emb_diff = emb_diff.to(device)
    
    # 添加seq维度 [batch, 1, 1536]
    emb_b = emb_b.unsqueeze(1)
    emb_c = emb_c.unsqueeze(1)
    emb_diff = emb_diff.unsqueeze(1)
    
    with torch.no_grad():
        # 编码特征
        enc_b = encnet(emb_b)      # [batch, 512]
        enc_c = encnet(emb_c)      # [batch, 512]
        enc_diff = encnet(emb_diff)    # [batch, 512]
        
        # 计算关系表示
        relation = prenet(enc_c, enc_diff)  # [batch, 512]
        
        # 计算相似度：enc_b与relation的余弦相似度
        similarities = F.cosine_similarity(enc_b, relation, dim=1)  # [batch]
        
        # 平均相似度
        avg_similarity = similarities.mean().item()
        similarities_list = similarities.cpu().numpy().tolist()
    
    return avg_similarity, similarities_list


def compute_similarity_for_intensity(encnet, prenet, intensity_key, device):
    """
    计算指定强度的血缘相似度
    
    Args:
        encnet: 编码器网络
        prenet: 关系网络
        intensity_key: 攻击强度键
        device: 计算设备
    """
    config = ATTACK_INTENSITIES[intensity_key]
    model_name = get_attacked_model_name(intensity_key)
    
    logger.info("="*70)
    logger.info(f"计算血缘相似度: {config['name']}")
    logger.info("="*70)
    logger.info(f"目标模型B: {TARGET_MODEL_NAME}")
    logger.info(f"攻击后模型C: {model_name}")
    logger.info("="*70)
    
    # 加载embeddings
    emb_b_file = get_embeddings_file('target')
    emb_c_file = get_embeddings_file('attacked', intensity_key)
    emb_diff_file = get_diff_embeddings_file(intensity_key)
    
    # 检查文件是否存在
    if not emb_b_file.exists():
        logger.error(f"目标模型embeddings不存在: {emb_b_file}")
        return None
    
    if not emb_c_file.exists():
        logger.error(f"攻击后模型embeddings不存在: {emb_c_file}")
        return None
    
    if not emb_diff_file.exists():
        logger.error(f"差异embeddings不存在: {emb_diff_file}")
        return None
    
    # 加载embeddings
    emb_b = torch.load(emb_b_file)
    emb_c = torch.load(emb_c_file)
    emb_diff = torch.load(emb_diff_file)
    
    logger.info(f"目标模型B embeddings shape: {emb_b.shape}")
    logger.info(f"攻击后模型C embeddings shape: {emb_c.shape}")
    logger.info(f"差异embeddings shape: {emb_diff.shape}")
    
    # 计算相似度
    logger.info("\n计算血缘相似度...")
    avg_sim, sims_per_q = compute_lineage_similarity(
        encnet, prenet, emb_b, emb_c, emb_diff, device
    )
    
    # 统计信息
    sims_array = np.array(sims_per_q)
    
    # 计算TPR (True Positive Rate) - 相似度超过阈值的比例
    threshold = 0.3
    above_threshold = (sims_array >= threshold).sum()
    tpr = above_threshold / len(sims_array)
    
    results = {
        'attack_intensity': intensity_key,
        'attack_intensity_name': config['name'],
        'target_model': TARGET_MODEL_NAME,
        'attacked_model': model_name,
        'attacker_model': ATTACKER_MODEL_NAME,
        'num_train_qa': config['num_train_qa'],
        'num_test_questions': len(sims_per_q),
        'avg_similarity': float(avg_sim),
        'std_similarity': float(np.std(sims_array)),
        'min_similarity': float(np.min(sims_array)),
        'max_similarity': float(np.max(sims_array)),
        'median_similarity': float(np.median(sims_array)),
        'threshold': threshold,
        'num_above_threshold': int(above_threshold),
        'tpr': float(tpr),
        'similarities_per_question': sims_per_q
    }
    
    # 保存结果
    results_file = get_results_file(intensity_key)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果已保存到: {results_file}")
    
    # 打印结果
    logger.info("\n" + "-"*70)
    logger.info("血缘相似度计算结果")
    logger.info("-"*70)
    logger.info(f"平均相似度: {results['avg_similarity']:.6f}")
    logger.info(f"标准差: {results['std_similarity']:.6f}")
    logger.info(f"中位数: {results['median_similarity']:.6f}")
    logger.info(f"最小值: {results['min_similarity']:.6f}")
    logger.info(f"最大值: {results['max_similarity']:.6f}")
    logger.info("-"*70)
    logger.info(f"TPR (阈值={threshold}): {results['tpr']:.4f} ({results['num_above_threshold']}/{results['num_test_questions']})")
    logger.info("="*70 + "\n")
    
    return results


def generate_summary_report(all_results):
    """生成汇总报告"""
    report_file = RESULTS_DIR / "knowledge_overwriting_attack_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Knowledge-Overwriting Attack Experiment Report\n")
        f.write("="*70 + "\n\n")
        
        f.write("实验设计:\n")
        f.write(f"  攻击者模型 (父模型A): {ATTACKER_MODEL_NAME}\n")
        f.write(f"  目标模型 (子模型B): {TARGET_MODEL_NAME}\n")
        f.write(f"  编码器模型: Qwen2.5-1.5B-Instruct\n")
        f.write(f"  测试问题数: {all_results[list(all_results.keys())[0]]['num_test_questions']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("攻击强度与血缘相似度结果:\n")
        f.write("-"*70 + "\n\n")
        
        for intensity_key in ['low', 'medium', 'high']:
            if intensity_key not in all_results:
                continue
            
            results = all_results[intensity_key]
            f.write(f"{results['attack_intensity_name']}:\n")
            f.write(f"  训练QA对数: {results['num_train_qa']}\n")
            f.write(f"  攻击后模型: {results['attacked_model']}\n")
            f.write(f"  平均相似度: {results['avg_similarity']:.6f}\n")
            f.write(f"  标准差: {results['std_similarity']:.6f}\n")
            f.write(f"  中位数: {results['median_similarity']:.6f}\n")
            f.write(f"  TPR (阈值={results['threshold']}): {results['tpr']:.4f}\n")
            
            if results['avg_similarity'] > 0.4:
                f.write(f"  结论: ✓ 相似度 {results['avg_similarity']:.4f} > 0.4，成功检测到血缘关系\n")
            else:
                f.write(f"  结论: ✗ 相似度 {results['avg_similarity']:.4f} < 0.4，未达到预期阈值\n")
            
            f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("总结:\n")
        f.write("-"*70 + "\n")
        f.write("本实验验证了血缘相似度计算方法在知识覆写攻击场景下的有效性。\n")
        f.write("通过三种不同强度的攻击（10%/30%/50%），观察血缘相似度的变化趋势。\n")
        f.write("\n" + "="*70 + "\n")
    
    logger.info(f"汇总报告已保存到: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Compute lineage similarity')
    parser.add_argument(
        '--intensity',
        type=str,
        choices=['low', 'medium', 'high', 'all'],
        default='all',
        help='Attack intensity to compute (default: all)'
    )
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Knowledge-Overwriting Attack - Step 6: Compute Similarity")
    logger.info("="*70 + "\n")
    
    # 检查关系模型
    if not Path(RELATION_MODEL_PATH).exists():
        logger.error(f"关系模型不存在: {RELATION_MODEL_PATH}")
        return
    
    # 加载关系网络
    logger.info(f"加载关系网络: {RELATION_MODEL_PATH}")
    checkpoint = torch.load(RELATION_MODEL_PATH, map_location='cpu', weights_only=False)
    
    encnet = TransformerEncoder()
    prenet = VectorRelationNet()
    
    encnet.load_state_dict(checkpoint['encnet_state_dict'])
    prenet.load_state_dict(checkpoint['prenet_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encnet = encnet.to(device)
    prenet = prenet.to(device)
    
    encnet.eval()
    prenet.eval()
    
    logger.info(f"关系网络加载成功，使用设备: {device}\n")
    
    # 确定要计算的强度
    if args.intensity == 'all':
        intensities_to_compute = list(ATTACK_INTENSITIES.keys())
    else:
        intensities_to_compute = [args.intensity]
    
    # 计算相似度
    all_results = {}
    for intensity_key in intensities_to_compute:
        results = compute_similarity_for_intensity(encnet, prenet, intensity_key, device)
        if results is not None:
            all_results[intensity_key] = results
    
    # 生成汇总报告
    if all_results:
        logger.info("\n生成汇总报告...")
        generate_summary_report(all_results)
    
    # 打印总结
    logger.info("\n" + "="*70)
    logger.info("所有相似度计算完成")
    logger.info("="*70)
    
    for intensity_key in intensities_to_compute:
        if intensity_key in all_results:
            results = all_results[intensity_key]
            logger.info(f"{results['attack_intensity_name']}: 平均相似度 = {results['avg_similarity']:.6f}")
        else:
            logger.info(f"{ATTACK_INTENSITIES[intensity_key]['name']}: ✗ 失败")
    
    logger.info("="*70)


if __name__ == "__main__":
    main()

