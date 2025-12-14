#!/usr/bin/env python3
"""
Step 6: Compute Lineage Similarity
Use trained relation network to compute lineage similarity between A and B1, B2, B3
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from config import *

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step5_compute_similarity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup font for plotting
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class TransformerEncoder(nn.Module):
    """Feature encoder: Encode 1536-dim features to 512-dim vectors"""
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
    """Relation network: Learn relationship between two vectors"""
    def __init__(self, embedding_dim=512):
        super(VectorRelationNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
        
    def forward(self, b_afea, cfea):
        h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1 = self.relu(h1)
        return h1


def load_embeddings(embedding_file):
    """Load embedding file"""
    data = torch.load(embedding_file, map_location='cpu', weights_only=False)
    
    if isinstance(data, dict):
        if 'embeddings' in data:
            embeddings = data['embeddings']
        elif 'ba_embedding' in data:
            embeddings = data['ba_embedding']
        else:
            embeddings = next(iter(data.values()))
    else:
        embeddings = data
    
    # Ensure 2D tensor: [batch, 1536]
    if embeddings.dim() == 3:
        # If 3D, average pool to 2D
        embeddings = embeddings.mean(dim=1)
    elif embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    
    # Ensure last dim is 1536
    if embeddings.shape[-1] != 1536:
        logger.warning(f"嵌入维度不是1536: {embeddings.shape}")
        # Padding或截断到1536
        if embeddings.shape[-1] < 1536:
            padding = torch.zeros(embeddings.shape[0], 1536 - embeddings.shape[-1])
            embeddings = torch.cat([embeddings, padding], dim=-1)
        else:
            embeddings = embeddings[..., :1536]
    
    return embeddings


def compute_lineage_similarity(encnet, prenet, emb_a, emb_b_variant, emb_ba_diff, device):
    """
    Compute Lineage Similarity
    
    Args:
        encnet: 编码器网络
        prenet: 关系网络
        emb_a: 模型A的嵌入 [batch, 1536]
        emb_b_variant: B变体的嵌入 [batch, 1536]
        emb_ba_diff: B-A参数差异嵌入 [batch, 1536]
        device: 计算设备
    
    Returns:
        similarity: 血缘相似度值
        similarities_per_question: 每个问题的相似度
    """
    # 移动到设备
    emb_a = emb_a.to(device)
    emb_b_variant = emb_b_variant.to(device)
    emb_ba_diff = emb_ba_diff.to(device)
    
    # 添加seq维度以匹配编码器输入 [batch, 1, 1536]
    emb_a = emb_a.unsqueeze(1)
    emb_b_variant = emb_b_variant.unsqueeze(1)
    emb_ba_diff = emb_ba_diff.unsqueeze(1)
    
    with torch.no_grad():
        # Encode features
        enc_a = encnet(emb_a)                    # [batch, 512]
        enc_b_variant = encnet(emb_b_variant)    # [batch, 512]
        enc_ba = encnet(emb_ba_diff)             # [batch, 512]
        
        # Compute relation representation
        relation = prenet(enc_b_variant, enc_ba)  # [batch, 512]
        
        # Compute similarity
        similarities = F.cosine_similarity(enc_a, relation, dim=1)  # [batch]
        
        # Average similarity
        avg_similarity = similarities.mean().item()
        similarities_list = similarities.cpu().numpy().tolist()
    
    return avg_similarity, similarities_list


def visualize_results(results, output_dir):
    """Visualize lineage similarity results"""
    
    # Prepare data
    variants = list(results.keys())
    similarities = [results[v]['avg_similarity'] for v in variants]
    training_samples = [results[v]['training_samples'] for v in variants]
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 柱状图：Average similarity
    ax1 = axes[0]
    bars = ax1.bar(variants, similarities, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_ylabel('Lineage Similarity', fontsize=12)
    ax1.set_title('Average Lineage Similarity with Model A', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{sim:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. 折线图：相似度 vs 训练样本数
    ax2 = axes[1]
    ax2.plot(training_samples, similarities, marker='o', linewidth=2, markersize=10, color='#E74C3C')
    ax2.set_xlabel('Number of Training Samples', fontsize=12)
    ax2.set_ylabel('Lineage Similarity', fontsize=12)
    ax2.set_title('Similarity vs Training Data Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加数据点标签
    for x, y, label in zip(training_samples, similarities, variants):
        ax2.annotate(f'{label}\n({y:.3f})', 
                    xy=(x, y), 
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # 3. 箱线图：每个问题的相似度分布
    ax3 = axes[2]
    data_for_box = [results[v]['similarities_per_question'] for v in variants]
    bp = ax3.boxplot(data_for_box, labels=variants, patch_artist=True)
    
    # Beautify boxplot
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Lineage Similarity', fontsize=12)
    ax3.set_title('Similarity Distribution per Question', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / 'lineage_similarity_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"可视化结果保存到: {output_file}")
    plt.close()
    
    # Generate detailed heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare heatmap data：每个模型在每个问题上的相似度
    heatmap_data = np.array([results[v]['similarities_per_question'] for v in variants])
    
    sns.heatmap(heatmap_data, 
                xticklabels=[f'Q{i+1}' for i in range(heatmap_data.shape[1])],
                yticklabels=variants,
                cmap='RdYlGn',
                annot=False,
                fmt='.3f',
                cbar_kws={'label': 'Lineage Similarity'},
                ax=ax)
    
    ax.set_title('Lineage Similarity Heatmap\n(Each Model vs Model A for Each Question)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Question Index', fontsize=12)
    ax.set_ylabel('Model Variant', fontsize=12)
    
    plt.tight_layout()
    heatmap_file = output_dir / 'lineage_similarity_heatmap.png'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    logger.info(f"热力图保存到: {heatmap_file}")
    plt.close()


def generate_report(results, output_dir):
    """Generate text report"""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("血缘相似度实验报告")
    report_lines.append("Lineage Similarity Experiment Report")
    report_lines.append("="*70)
    report_lines.append("")
    
    report_lines.append("Experiment Setup:")
    report_lines.append(f"  基础模型 (Instruct): {BASE_INSTRUCT_MODEL}")
    report_lines.append(f"  模型A: {MODEL_A_NAME}")
    report_lines.append(f"  模型B: {MODEL_B_NAME}")
    report_lines.append(f"  测试问题数: 30")
    report_lines.append("")
    
    report_lines.append("-"*70)
    report_lines.append("Experiment Results:")
    report_lines.append("-"*70)
    
    for variant_name in ['B1', 'B2', 'B3']:
        if variant_name in results:
            result = results[variant_name]
            report_lines.append(f"\n{variant_name}:")
            report_lines.append(f"  训练数据: {result['training_data_range']}")
            report_lines.append(f"  训练样本数: {result['training_samples']}")
            report_lines.append(f"  与模型A的平均血缘相似度: {result['avg_similarity']:.6f}")
            report_lines.append(f"  相似度标准差: {result['std_similarity']:.6f}")
            report_lines.append(f"  最小相似度: {result['min_similarity']:.6f}")
            report_lines.append(f"  最大相似度: {result['max_similarity']:.6f}")
    
    report_lines.append("")
    report_lines.append("-"*70)
    report_lines.append("Analysis:")
    report_lines.append("-"*70)
    
    # Sorting
    sorted_variants = sorted(results.items(), key=lambda x: x[1]['avg_similarity'], reverse=True)
    
    report_lines.append(f"\n血缘相似度排名 (从高到低):")
    for rank, (variant, result) in enumerate(sorted_variants, 1):
        report_lines.append(f"  {rank}. {variant}: {result['avg_similarity']:.6f}")
    
    # 增长Analysis
    if 'B1' in results and 'B2' in results and 'B3' in results:
        sim_b1 = results['B1']['avg_similarity']
        sim_b2 = results['B2']['avg_similarity']
        sim_b3 = results['B3']['avg_similarity']
        
        report_lines.append(f"\n相似度增长Analysis:")
        report_lines.append(f"  B1 → B2: {sim_b2 - sim_b1:+.6f} ({(sim_b2/sim_b1 - 1)*100:+.2f}%)")
        report_lines.append(f"  B2 → B3: {sim_b3 - sim_b2:+.6f} ({(sim_b3/sim_b2 - 1)*100:+.2f}%)")
        report_lines.append(f"  B1 → B3: {sim_b3 - sim_b1:+.6f} ({(sim_b3/sim_b1 - 1)*100:+.2f}%)")
    
    report_lines.append("")
    report_lines.append("="*70)
    report_lines.append("Conclusion:")
    report_lines.append("="*70)
    
    if 'B1' in results and 'B2' in results and 'B3' in results:
        if sim_b1 < sim_b2 < sim_b3:
            report_lines.append("✓ Lineage similarity increases monotonically with training data size")
            report_lines.append("  This matches expectation：使用更多来自模型A的QA数据微调B，")
            report_lines.append("  会使B的变体更接近A的知识状态。")
        else:
            report_lines.append("✗ 血缘相似度未呈现单调递增趋势")
            report_lines.append("  Possible reasons：")
            report_lines.append("  1. 过拟合：使用更多数据反而导致过拟合特定样本")
            report_lines.append("  2. 数据质量：不同范围的QA数据质量存在差异")
            report_lines.append("  3. 微调超参数：需要针对不同数据量调整学习率或轮数")
    
    report_lines.append("")
    report_lines.append("="*70)
    
    # 保存报告
    report_file = output_dir / "lineage_similarity_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"报告保存到: {report_file}")
    
    # 打印到控制台
    print('\n'.join(report_lines))


def main():
    """Main function: Compute all lineage similarities"""
    
    # Check if relation model exists
    if not Path(RELATION_MODEL_PATH).exists():
        logger.error(f"关系模型不存在: {RELATION_MODEL_PATH}")
        logger.error("请确保已经训练了关系网络模型")
        return
    
    # Load relation network
    logger.info(f"Load relation network: {RELATION_MODEL_PATH}")
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
    
    logger.info(f"关系网络加载成功，使用设备: {device}")
    
    # Load model A embeddings
    emb_a_file = EMBEDDINGS_DIR / 'A_embeddings.pt'
    if not emb_a_file.exists():
        logger.error(f"模型A嵌入不存在: {emb_a_file}")
        logger.error("请先运行 step3_generate_answers.py 和 step3_5_generate_embeddings.py")
        return
    
    logger.info("加载模型A嵌入...")
    emb_a = load_embeddings(emb_a_file)
    logger.info(f"模型A嵌入形状: {emb_a.shape}")
    
    # Compute similarity for each variant
    results = {}
    
    for variant_name, split_config in QA_SPLITS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"计算 {variant_name} 与模型A的血缘相似度")
        logger.info(f"{'='*60}")
        
        # 加载B变体嵌入
        emb_b_file = EMBEDDINGS_DIR / f'{variant_name}_embeddings.pt'
        if not emb_b_file.exists():
            logger.warning(f"嵌入文件不存在: {emb_b_file}")
            continue
        
        # 加载B-A差异（从ba_embeddings目录）
        ba_diff_dir = EXPERIMENT_DIR / 'ba_embeddings'
        ba_diff_file = ba_diff_dir / f'{variant_name}_minus_A.pt'
        if not ba_diff_file.exists():
            logger.warning(f"B-A差异文件不存在: {ba_diff_file}")
            continue
        
        try:
            emb_b = load_embeddings(emb_b_file)
            emb_ba = load_embeddings(ba_diff_file)
            
            logger.info(f"  {variant_name}嵌入形状: {emb_b.shape}")
            logger.info(f"  B-A差异形状: {emb_ba.shape}")
            
            # Compute similarity
            avg_sim, sims_per_q = compute_lineage_similarity(
                encnet, prenet, emb_a, emb_b, emb_ba, device
            )
            
            # Statistics
            sims_array = np.array(sims_per_q)
            
            results[variant_name] = {
                'avg_similarity': avg_sim,
                'std_similarity': float(np.std(sims_array)),
                'min_similarity': float(np.min(sims_array)),
                'max_similarity': float(np.max(sims_array)),
                'similarities_per_question': sims_per_q,
                'training_data_range': f"Q{split_config['range'][0]+1}-Q{split_config['range'][1]}",
                'training_samples': split_config['range'][1] - split_config['range'][0]
            }
            
            logger.info(f"  Average similarity: {avg_sim:.6f}")
            logger.info(f"  标准差: {results[variant_name]['std_similarity']:.6f}")
            
        except Exception as e:
            logger.error(f"处理 {variant_name} 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Save results
    results_file = RESULTS_DIR / "lineage_similarity_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n结果保存到: {results_file}")
    
    # 可视化
    if results:
        logger.info("\n生成可视化...")
        visualize_results(results, RESULTS_DIR)
        
        logger.info("\n生成报告...")
        generate_report(results, RESULTS_DIR)
    
    logger.info(f"\n{'='*60}")
    logger.info("血缘相似度计算完成！")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
