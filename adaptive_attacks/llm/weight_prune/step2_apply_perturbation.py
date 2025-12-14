#!/usr/bin/env python3
"""
Step 2: Apply Weight Pruning to Models
Applies magnitude-based pruning to remove low-magnitude weights at different ratios.
"""

import os
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import copy

from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step2_apply_pruning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def magnitude_pruning(model, pruning_ratio):
    """
    Apply magnitude-based pruning to model parameters.
    
    Args:
        model: Model to prune
        pruning_ratio: Ratio of parameters to prune (0.2 = 20%)
    
    Returns:
        model: Pruned model
        pruning_stats: Statistics about pruning
    """
    if pruning_ratio == 0:
        return model, {'pruning_ratio': 0, 'total_params': 0, 'pruned_params': 0}
    
    # Collect all prunable weights
    all_weights = []
    prunable_params = []
    
    for name, param in model.named_parameters():
        # Only prune weight matrices, not bias and LayerNorm
        if 'weight' in name and not any(pattern in name.lower() for pattern in EXCLUDE_PATTERNS):
            all_weights.append(param.data.abs().view(-1))
            prunable_params.append((param, name))
    
    if not all_weights:
        logger.warning("No prunable parameters found")
        return model, {'pruning_ratio': 0, 'total_params': 0, 'pruned_params': 0}
    
    # Compute global threshold (Xth percentile)
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, pruning_ratio)
    
    # Apply pruning
    total_params = 0
    pruned_params = 0
    layer_stats = {}
    
    for param, name in prunable_params:
        mask = (param.data.abs() > threshold).float()
        param.data *= mask
        
        layer_total = param.numel()
        layer_pruned = (mask == 0).sum().item()
        
        total_params += layer_total
        pruned_params += layer_pruned
        
        layer_stats[name] = {
            'total': layer_total,
            'pruned': layer_pruned,
            'pruning_ratio': layer_pruned / layer_total
        }
    
    actual_pruning_ratio = pruned_params / total_params if total_params > 0 else 0
    
    pruning_stats = {
        'target_pruning_ratio': pruning_ratio,
        'actual_pruning_ratio': actual_pruning_ratio,
        'total_params': total_params,
        'pruned_params': pruned_params,
        'remaining_params': total_params - pruned_params,
        'threshold': float(threshold),
        'layer_stats': layer_stats
    }
    
    return model, pruning_stats


def apply_pruning_to_encoder_and_relation_net(encnet, prenet, pruning_ratio):
    """
    Apply magnitude-based pruning to both encoder and relation network.
    
    Args:
        encnet: TransformerEncoder model
        prenet: VectorRelationNet model
        pruning_ratio: Pruning ratio (0.0-1.0)
    
    Returns:
        pruned_encnet, pruned_prenet, stats
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Applying Pruning - Ratio: {pruning_ratio*100:.0f}%")
    logger.info(f"{'='*80}")
    
    if pruning_ratio == 0:
        logger.info("Pruning ratio is 0%, using original models")
        return encnet, prenet, {'pruning_ratio': 0}
    
    # Deep copy models to avoid modifying originals
    pruned_encnet = copy.deepcopy(encnet)
    pruned_prenet = copy.deepcopy(prenet)
    
    # Apply magnitude-based pruning
    logger.info(f"Method: Global magnitude-based pruning with ratio {pruning_ratio}")
    pruned_encnet, enc_stats = magnitude_pruning(pruned_encnet, pruning_ratio)
    pruned_prenet, pre_stats = magnitude_pruning(pruned_prenet, pruning_ratio)
    
    stats = {
        'method': 'magnitude_pruning',
        'encoder_stats': enc_stats,
        'relation_net_stats': pre_stats,
        'total_params': enc_stats['total_params'] + pre_stats['total_params'],
        'total_pruned': enc_stats['pruned_params'] + pre_stats['pruned_params']
    }
    
    logger.info(f"Encoder: {enc_stats['pruned_params']:,}/{enc_stats['total_params']:,} pruned "
               f"({enc_stats['actual_pruning_ratio']*100:.1f}%)")
    logger.info(f"RelationNet: {pre_stats['pruned_params']:,}/{pre_stats['total_params']:,} pruned "
               f"({pre_stats['actual_pruning_ratio']*100:.1f}%)")
    logger.info(f"Pruning completed")
    
    return pruned_encnet, pruned_prenet, stats


def main():
    """Main function to apply pruning"""
    logger.info("="*80)
    logger.info("Step 2: Apply Weight Pruning to Lineage Detector")
    logger.info("="*80)
    
    # Check if trained model exists
    model_path = Path(RELATION_MODEL_PATH)
    if not model_path.exists():
        logger.error(f"Trained model not found: {model_path}")
        logger.error("Please run step1_train_detector.py first")
        return
    
    # Load trained model
    logger.info(f"\nLoading trained model from: {model_path}")
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Training accuracy: {checkpoint['accuracy']:.4f}")
    logger.info(f"  Training epoch: {checkpoint['epoch']}")
    
    # Import models
    from models import TransformerEncoder, VectorRelationNet
    
    # Create model instances
    encnet = TransformerEncoder(
        feat_dim=MODEL_CONFIG['feat_dim'],
        d_model=MODEL_CONFIG['d_model'],
        kernel_size=MODEL_CONFIG['kernel_size'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    prenet = VectorRelationNet(
        embedding_dim=MODEL_CONFIG['embedding_dim']
    ).to(device)
    
    # Load state dicts
    encnet.load_state_dict(checkpoint['encnet_state_dict'])
    prenet.load_state_dict(checkpoint['prenet_state_dict'])
    
    logger.info(f"\nPruning method: {PRUNING_METHOD}")
    logger.info(f"Pruning ratios: {[f'{r*100:.0f}%' for r in PRUNING_RATIOS]}")
    
    # Apply pruning at different ratios
    all_stats = {}
    
    for ratio in tqdm(PRUNING_RATIOS, desc="Applying pruning"):
        # Apply pruning
        pruned_encnet, pruned_prenet, stats = apply_pruning_to_encoder_and_relation_net(
            encnet, prenet, ratio
        )
        
        # Save pruned models
        save_path = PRUNED_MODELS_DIR / f"pruned_{int(ratio*100)}pct.pth"
        torch.save({
            'encnet_state_dict': pruned_encnet.state_dict(),
            'prenet_state_dict': pruned_prenet.state_dict(),
            'pruning_ratio': ratio,
            'pruning_stats': stats,
            'original_accuracy': checkpoint['accuracy'],
            'original_epoch': checkpoint['epoch']
        }, save_path)
        
        logger.info(f"Saved pruned model to: {save_path}")
        all_stats[ratio] = stats
    
    # Save statistics
    import json
    with open(RESULTS_DIR / 'pruning_stats.json', 'w') as f:
        json.dump({
            'method': PRUNING_METHOD,
            'ratios': PRUNING_RATIOS,
            'stats': {str(k): v for k, v in all_stats.items()}
        }, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("Pruning completed for all ratios")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
