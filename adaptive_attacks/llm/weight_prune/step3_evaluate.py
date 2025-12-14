#!/usr/bin/env python3
"""
Step 3: Evaluate Pruned Models
Evaluates lineage detection performance on pruned models.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import *
from models import TransformerEncoder, VectorRelationNet
from dataset import QwenEmbeddingDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step3_evaluate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_pruned_model(encnet, prenet, test_loader, device, threshold=0.5):
    """
    Evaluate pruned model on test set.
    Uses the same evaluation logic as training.
    
    Args:
        encnet: Encoder network
        prenet: Relation network
        test_loader: Test data loader
        device: Computing device
        threshold: Similarity threshold for classification
    
    Returns:
        eval_result: Dictionary containing evaluation metrics
    """
    encnet.eval()
    prenet.eval()
    
    pos_similarities = []
    neg_similarities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            parent = batch['parent'].to(device)
            child = batch['child'].to(device)
            ba = batch['ba'].to(device)
            neg_child = batch['neg_child'].to(device)
            neg_ba = batch['neg_ba'].to(device)
            
            # Encode features
            enc_parent = encnet(parent)
            enc_child = encnet(child)
            enc_ba = encnet(ba)
            enc_neg_child = encnet(neg_child)
            enc_neg_ba = encnet(neg_ba)
            
            # Compute relation representations
            pos_relation = prenet(enc_child, enc_ba)
            neg_relation = prenet(enc_neg_child, enc_neg_ba)
            
            # Compute similarities
            pos_sim = F.cosine_similarity(enc_parent, pos_relation).cpu().numpy()
            neg_sim = F.cosine_similarity(enc_parent, neg_relation).cpu().numpy()
            
            pos_similarities.extend(pos_sim.tolist())
            neg_similarities.extend(neg_sim.tolist())
    
    # Compute metrics
    pos_sims = np.array(pos_similarities)
    neg_sims = np.array(neg_similarities)
    
    # Accuracy: how often positive similarity > negative similarity
    accuracy = sum(p > n for p, n in zip(pos_similarities, neg_similarities)) / len(pos_similarities)
    
    # TPR and FPR using fixed threshold
    tp = np.sum(pos_sims > threshold)
    fn = np.sum(pos_sims <= threshold)
    tn = np.sum(neg_sims <= threshold)
    fp = np.sum(neg_sims > threshold)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    result = {
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'threshold': threshold,
        'num_samples': len(pos_similarities),
        'pos_mean': float(pos_sims.mean()),
        'pos_std': float(pos_sims.std()),
        'neg_mean': float(neg_sims.mean()),
        'neg_std': float(neg_sims.std()),
        'separation': float(pos_sims.mean() - neg_sims.mean())
    }
    
    return result


def main():
    """Main evaluation function"""
    logger.info("="*80)
    logger.info("Step 3: Evaluate Pruned Models")
    logger.info("="*80)
    
    # Setup device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test dataset
    logger.info("\nLoading test dataset...")
    test_dataset = QwenEmbeddingDataset(
        emb_dir=EMBEDDING_ROOT,
        split='test',
        test_ratio=EVAL_CONFIG['test_ratio'],
        tasks=TASKS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Check for pruned models
    pruned_model_files = sorted(PRUNED_MODELS_DIR.glob("pruned_*.pth"))
    if not pruned_model_files:
        logger.error(f"No pruned models found in {PRUNED_MODELS_DIR}")
        logger.error("Please run step2_apply_perturbation.py first")
        return
    
    logger.info(f"\nFound {len(pruned_model_files)} pruned models")
    logger.info(f"Evaluation threshold: {EVAL_CONFIG['threshold']}")
    
    # Store all results
    all_results = {}
    
    for model_file in pruned_model_files:
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_file.name}")
        logger.info(f"{'='*80}")
        
        try:
            # Load pruned model
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)
            pruning_ratio = checkpoint['pruning_ratio']
            
            logger.info(f"Pruning ratio: {pruning_ratio*100:.0f}%")
            logger.info(f"Original training accuracy: {checkpoint['original_accuracy']:.4f}")
            
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
            
            # Load pruned state dicts
            encnet.load_state_dict(checkpoint['encnet_state_dict'])
            prenet.load_state_dict(checkpoint['prenet_state_dict'])
            
            # Evaluate
            eval_result = evaluate_pruned_model(
                encnet, prenet, test_loader, device,
                threshold=EVAL_CONFIG['threshold']
            )
            
            logger.info(f"\nEvaluation Results:")
            logger.info(f"  Accuracy: {eval_result['accuracy']:.4f}")
            logger.info(f"  TPR: {eval_result['tpr']:.4f}")
            logger.info(f"  FPR: {eval_result['fpr']:.4f}")
            logger.info(f"  Positive similarity: {eval_result['pos_mean']:.4f} ± {eval_result['pos_std']:.4f}")
            logger.info(f"  Negative similarity: {eval_result['neg_mean']:.4f} ± {eval_result['neg_std']:.4f}")
            logger.info(f"  Separation: {eval_result['separation']:.4f}")
            
            # Store results
            all_results[pruning_ratio] = {
                'pruning_stats': checkpoint.get('pruning_stats', {}),
                'eval_result': eval_result,
                'original_accuracy': checkpoint['original_accuracy']
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Save detailed results
    logger.info("\n" + "="*80)
    logger.info("Saving results...")
    
    results_file = get_results_file()
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_type': 'weight_pruning',
            'pruning_method': PRUNING_METHOD,
            'pruning_ratios': PRUNING_RATIOS,
            'threshold': EVAL_CONFIG['threshold'],
            'results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("Evaluation Results Summary")
    logger.info("="*80)
    logger.info(f"{'Ratio':<10} {'Accuracy':<12} {'TPR':<10} {'FPR':<10} {'Separation':<12}")
    logger.info("-"*80)
    
    sorted_ratios = sorted(all_results.keys())
    for ratio in sorted_ratios:
        er = all_results[ratio]['eval_result']
        logger.info(f"{ratio*100:<10.0f}% {er['accuracy']:<12.4f} {er['tpr']:<10.4f} "
                   f"{er['fpr']:<10.4f} {er['separation']:<12.4f}")
    
    # Performance degradation analysis
    if 0.0 in all_results:
        baseline_acc = all_results[0.0]['eval_result']['accuracy']
        logger.info("-"*80)
        logger.info("Performance Degradation (relative to 0% pruning):")
        for ratio in sorted_ratios:
            if ratio > 0:
                acc = all_results[ratio]['eval_result']['accuracy']
                drop = baseline_acc - acc
                drop_pct = (drop / baseline_acc) * 100 if baseline_acc > 0 else 0
                logger.info(f"  {ratio*100:.0f}% pruning: Accuracy drop {drop:.4f} ({drop_pct:.1f}%)")
    
    logger.info("="*80)
    logger.info("Evaluation completed!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
