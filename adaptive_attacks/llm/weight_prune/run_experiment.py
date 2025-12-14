#!/usr/bin/env python3
"""
Weight Pruning Experiment Runner
Complete experiment: Pruning + PPL Evaluation + Lineage Evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import config

# Add lineage model code path
sys.path.insert(0, config.LINEAGE_CODE_PATH)
from szy_qwen import TransformerEncoder, VectorRelationNet

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelPruner:
    """Model Pruner"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def magnitude_pruning(self, pruning_ratio):
        """Magnitude-based pruning"""
        if pruning_ratio == 0:
            return {'pruning_ratio': 0}
        
        logger.info(f"Starting pruning - Target ratio: {pruning_ratio*100:.0f}%")
        
        # Collect prunable parameters
        all_weights = []
        prunable_params = []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'norm' not in name.lower() and 'embed' not in name.lower():
                all_weights.append(param.data.abs().view(-1))
                prunable_params.append((param, name))
        
        # Calculate threshold
        layer_thresholds = []
        for w in all_weights:
            layer_threshold = torch.quantile(w.float(), pruning_ratio)
            layer_thresholds.append(layer_threshold.item())
        threshold = np.median(layer_thresholds)
        
        # Apply pruning
        total_params = 0
        pruned_params = 0
        for param, name in prunable_params:
            mask = (param.data.abs() > threshold).float()
            param.data *= mask
            total_params += param.numel()
            pruned_params += (mask == 0).sum().item()
        
        actual_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"Pruning completed - Actual ratio: {actual_ratio*100:.1f}%")
        
        return {
            'pruning_ratio': pruning_ratio,
            'actual_pruning_ratio': actual_ratio,
            'total_params': total_params,
            'pruned_params': pruned_params
        }
    
    def evaluate_perplexity(self, texts):
        """Evaluate perplexity"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Computing PPL"):
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': float(perplexity),
            'avg_loss': float(avg_loss)
        }
    
    def extract_embeddings(self, texts):
        """Extract embeddings"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embeddings.append(hidden_states.cpu())
        
        return embeddings


def evaluate_lineage(pruned_embeddings, parent_embeddings, negative_embeddings, lineage_model_path, device):
    """
    Evaluate lineage relationship
    
    Positive samples: Original Qwen-Instruct (parent) + Pruned Qwen-Instruct (child)
    Negative samples: Original Qwen-Instruct (parent) + Qwen-Math (unrelated)
    """
    logger.info("\nEvaluating lineage relationship...")
    
    # Load lineage model
    checkpoint = torch.load(lineage_model_path, map_location=device, weights_only=False)
    encnet = TransformerEncoder().to(device)
    prenet = VectorRelationNet().to(device)
    encnet.load_state_dict(checkpoint['encnet_state_dict'])
    prenet.load_state_dict(checkpoint['prenet_state_dict'])
    encnet.eval()
    prenet.eval()
    
    pos_similarities = []
    neg_similarities = []
    
    with torch.no_grad():
        for i in range(len(pruned_embeddings)):
            # Convert to float32
            parent_emb = parent_embeddings[i].to(device).float()
            pruned_emb = pruned_embeddings[i].to(device).float()
            negative_emb = negative_embeddings[i].to(device).float()
            
            # Compute differences
            pruned_diff = pruned_emb - parent_emb
            negative_diff = negative_emb - parent_emb
            
            # Encode
            enc_parent = encnet(parent_emb)
            enc_pruned = encnet(pruned_emb)
            enc_pruned_diff = encnet(pruned_diff)
            enc_negative = encnet(negative_emb)
            enc_negative_diff = encnet(negative_diff)
            
            # Compute relations
            pos_relation = prenet(enc_pruned, enc_pruned_diff)
            neg_relation = prenet(enc_negative, enc_negative_diff)
            
            # Compute similarities
            pos_sim = F.cosine_similarity(enc_parent, pos_relation, dim=-1).cpu().numpy()
            neg_sim = F.cosine_similarity(enc_parent, neg_relation, dim=-1).cpu().numpy()
            
            pos_similarities.append(float(pos_sim.mean()))
            neg_similarities.append(float(neg_sim.mean()))
    
    # Compute metrics
    pos_sims = np.array(pos_similarities)
    neg_sims = np.array(neg_similarities)
    
    threshold = config.LINEAGE_THRESHOLD
    tp = np.sum(pos_sims > threshold)
    fn = np.sum(pos_sims <= threshold)
    tn = np.sum(neg_sims <= threshold)
    fp = np.sum(neg_sims > threshold)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    accuracy = sum(p > n for p, n in zip(pos_similarities, neg_similarities)) / len(pos_similarities)
    
    return {
        'tpr': float(tpr),
        'fpr': float(fpr),
        'accuracy': float(accuracy),
        'pos_mean': float(pos_sims.mean()),
        'neg_mean': float(neg_sims.mean())
    }


def load_test_texts():
    """Load test texts"""
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50]
        return texts[:config.PPL_NUM_SAMPLES]
    except:
        # Use default texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world.",
            "Natural language processing enables AI to understand text.",
        ] * (config.PPL_NUM_SAMPLES // 3)
        return texts[:config.PPL_NUM_SAMPLES]


def main():
    """Main experiment flow"""
    logger.info("\n" + "="*80)
    logger.info("Weight Pruning Experiment: PPL + Lineage Analysis")
    logger.info("="*80)
    
    # Validate configuration
    if not config.validate_config():
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test texts
    test_texts = load_test_texts()
    lineage_texts = test_texts[:config.LINEAGE_NUM_SAMPLES]
    
    # Extract parent and negative embeddings (only once)
    logger.info("\nExtracting parent model embeddings...")
    parent_pruner = ModelPruner(config.PARENT_MODEL_PATH, device)
    parent_embeddings = parent_pruner.extract_embeddings(lineage_texts)
    del parent_pruner
    torch.cuda.empty_cache()
    
    logger.info("\nExtracting negative model embeddings...")
    negative_pruner = ModelPruner(config.NEGATIVE_MODEL_PATH, device)
    negative_embeddings = negative_pruner.extract_embeddings(lineage_texts)
    del negative_pruner
    torch.cuda.empty_cache()
    
    # Run experiment for each pruning ratio
    all_results = {}
    
    for i, pruning_ratio in enumerate(config.PRUNING_RATIOS, 1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# [{i}/{len(config.PRUNING_RATIOS)}] Pruning Ratio: {pruning_ratio*100:.0f}%")
        logger.info(f"{'#'*80}")
        
        try:
            # Load and prune model
            pruner = ModelPruner(config.PARENT_MODEL_PATH, device)
            pruning_stats = pruner.magnitude_pruning(pruning_ratio)
            
            # Evaluate PPL
            logger.info("\nEvaluating PPL...")
            ppl_result = pruner.evaluate_perplexity(test_texts)
            logger.info(f"  PPL: {ppl_result['perplexity']:.4f}")
            
            # Evaluate lineage
            logger.info("\nEvaluating lineage...")
            pruned_embeddings = pruner.extract_embeddings(lineage_texts)
            lineage_result = evaluate_lineage(
                pruned_embeddings, parent_embeddings, negative_embeddings,
                config.LINEAGE_MODEL_PATH, device
            )
            logger.info(f"  TPR: {lineage_result['tpr']:.4f}")
            logger.info(f"  FPR: {lineage_result['fpr']:.4f}")
            
            # Save results
            all_results[pruning_ratio] = {
                'pruning_stats': pruning_stats,
                'ppl_result': ppl_result,
                'lineage_result': lineage_result
            }
            
            del pruner
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Pruning ratio {pruning_ratio*100:.0f}% failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Save results
    with open(config.RESULTS_FILE, 'w') as f:
        json.dump({
            'experiment': config.EXPERIMENT_NAME,
            'parent_model': config.PARENT_MODEL_NAME,
            'negative_model': config.NEGATIVE_MODEL_NAME,
            'results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*100)
    logger.info("Experiment Results Summary")
    logger.info("="*100)
    logger.info(f"{'Pruning':<10} {'PPL':<15} {'PPL Change':<15} {'TPR':<10} {'FPR':<10} {'Accuracy':<12}")
    logger.info("-"*100)
    
    baseline_ppl = None
    for ratio in config.PRUNING_RATIOS:
        if ratio in all_results:
            ppl = all_results[ratio]['ppl_result']['perplexity']
            lr = all_results[ratio]['lineage_result']
            
            if baseline_ppl is None:
                baseline_ppl = ppl
                ppl_change = 0.0
            else:
                ppl_change = ((ppl - baseline_ppl) / baseline_ppl) * 100
            
            logger.info(f"{ratio*100:<10.0f}% {ppl:<15.2f} {ppl_change:+.1f}%{'':<9} "
                       f"{lr['tpr']:<10.4f} {lr['fpr']:<10.4f} {lr['accuracy']:<12.4f}")
    
    logger.info("="*100)
    logger.info(f"\nResults saved to: {config.RESULTS_FILE}")
    logger.info("Experiment completed!")


if __name__ == "__main__":
    main()
