#!/usr/bin/env python3
"""
Step 1: Train Lineage Detector Model
Trains encoder and relation network for detecting parent-child relationships.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import random
import logging
from tqdm import tqdm
from pathlib import Path

from config import *
from models import TransformerEncoder, VectorRelationNet, TripletLoss
from dataset import QwenEmbeddingDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'step1_train_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_model(encnet, prenet, test_loader, criterion, device):
    """Evaluate the lineage detection model"""
    encnet.eval()
    prenet.eval()
    
    total_loss = 0.0
    pos_similarities = []
    neg_similarities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
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
            pos_relation = prenet(enc_child, enc_ba)  # Positive: child + B-A difference
            neg_relation = prenet(enc_neg_child, enc_neg_ba)  # Negative
            
            # Compute loss
            loss = criterion(enc_parent, pos_relation, neg_relation)
            total_loss += loss.item()
            
            # Compute similarities
            pos_sim = F.cosine_similarity(enc_parent, pos_relation).cpu().numpy()
            neg_sim = F.cosine_similarity(enc_parent, neg_relation).cpu().numpy()
            
            pos_similarities.extend(pos_sim.tolist())
            neg_similarities.extend(neg_sim.tolist())
    
    avg_loss = total_loss / len(test_loader)
    
    # Compute accuracy
    accuracy = sum(p > n for p, n in zip(pos_similarities, neg_similarities)) / len(pos_similarities)
    
    # Compute TPR and FPR
    pos_sims = np.array(pos_similarities)
    neg_sims = np.array(neg_similarities)
    threshold = EVAL_CONFIG['threshold']
    
    tp = np.sum(pos_sims > threshold)
    fn = np.sum(pos_sims <= threshold)
    tn = np.sum(neg_sims <= threshold)
    fp = np.sum(neg_sims > threshold)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'pos_similarities': pos_similarities,
        'neg_similarities': neg_similarities
    }


def train_epoch(encnet, prenet, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    encnet.train()
    prenet.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move data to device
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
        pos_relation = prenet(enc_child, enc_ba)  # Positive: child + B-A difference
        neg_relation = prenet(enc_neg_child, enc_neg_ba)  # Negative
        
        # Compute loss (use dual triplets for better learning)
        loss1 = criterion(enc_parent, pos_relation, neg_relation)  # Standard triplet
        loss2 = criterion(pos_relation, enc_parent, neg_relation)  # Swap anchor
        loss = loss1 + loss2
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training function"""
    logger.info("="*80)
    logger.info("Step 1: Train Lineage Detector Model")
    logger.info("="*80)
    
    # Set random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Setup device
    device = torch.device(TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
    logger.info("\nLoading datasets...")
    train_dataset = QwenEmbeddingDataset(
        emb_dir=EMBEDDING_ROOT,
        split='train',
        test_ratio=EVAL_CONFIG['test_ratio'],
        tasks=TASKS
    )
    test_dataset = QwenEmbeddingDataset(
        emb_dir=EMBEDDING_ROOT,
        split='test',
        test_ratio=EVAL_CONFIG['test_ratio'],
        tasks=TASKS
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create models
    logger.info("\nInitializing models...")
    encnet = TransformerEncoder(
        feat_dim=MODEL_CONFIG['feat_dim'],
        d_model=MODEL_CONFIG['d_model'],
        kernel_size=MODEL_CONFIG['kernel_size'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    
    prenet = VectorRelationNet(
        embedding_dim=MODEL_CONFIG['embedding_dim']
    ).to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        list(encnet.parameters()) + list(prenet.parameters()),
        lr=TRAINING_CONFIG['learning_rate']
    )
    criterion = TripletLoss(margin=TRAINING_CONFIG['triplet_margin'])
    
    # Training history
    train_losses = []
    test_results = []
    best_accuracy = 0.0
    
    logger.info("\nStarting training...")
    logger.info(f"Epochs: {TRAINING_CONFIG['num_epochs']}")
    logger.info(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    logger.info(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    logger.info(f"Triplet margin: {TRAINING_CONFIG['triplet_margin']}")
    
    for epoch in range(TRAINING_CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{TRAINING_CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(encnet, prenet, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate every N epochs
        if (epoch + 1) % TRAINING_CONFIG['eval_interval'] == 0:
            test_result = evaluate_model(encnet, prenet, test_loader, criterion, device)
            test_results.append(test_result)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Test Loss: {test_result['loss']:.4f}")
            logger.info(f"Accuracy: {test_result['accuracy']:.4f}")
            logger.info(f"TPR: {test_result['tpr']:.4f}, FPR: {test_result['fpr']:.4f}")
            
            # Save best model
            if test_result['accuracy'] > best_accuracy:
                best_accuracy = test_result['accuracy']
                
                # Save to relation network path
                model_save_path = Path(RELATION_MODEL_PATH)
                model_save_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'encnet_state_dict': encnet.state_dict(),
                    'prenet_state_dict': prenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'test_result': test_result
                }, model_save_path)
                
                logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
                logger.info(f"Saved to: {model_save_path}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'test_result': test_result,
                'train_loss': train_loss
            }, RESULTS_DIR / f'epoch_{epoch+1}_results.pth')
    
    # Save training history
    logger.info("\nSaving training history...")
    training_history = {
        'train_losses': train_losses,
        'test_results': [{k: v for k, v in r.items() if k not in ['pos_similarities', 'neg_similarities']} 
                        for r in test_results],
        'best_accuracy': best_accuracy
    }
    
    with open(RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("="*80)
    logger.info("Training completed!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    logger.info(f"Model saved to: {RELATION_MODEL_PATH}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
