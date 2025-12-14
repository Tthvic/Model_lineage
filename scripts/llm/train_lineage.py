#!/usr/bin/env python3
"""
Qwen Knowledge Relation Learning System
Training Script: Learns the knowledge lineage relationship of Qwen models.
"""

import os
import sys
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm.models import TransformerEncoder, VectorRelationNet, TripletLoss
from src.llm.dataset import LineageEmbeddingDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(encnet, prenet, test_loader, criterion, device):
    """Evaluate the model."""
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
            
            # Compute relation representation
            pos_relation = prenet(enc_child, enc_ba)  # Positive: Child + B-A
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
    threshold = 0.5
    
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
    """Train for one epoch."""
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
        
        # Compute relation representation
        pos_relation = prenet(enc_child, enc_ba)  # Positive: Child + B-A
        neg_relation = prenet(enc_neg_child, enc_neg_ba)  # Negative
        
        # Compute loss
        loss1 = criterion(enc_parent, pos_relation, neg_relation)  # Standard triplet
        loss2 = criterion(pos_relation, enc_parent, neg_relation)  # Swapped anchor
        loss = loss1 + loss2
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Lineage Model")
    parser.add_argument("--config", default="configs/llm/qwen2_5.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error(f"Could not load config from {args.config}")
        sys.exit(1)
    
    # Setup device
    device_str = config['training']['device'] if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Create save directory
    root_dir = Path(config['data']['root_dir'])
    save_dir = root_dir / config['data']['result_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets and dataloaders
    train_dataset = LineageEmbeddingDataset(config, split='train')
    test_dataset = LineageEmbeddingDataset(config, split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['training']['num_workers']
    )
    
    # Create models
    encnet = TransformerEncoder(
        feat_dim=config['model']['encoder']['feat_dim'],
        d_model=config['model']['encoder']['d_model'],
        kernel_size=config['model']['encoder']['kernel_size'],
        dropout=config['model']['encoder']['dropout']
    ).to(device)
    
    prenet = VectorRelationNet(
        embedding_dim=config['model']['relation_net']['embedding_dim']
    ).to(device)
    
    # Create optimizer and loss
    optimizer = optim.Adam(
        list(encnet.parameters()) + list(prenet.parameters()), 
        lr=config['training']['learning_rate']
    )
    criterion = TripletLoss(margin=config['model']['triplet_loss']['margin'])
    
    # Training history
    train_losses = []
    test_results = []
    best_accuracy = 0.0
    
    logger.info("Starting training...")
    num_epochs = config['training']['num_epochs']
    eval_interval = config['training']['eval_interval']
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(encnet, prenet, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        if (epoch + 1) % eval_interval == 0:
            test_result = evaluate_model(encnet, prenet, test_loader, criterion, device)
            test_results.append(test_result)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Test Loss: {test_result['loss']:.4f}")
            logger.info(f"Accuracy: {test_result['accuracy']:.4f}")
            logger.info(f"TPR: {test_result['tpr']:.4f}, FPR: {test_result['fpr']:.4f}")
            
            # Save best model
            if test_result['accuracy'] > best_accuracy:
                best_accuracy = test_result['accuracy']
                torch.save({
                    'epoch': epoch,
                    'encnet_state_dict': encnet.state_dict(),
                    'prenet_state_dict': prenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'test_result': test_result
                }, save_dir / 'best_model.pth')
                logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            # Save detailed results
            torch.save({
                'epoch': epoch,
                'test_result': test_result,
                'train_loss': train_loss
            }, save_dir / f'epoch_{epoch+1}_results.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'test_results': test_results,
        'best_accuracy': best_accuracy
    }
    
    with open(save_dir / 'training_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in training_history.items():
            if key == 'test_results':
                serializable_results = []
                for result in value:
                    serializable_result = {}
                    for k, v in result.items():
                        if isinstance(v, (list, np.ndarray)):
                            serializable_result[k] = v if isinstance(v, list) else v.tolist()
                        else:
                            serializable_result[k] = v
                    serializable_results.append(serializable_result)
                serializable_history[key] = serializable_results
            else:
                serializable_history[key] = value
        
        json.dump(serializable_history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    logger.info(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
