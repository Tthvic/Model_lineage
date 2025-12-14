"""
Train lineage detector for diffusion models.

This script trains the encoder and relation network to detect lineage relationships
between Stable Diffusion models by verifying consistency between parameter changes
and knowledge evolution.

Usage:
    python scripts/diffusion/train_lineage.py
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision
import random 
import numpy as np
from torch.optim.lr_scheduler import StepLR
from src.diffusion.networks import Encoder, VectorRelationNet
from src.diffusion.loss import ContrastiveLoss
from src.diffusion.diffusion_dataset import DiffusionDataset
from src.diffusion.lineage_model import LineageDetectorModel
import datetime


def load_half_tensor(path):
    """Load first half of saved tensor for training"""
    data = torch.load(path)
    half_size = data.shape[0] // 2
    return data[:half_size]


def test(model, pos_parent, pos_child, pos_minus, neg, neg_minus, batch_size=10, batch_jump=5, thresh=0.3):
    """
    Test model accuracy on positive and negative pairs.
    
    Args:
        model: Trained lineage detector model
        pos_parent: Parent model embeddings
        pos_child: Child model embeddings  
        pos_minus: Task vector embeddings
        neg: Negative sample embeddings
        neg_minus: Negative task vector embeddings
        batch_size: Batch size for testing
        batch_jump: Number of batches to skip between tests
        thresh: Similarity threshold for classification
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    length = pos_parent.shape[0]
    
    # Test positive pairs
    with torch.no_grad():
        for i in range(0, length, batch_size * batch_jump):
            parent = pos_parent[i:i + batch_size]
            child = pos_child[i:i + batch_size]
            minus = pos_minus[i:i + batch_size]
            n = neg[i:i + batch_size]
            n_min = neg_minus[i:i + batch_size]

            child_fea, predicted_child, neg_preknow = model(
                parent, child, minus, n, n_min
            )

            similarity = F.cosine_similarity(predicted_child, child_fea)
            correct = (similarity > thresh).sum().item()
            total_correct += correct
            total_samples += batch_size

        accuracy = total_correct / total_samples * 100
        print(f"Test Accuracy on True Pairs: {accuracy:.2f}%")

    # Test negative pairs
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for i in range(0, length, batch_size * 5):
            parent = pos_parent[i:i + batch_size]
            child = pos_child[i:i + batch_size]
            minus = pos_minus[i:i + batch_size]
            n = neg[i:i + batch_size]
            n_min = neg_minus[i:i + batch_size]

            child_fea, predicted_child, neg_preknow = model(
                parent, child, minus, n, n_min
            )

            similarity = F.cosine_similarity(neg_preknow, child_fea)
            correct = (similarity < thresh).sum().item()
            total_correct += correct
            total_samples += batch_size

        accuracy = total_correct / total_samples * 100
        print(f"Test Accuracy on False Pairs: {accuracy:.2f}%")


def save_checkpoint(model, optimizer, epoch, loss, time):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    current_time = time
    checkpoint_folder = f"data/models/diffusion/checkpoint-{current_time}/"
    os.makedirs(checkpoint_folder, exist_ok=True)
    filepath = os.path.join(checkpoint_folder, f"epoch_{epoch}.pth")
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at epoch {epoch} to {filepath}")


def main():
    batch_size = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and optimizer
    model = LineageDetectorModel(device)
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.relation_net.parameters()), 
        lr=0.0001
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    cosine_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.1)
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=1)
    
    # Load pre-generated embedding data
    save_path = "data/embeddings/diffusion/"
    pos_parent = load_half_tensor(os.path.join(save_path, "pos_parent.pt")).to(device)
    pos_child = load_half_tensor(os.path.join(save_path, "pos_child.pt")).to(device)
    pos_minus = load_half_tensor(os.path.join(save_path, "pos_minus.pt")).to(device)
    posn_minus = load_half_tensor(os.path.join(save_path, "posn_minus.pt")).to(device)
    neg = load_half_tensor(os.path.join(save_path, "neg.pt")).to(device)
    neg_minus = load_half_tensor(os.path.join(save_path, "neg_minus.pt")).to(device)
    negp_minus = load_half_tensor(os.path.join(save_path, "negp_minus.pt")).to(device)
    pos_neg = load_half_tensor(os.path.join(save_path, "pos_neg.pt")).to(device)
    
    length = pos_parent.shape[0] // 2
    
    # Training
    num_epochs = 100
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for epoch in range(num_epochs):
        model.train()
        
        for i in range(0, length, batch_size):
            parent = pos_parent[i:i + batch_size]
            child = pos_child[i:i + batch_size]
            minus = pos_minus[i:i + batch_size]
            n = neg[i:i + batch_size]
            n_min = neg_minus[i:i + batch_size]
            n_child = pos_neg[i:i + batch_size]
            n_child_min = posn_minus[i:i + batch_size]
            n_child_nmin = negp_minus[i:i + batch_size]
            
            # Forward pass
            child_fea, predicted_child, neg_predicted = model(
                parent, child, minus, n, n_min
            )
            neg_fea, npreknow, negp_preknow = model(
                parent, n_child, n_child_min, n, n_child_nmin
            )
            
            # Create labels
            cosine_labels1 = torch.ones(batch_size).to(device)
            cosine_labels2 = -torch.ones(batch_size).to(device)

            # Calculate losses
            loss_pos = cosine_loss_fn(child_fea, predicted_child, cosine_labels1)
            loss_neg = cosine_loss_fn(child_fea, neg_predicted, cosine_labels2)
            loss_triplet = triplet_loss_fn(child_fea, predicted_child, neg_predicted)
            loss_triplet2 = triplet_loss_fn(predicted_child, child_fea, neg_predicted)
            loss_triplet1 = triplet_loss_fn(neg_fea, negp_preknow, npreknow)
            loss_triplet3 = triplet_loss_fn(negp_preknow, neg_fea, npreknow)
            loss_posn = cosine_loss_fn(neg_fea, npreknow, cosine_labels2)
            loss_negp = cosine_loss_fn(neg_fea, negp_preknow, cosine_labels1)
            
            # Total loss
            loss = (loss_pos + 5 * loss_neg + loss_triplet + loss_posn + 
                   5 * loss_negp + loss_triplet1 + loss_triplet2 + loss_triplet3)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Test
        model.eval()
        test(model, pos_parent, pos_child, pos_minus, neg, neg_minus)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        save_checkpoint(model, optimizer, epoch + 1, loss.item(), time)


if __name__ == "__main__":
    main()
