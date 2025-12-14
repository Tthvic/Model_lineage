"""
Test lineage detector on diffusion model pairs.

This script evaluates the trained lineage detector on parent-child model pairs
to verify lineage relationships.

Usage:
    python scripts/diffusion/test_lineage.py
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.diffusion.task_vectors import TaskVector
import numpy as np
from torch.optim.lr_scheduler import StepLR
from src.diffusion.networks import Encoder, VectorRelationNet
from src.diffusion.loss import ContrastiveLoss
from src.diffusion.diffusion_dataset import DiffusionDataset
import datetime
from src.diffusion.dataset_with_attacks import DiffusionDatasetWithAttacks
from src.diffusion.lineage_model import LineageDetectorModel
import torchvision
import torch.nn.functional as F


def add_noise_to_model(model, noise_ratio=0.20):
    """
    Add Gaussian noise to model parameters.
    
    Args:
        model: The model to add noise to
        noise_ratio: Noise standard deviation as ratio of parameter mean absolute value (default: 0.20)
    
    Returns:
        Model with added noise
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data
                mean_abs = param_data.abs().mean()
                noise_std = mean_abs * noise_ratio
                noise = torch.randn_like(param_data) * noise_std
                param.add_(noise)
    return model


def test(model, pos_parent, pos_child, pos_minus, neg, neg_minus, batch_size=10, batch_jump=1, thresh=0.3):
    """Test model on positive and negative pairs"""
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


def load_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist.")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return model


def main():
    # Model paths - update these to your models
    child_model_name = "stabilityai/stable-diffusion-2"
    
    # Create dataset
    dataset = DiffusionDatasetWithAttacks(
        pos_child_name=child_model_name,
        neg_name="stabilityai/stable-diffusion-2-1-base",
        num_classes=1
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    pos_parent_l = []
    pos_child_l = []
    pos_min_l = []
    neg_l = []
    neg_min_l = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = LineageDetectorModel(device)
    checkpoint_path = "data/models/diffusion/checkpoint.pth"
    model = load_checkpoint(model, checkpoint_path)
    
    if model is None:
        print("Please train the model first using train_lineage.py")
        return
        
    model.eval()

    # Extract embeddings from test set
    for i, batch in enumerate(dataloader):
        images = batch["image"]
        caption = batch["caption"]
        pos_parent = batch["pos_parent"]
        pos_child = batch["pos_child"]
        pos_min = batch["pos_minus"]
        neg = batch["neg"]
        neg_min = batch["neg_minus"]
        
        pos_parent_l.append(pos_parent.cpu())
        pos_child_l.append(pos_child.cpu())
        pos_min_l.append(pos_min.cpu())
        neg_l.append(neg.cpu())
        neg_min_l.append(neg_min.cpu())
        
        if i > 50:
            break
    
    # Concatenate and test
    pos_parent_tensor = torch.cat(pos_parent_l, dim=0).to(device)
    pos_child_tensor = torch.cat(pos_child_l, dim=0).to(device)
    pos_min_tensor = torch.cat(pos_min_l, dim=0).to(device)
    neg_tensor = torch.cat(neg_l, dim=0).to(device)
    neg_min_tensor = torch.cat(neg_min_l, dim=0).to(device)
    
    test(model, pos_parent_tensor, pos_child_tensor, pos_min_tensor, 
         neg_tensor, neg_min_tensor, batch_jump=1)


if __name__ == "__main__":
    main()
