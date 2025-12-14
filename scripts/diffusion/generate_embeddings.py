"""
Generate embeddings for diffusion model lineage detection.

This script extracts U-Net features from parent and child Stable Diffusion models
using COCO dataset images as probe samples.

Usage:
    python scripts/diffusion/generate_embeddings.py
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
from src.diffusion.diffusion_dataset import DiffusionDataset


if __name__ == "__main__":
    # Create dataset instance
    dataset = DiffusionDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    pos_parent_list = []
    pos_child_list = []
    pos_minus_list = []
    posn_minus_list = []
    neg_list = []
    neg_minus_list = []
    negp_minus_list = []
    pos_neg_list = []
    
    for i, batch in enumerate(dataloader):
        images = batch["image"]
        caption = batch["caption"]
        pos_parent = batch["pos_parent"]
        pos_child = batch["pos_child"]
        pos_minus = batch["pos_minus"]
        posn_minus = batch["posn_minus"]
        neg = batch["neg"]
        neg_minus = batch["neg_minus"]
        negp_minus = batch["negp_minus"]
        pos_neg = batch["pos_neg"]
        
        pos_parent_list.append(pos_parent.cpu())
        pos_child_list.append(pos_child.cpu())
        pos_minus_list.append(pos_minus.cpu())
        posn_minus_list.append(posn_minus.cpu())
        neg_list.append(neg.cpu())
        neg_minus_list.append(neg_minus.cpu())
        negp_minus_list.append(negp_minus.cpu())
        pos_neg_list.append(pos_neg.cpu())
        
        if i > 1000:
            break
    
    # Concatenate all tensors
    pos_parent_tensor = torch.cat(pos_parent_list, dim=0)
    pos_child_tensor = torch.cat(pos_child_list, dim=0)
    pos_minus_tensor = torch.cat(pos_minus_list, dim=0)
    posn_minus_tensor = torch.cat(posn_minus_list, dim=0)
    pos_neg_tensor = torch.cat(pos_neg_list, dim=0)
    neg_tensor = torch.cat(neg_list, dim=0)
    neg_minus_tensor = torch.cat(neg_minus_list, dim=0)
    negp_minus_tensor = torch.cat(negp_minus_list, dim=0)
    
    # Save embeddings
    save_path = "data/embeddings/diffusion/"
    os.makedirs(save_path, exist_ok=True)
    
    torch.save(pos_parent_tensor, os.path.join(save_path, "pos_parent.pt"))
    torch.save(pos_child_tensor, os.path.join(save_path, "pos_child.pt"))
    torch.save(pos_minus_tensor, os.path.join(save_path, "pos_minus.pt"))
    torch.save(posn_minus_tensor, os.path.join(save_path, "posn_minus.pt"))
    torch.save(neg_tensor, os.path.join(save_path, "neg.pt"))
    torch.save(neg_minus_tensor, os.path.join(save_path, "neg_minus.pt"))
    torch.save(negp_minus_tensor, os.path.join(save_path, "negp_minus.pt"))
    torch.save(pos_neg_tensor, os.path.join(save_path, "pos_neg.pt"))
    
    print(f"Embeddings saved to {save_path}")
    print(f"Parent shape: {pos_parent_tensor.shape}")
    print(f"Child shape: {pos_child_tensor.shape}")
