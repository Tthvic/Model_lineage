"""
Test lineage detector on unrelated dataset (CIFAR-10).

This script tests the lineage detector on CIFAR-10 images instead of COCO to verify
that it doesn't produce false positives on out-of-distribution data.

Usage:
    python scripts/diffusion/test_unrelated_dataset.py
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
from scripts.diffusion.train_lineage import test
from scripts.diffusion.test_adaptive_attack import getvec, load_vec, extract_unet_features, load_checkpoint
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F


def main():
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    cifar_dataset = datasets.CIFAR10(
        root='data/datasets/cifar10/', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Limit to first 100 images for testing
    subset_indices = list(range(100))
    cifar_subset = torch.utils.data.Subset(cifar_dataset, subset_indices)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    init_name = "stabilityai/stable-diffusion-2-base"
    pos_parent_name = "stabilityai/stable-diffusion-2-base"
    pos_child_name = "stabilityai/stable-diffusion-2"
    neg_name = "stabilityai/stable-diffusion-2-1-base"
    neg_child_name = "stabilityai/stable-diffusion-2-1"

    init_model = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    pos_parent = StableDiffusionPipeline.from_pretrained(pos_parent_name).to(device)
    pos_child = StableDiffusionPipeline.from_pretrained(pos_child_name).to(device)
    neg = StableDiffusionPipeline.from_pretrained(neg_name).to(device)
    pos_neg = StableDiffusionPipeline.from_pretrained(neg_child_name).to(device)

    # Get task vectors
    initvec = getvec(init_model)
    pos_vec = getvec(pos_parent, pos_child)
    neg_vec = getvec(neg, pos_child)
    pos_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    neg_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    pos_minus = load_vec(pos_minus, pos_vec, initvec)
    neg_minus = load_vec(neg_minus, neg_vec, initvec)

    posn_vec = getvec(pos_parent, pos_neg)
    negp_vec = getvec(neg, pos_neg)
    posn_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    negp_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    posn_minus = load_vec(posn_minus, posn_vec, initvec)
    negp_minus = load_vec(negp_minus, negp_vec, initvec)

    # Collect features from CIFAR-10 images
    pos_parent_l = []
    pos_child_l = []
    pos_min_l = []
    neg_l = []
    neg_min_l = []

    for image, label in tqdm(cifar_subset):
        image = image.unsqueeze(0).to(device)
        class_name = class_names[label]
        caption = f"a photo of {class_name}"

        with torch.no_grad():
            pos_parent_emb = extract_unet_features(image, caption, pos_parent, device)
            pos_child_emb = extract_unet_features(image, caption, pos_child, device)
            pos_min_emb = extract_unet_features(image, caption, pos_minus, device)
            neg_emb = extract_unet_features(image, caption, neg, device)
            neg_min_emb = extract_unet_features(image, caption, neg_minus, device)

        pos_parent_l.append(pos_parent_emb.cpu())
        pos_child_l.append(pos_child_emb.cpu())
        pos_min_l.append(pos_min_emb.cpu())
        neg_l.append(neg_emb.cpu())
        neg_min_l.append(neg_min_emb.cpu())

    # Concatenate features
    pos_parent_tensor = torch.cat(pos_parent_l, dim=0).to(device)
    pos_child_tensor = torch.cat(pos_child_l, dim=0).to(device)
    pos_min_tensor = torch.cat(pos_min_l, dim=0).to(device)
    neg_tensor = torch.cat(neg_l, dim=0).to(device)
    neg_min_tensor = torch.cat(neg_min_l, dim=0).to(device)

    # Load lineage detector and test
    model = LineageDetectorModel(device)
    checkpoint_path = "data/models/diffusion/checkpoint.pth"
    model = load_checkpoint(model, checkpoint_path)
    
    if model is None:
        print("Please train the model first using train_lineage.py")
        return
        
    model.eval()

    print("\nTesting on CIFAR-10 images (unrelated dataset):")
    print("This should produce lower similarity scores compared to COCO-based embeddings")
    test(model, pos_parent_tensor, pos_child_tensor, pos_min_tensor, 
         neg_tensor, neg_min_tensor, batch_jump=1)


if __name__ == "__main__":
    main()
