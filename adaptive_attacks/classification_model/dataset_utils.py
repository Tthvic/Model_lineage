"""
Dataset utilities for small model adaptive attacks.
Provides unified data loading interface for different datasets.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_caltech101_loader(dataset_root: Path, selected_classes: list, batch_size: int = 64, train: bool = True):
    """
    Get Caltech-101 data loader for selected classes.
    
    Args:
        dataset_root: Root directory of Caltech-101 dataset
        selected_classes: List of class indices to include
        batch_size: Batch size
        train: Whether this is for training or testing
    
    Returns:
        DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root=str(dataset_root), transform=transform)
    
    # Filter by selected classes
    indices = [i for i, (path, label) in enumerate(dataset.samples) if label in selected_classes]
    subset = Subset(dataset, indices)
    
    # Split train/test (80/20)
    if train:
        subset_indices = indices[:int(len(indices) * 0.8)]
    else:
        subset_indices = indices[int(len(indices) * 0.8):]
    
    subset = Subset(dataset, subset_indices)
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def get_tinyimagenet_loader(dataset_root: Path, selected_classes: list, batch_size: int = 64, train: bool = True):
    """
    Get Tiny-ImageNet data loader for selected classes.
    
    Args:
        dataset_root: Root directory of Tiny-ImageNet dataset
        selected_classes: List of class indices to include
        batch_size: Batch size
        train: Whether this is for training or testing
    
    Returns:
        DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tiny-ImageNet has train/val splits
    split_dir = 'train' if train else 'val'
    dataset_path = dataset_root / split_dir
    
    dataset = ImageFolder(root=str(dataset_path), transform=transform)
    
    # Filter by selected classes
    indices = [i for i, (path, label) in enumerate(dataset.samples) if label in selected_classes]
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def get_data_loaders(dataset_name: str, dataset_root: Path, selected_classes: list, batch_size: int = 64):
    """
    Get train and test data loaders for specified dataset.
    
    Args:
        dataset_name: Name of dataset ('caltech101', 'tiny-imagenet', etc.)
        dataset_root: Root directory of dataset
        selected_classes: List of class indices to include
        batch_size: Batch size
    
    Returns:
        train_loader, test_loader
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'caltech101':
        train_loader = get_caltech101_loader(dataset_root, selected_classes, batch_size, train=True)
        test_loader = get_caltech101_loader(dataset_root, selected_classes, batch_size, train=False)
    elif dataset_name == 'tiny-imagenet':
        train_loader = get_tinyimagenet_loader(dataset_root, selected_classes, batch_size, train=True)
        test_loader = get_tinyimagenet_loader(dataset_root, selected_classes, batch_size, train=False)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    logger.info(f"Loaded {dataset_name} dataset with {len(selected_classes)} classes")
    
    return train_loader, test_loader
