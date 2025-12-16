"""
Tiny-ImageNet Dataset Loader

Provides dataset loading functionality for Tiny-ImageNet with custom class selection.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random

import config as cfg


class TinyImageNetDataset(Dataset):
    """
    Tiny-ImageNet dataset with support for class selection.
    """
    
    def __init__(self, root, selected_classes=None, train=True, 
                 transform=None, target_transform=None, max_samples_per_class=None):
        """
        Args:
            root (str): Root directory of Tiny-ImageNet dataset
            selected_classes (list): List of class indices to include
            train (bool): Whether this is training or test data
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
            max_samples_per_class (int): Maximum samples per class (None = use all)
        """
        self.root = root
        self.selected_classes = selected_classes
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'train' if train else 'val'
        self.max_samples_per_class = max_samples_per_class
        self.data = self._load_data()
    
    def _load_data(self):
        """Load and filter dataset based on selected classes"""
        data_folder = os.path.join(self.root, self.split)
        
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"Tiny-ImageNet not found at {data_folder}. "
                f"Please download and update TINY_IMAGENET_PATH in config.py"
            )
        
        dataset = ImageFolder(root=data_folder, transform=self.transform)
        
        if self.selected_classes is not None:
            # Filter data to include only selected classes
            selected_data = []
            class_counts = {cls: 0 for cls in self.selected_classes}
            skipped_count = 0
            
            for idx in range(len(dataset)):
                try:
                    image, label = dataset[idx]
                    if label in self.selected_classes:
                        # Limit samples per class if specified
                        if self.max_samples_per_class is None or class_counts[label] < self.max_samples_per_class:
                            selected_data.append((image, label))
                            class_counts[label] += 1
                except Exception as e:
                    skipped_count += 1
                    continue
            
            if cfg.VERBOSE:
                if self.max_samples_per_class:
                    print(f"Selected {len(selected_data)} samples (max {self.max_samples_per_class} per class) from Tiny-ImageNet classes {self.selected_classes}")
                else:
                    print(f"Selected {len(selected_data)} samples from Tiny-ImageNet classes {self.selected_classes}")
                if skipped_count > 0:
                    print(f"Skipped {skipped_count} corrupted images")
            
            return selected_data
        else:
            # Return all data
            selected_data = []
            for idx in range(len(dataset)):
                try:
                    selected_data.append(dataset[idx])
                except:
                    continue
            return selected_data
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        image, label = self.data[idx]
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)


class TargetTransform:
    """Transform original class labels to sequential labels"""
    
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping
    
    def __call__(self, target):
        return self.class_mapping[target]


def get_transforms():
    """Get standard transforms for Tiny-ImageNet (64x64 -> 128x128)"""
    return transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),  # Resize from 64x64 to 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD)
    ])


def get_tinyimagenet_dataset(selected_classes, max_samples_per_class=None):
    """
    Create train and test datasets from Tiny-ImageNet.
    
    Args:
        selected_classes (list): List of class indices to include (0-199)
        max_samples_per_class (int): Maximum samples per class (None = use all)
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    transform = get_transforms()
    
    # Create class mapping
    class_mapping = {
        original_class: new_class 
        for new_class, original_class in enumerate(selected_classes)
    }
    target_transform = TargetTransform(class_mapping)
    
    # Load full dataset with selected classes
    full_dataset = TinyImageNetDataset(
        root=cfg.TINY_IMAGENET_PATH,
        selected_classes=selected_classes,
        train=True,
        transform=transform,
        target_transform=target_transform,
        max_samples_per_class=max_samples_per_class
    )
    
    # Split into train and test sets with fixed seed for reproducibility
    train_size = int(cfg.TRAIN_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Use a generator with fixed seed for consistent splits
    generator = torch.Generator().manual_seed(888)
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=generator
    )
    
    return train_dataset, test_dataset


def get_tinyimagenet_loader(selected_classes, batch_size=None, max_samples_per_class=None):
    """
    Create data loaders for Tiny-ImageNet.
    
    Args:
        selected_classes (list): List of class indices to include
        batch_size (int): Batch size for data loaders
        max_samples_per_class (int): Maximum samples per class (None = use all)
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    
    train_dataset, test_dataset = get_tinyimagenet_dataset(selected_classes, max_samples_per_class)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, test_loader


def generate_child_classes(num_models, num_classes, seed):
    """
    Generate random class selections for child models from Tiny-ImageNet.
    
    Args:
        num_models (int): Number of child models
        num_classes (int): Number of classes per model
        seed (int): Random seed
    
    Returns:
        list: List of class selections
    """
    random.seed(seed)
    
    child_classes = []
    for i in range(num_models):
        # Tiny-ImageNet has 200 classes (0-199)
        classes = random.sample(range(0, 200), num_classes)
        child_classes.append(classes)
    
    return child_classes


# Pre-generate child class selections for reproducibility
# Rebuttal experiment: 6→6 with 40 models (20 ResNet-50 + 20 MobileNet)
CHILD_CLASSES_6_REBUTTAL = generate_child_classes(40, 6, seed=103)

# Revised experiment: Mixed tasks with 120 models total
# 6→6: 24 models (12 ResNet-50 + 12 MobileNet) - 20%
# 6→8: 36 models (18 ResNet-50 + 18 MobileNet) - 30%
# 6→12: 60 models (30 ResNet-50 + 30 MobileNet) - 50%
CHILD_CLASSES_6 = generate_child_classes(24, 6, seed=104)
CHILD_CLASSES_8 = generate_child_classes(36, 8, seed=101)
CHILD_CLASSES_12 = generate_child_classes(60, 12, seed=102)

# Legacy 4-class for testing
CHILD_CLASSES_4 = generate_child_classes(20, 4, seed=100)
