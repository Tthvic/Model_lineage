"""
Caltech-101 Dataset Loader and Utilities

This module provides dataset loading functionality for the Caltech-101 dataset
with support for custom class selection and train/test splitting.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import random

import config as cfg


# ============================================================================
# Pre-defined Class Selections
# ============================================================================

# Parent model classes: 20 models, each with 6 classes from categories 0-59
PARENT_CLASSES = cfg.PARENT_CLASSES

# Child model classes: 60 models, each with 4 classes from categories 60-100
CHILD_CLASSES = cfg.CHILD_CLASSES

# Generate 8-class and 12-class child selections
CHILD_CLASSES_8 = []
CHILD_CLASSES_12 = []

# Generate selections for different class numbers
def _generate_child_classes_multi():
    """Generate child class selections for 8 and 12 classes"""
    import random
    random.seed(7)  # Different seed for 8-class
    for i in range(20):  # 20 models for 8-class
        classes = random.sample(range(60, 101), 8)
        CHILD_CLASSES_8.append(classes)
    
    random.seed(8)  # Different seed for 12-class
    for i in range(20):  # 20 models for 12-class
        classes = random.sample(range(60, 101), 12)
        CHILD_CLASSES_12.append(classes)

_generate_child_classes_multi()


# ============================================================================
# Dataset Classes
# ============================================================================

class TargetTransform:
    """
    Transform original class labels to sequential labels (0, 1, 2, ...).
    
    This is necessary when training on a subset of classes, as we need
    to map the original class indices to a continuous range starting from 0.
    """
    
    def __init__(self, class_mapping):
        """
        Args:
            class_mapping (dict): Maps original class index to new class index
        """
        self.class_mapping = class_mapping
    
    def __call__(self, target):
        """
        Transform a target label.
        
        Args:
            target (int): Original class index
        
        Returns:
            int: Transformed class index
        """
        return self.class_mapping[target]


class Caltech101Dataset(Dataset):
    """
    Custom Caltech-101 dataset with support for class selection.
    
    This dataset allows selecting specific classes from Caltech-101 and
    automatically remaps labels to a continuous range.
    """
    
    def __init__(self, root, selected_classes=None, train=True, 
                 transform=None, target_transform=None):
        """
        Args:
            root (str): Root directory of Caltech-101 dataset
            selected_classes (list): List of class indices to include
            train (bool): Whether this is training or test data
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
        """
        self.root = root
        self.selected_classes = selected_classes
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'train' if train else 'test'
        self.data = self._load_data()
    
    def _load_data(self):
        """
        Load and filter dataset based on selected classes.
        
        Returns:
            list: List of (image, label) tuples
        """
        data_folder = os.path.join(self.root, '101_ObjectCategories')
        
        if not os.path.exists(data_folder):
            raise FileNotFoundError(
                f"Dataset not found at {data_folder}. "
                f"Please download Caltech-101 and update DATA_PATH in config.py"
            )
        
        dataset = ImageFolder(root=data_folder, transform=self.transform)
        
        if self.selected_classes is not None:
            # Filter data to include only selected classes
            selected_data = []
            for image, label in dataset:
                if label in self.selected_classes:
                    selected_data.append((image, label))
            
            if cfg.VERBOSE:
                print(f"Selected {len(selected_data)} samples from classes {self.selected_classes}")
            
            return selected_data
        else:
            # If no selection, use all data
            if self.split == 'train':
                # First 50 categories as training
                return [(img, lbl) for img, lbl in dataset if lbl < 50]
            else:
                # Last 50 categories as test
                return [(img, lbl) for img, lbl in dataset if lbl >= 50]
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of sample
        
        Returns:
            tuple: (image, label)
        """
        image, label = self.data[idx]
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)


# ============================================================================
# Data Loading Functions
# ============================================================================

def get_transforms():
    """
    Get the standard data transforms for Caltech-101.
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.NORMALIZE_MEAN, std=cfg.NORMALIZE_STD)
    ])


def get_dataset(selected_classes):
    """
    Create train and test datasets for specified classes.
    
    Args:
        selected_classes (list): List of class indices to include
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    transform = get_transforms()
    
    # Create class mapping: original class index -> new sequential index
    class_mapping = {
        original_class: new_class 
        for new_class, original_class in enumerate(selected_classes)
    }
    target_transform = TargetTransform(class_mapping)
    
    # Load full dataset with selected classes
    full_dataset = Caltech101Dataset(
        root=cfg.DATA_PATH,
        selected_classes=selected_classes,
        train=True,
        transform=transform,
        target_transform=target_transform
    )
    
    # Split into train and test sets
    train_size = int(cfg.TRAIN_RATIO * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size]
    )
    
    return train_dataset, test_dataset


def get_data_loader(selected_classes, batch_size=None):
    """
    Create data loaders for specified classes.
    
    Args:
        selected_classes (list): List of class indices to include
        batch_size (int): Batch size for data loaders (default: from config)
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    
    train_dataset, test_dataset = get_dataset(selected_classes)
    
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


# ============================================================================
# Class Selection Utilities
# ============================================================================

def generate_random_parent_classes(num_models=20, num_classes=6, seed=42):
    """
    Generate random class selections for parent models.
    
    Args:
        num_models (int): Number of parent models
        num_classes (int): Number of classes per model
        seed (int): Random seed
    
    Returns:
        list: List of class selections, each a list of class indices
    """
    random.seed(seed)
    
    selected_classes = []
    for i in range(num_models):
        classes = random.sample(range(0, 60), num_classes)
        selected_classes.append(classes)
    
    return selected_classes


def generate_random_child_classes(num_models=60, num_classes=4, 
                                  parent_classes=None, seed=6):
    """
    Generate random class selections for child models.
    
    Args:
        num_models (int): Number of child models
        num_classes (int): Number of classes per model
        parent_classes (list): Parent model class selections (to avoid overlap)
        seed (int): Random seed
    
    Returns:
        list: List of class selections, each a list of class indices
    """
    random.seed(seed)
    
    # Determine available classes (avoiding parent classes if provided)
    if parent_classes is not None:
        used_classes = set()
        for classes in parent_classes:
            used_classes.update(classes)
        available_classes = list(set(range(60, 101)) - used_classes)
    else:
        available_classes = list(range(60, 101))
    
    selected_classes = []
    all_selected = set()
    
    for i in range(num_models):
        # Select classes that haven't been used yet
        remaining = list(set(available_classes) - all_selected)
        
        if len(remaining) < num_classes:
            # Reset if we run out of classes
            remaining = available_classes
            all_selected = set()
        
        classes = random.sample(remaining, num_classes)
        selected_classes.append(classes)
        all_selected.update(classes)
    
    return selected_classes


# ============================================================================
# Testing and Utilities
# ============================================================================

def verify_dataset():
    """
    Verify that the dataset can be loaded correctly.
    
    This function is useful for debugging dataset issues.
    """
    print("Verifying Caltech-101 dataset...")
    print(f"Dataset path: {cfg.DATA_PATH}")
    
    # Test with first parent model classes
    test_classes = PARENT_CLASSES[0]
    print(f"Testing with classes: {test_classes}")
    
    try:
        train_loader, test_loader = get_data_loader(test_classes)
        
        # Get one batch
        images, labels = next(iter(train_loader))
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Batch shape: {images.shape}")
        print(f"  Label shape: {labels.shape}")
        print(f"  Label range: {labels.min().item()} - {labels.max().item()}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise


if __name__ == "__main__":
    # Verify dataset when run directly
    verify_dataset()
