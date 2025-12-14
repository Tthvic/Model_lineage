import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
import numpy as np

class RemappedDataset(Dataset):
    def __init__(self, dataset, selected_classes, class_mapping):
        self.dataset = dataset
        self.selected_classes = set(selected_classes)
        self.class_mapping = class_mapping
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in self.selected_classes]

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        image, label = self.dataset[original_idx]
        return image, self.class_mapping[label]

    def __len__(self):
        return len(self.indices)

def get_transforms(image_size=224, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataset(config, split='train', selected_classes=None):
    name = config['dataset']['name'].lower()
    root = config['dataset']['root']
    image_size = config['dataset'].get('image_size', 224)
    
    is_training = (split == 'train')
    transform = get_transforms(image_size, is_training)
    
    os.makedirs(root, exist_ok=True)

    if name == 'caltech101':
        # Check for existing data in various common structures
        possible_paths = [
            os.path.join(root, 'caltech-101', '101_ObjectCategories'),
            os.path.join(root, '101_ObjectCategories'),
            root
        ]
        
        target_dir = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path) and '101_ObjectCategories' in path:
                 # Verify it actually has subdirectories (classes)
                 if any(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)):
                     target_dir = path
                     break
        
        if target_dir:
             print(f"Loading Caltech101 from local path: {target_dir}")
             base_dataset = ImageFolder(root=target_dir, transform=transform)
        else:
             print(f"Local Caltech101 not found at {root}. Attempting download...")
             # Fallback to standard download if custom path doesn't exist
             # Note: Standard Caltech101 might not work with ImageFolder logic directly without download
             base_dataset = datasets.Caltech101(root=root, download=True, transform=transform)

    elif name == 'cifar10':
        base_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform) # Load all for filtering
        
    elif name == 'flowers':
        # Assuming standard structure or torchvision
        base_dataset = datasets.Flowers102(root=root, split='train' if is_training else 'test', download=True, transform=transform)
        
    else:
        # Generic ImageFolder fallback
        base_dataset = ImageFolder(root=root, transform=transform)

    # Filter classes if specified
    if selected_classes:
        # Create mapping: original_class -> 0, 1, 2...
        class_mapping = {orig: new for new, orig in enumerate(selected_classes)}
        dataset = RemappedDataset(base_dataset, selected_classes, class_mapping)
        return dataset
    else:
        return base_dataset

def get_dataloader(config, split='train', selected_classes=None):
    dataset = get_dataset(config, split, selected_classes)
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset'].get('num_workers', 4)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        num_workers=num_workers
    )
