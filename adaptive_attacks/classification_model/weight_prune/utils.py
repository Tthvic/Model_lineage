"""
Utility functions for the Hierarchical Fine-Tuning Lineage Experiment
"""

import os
import torch
import torch.nn.utils.prune as prune
import numpy as np
import random
import json
import logging
from datetime import datetime


def set_all_seeds(seed):
    """
    Set seeds for all random number generators for reproducibility.
    
    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str='cuda:0'):
    """
    Get the appropriate device for training.
    
    Args:
        device_str (str): Device string (e.g., 'cuda:0', 'cpu')
    
    Returns:
        torch.device: PyTorch device object
    """
    if 'cuda' in device_str and torch.cuda.is_available():
        return torch.device(device_str)
    else:
        print(f"CUDA not available, using CPU instead")
        return torch.device('cpu')


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        accuracy (float): Current accuracy
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load a training checkpoint.
    
    Args:
        filepath (str): Path to checkpoint file
        model: PyTorch model
        optimizer: PyTorch optimizer (optional)
    
    Returns:
        dict: Checkpoint information
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_experiment_config(config_dict, filepath):
    """
    Save experiment configuration to a JSON file.
    
    Args:
        config_dict (dict): Configuration dictionary
        filepath (str): Path to save configuration
    """
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)


def load_experiment_config(filepath):
    """
    Load experiment configuration from a JSON file.
    
    Args:
        filepath (str): Path to configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples (for batch processing)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if training should stop.
        
        Args:
            score (float): Current metric value
        
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        """Check if score is an improvement"""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def format_time(seconds):
    """
    Format seconds into a readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_summary(model, input_size=(1, 3, 128, 128)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size (tuple): Input tensor size
    """
    print("\n" + "=" * 70)
    print("Model Summary")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 70 + "\n")


def get_gpu_memory_usage():
    """
    Get current GPU memory usage.
    
    Returns:
        dict: Dictionary with memory statistics
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
        }
    return {'allocated': 0, 'cached': 0}


def log_metrics(metrics, filepath, mode='a'):
    """
    Log metrics to a CSV file.
    
    Args:
        metrics (dict): Dictionary of metrics to log
        filepath (str): Path to log file
        mode (str): File open mode ('a' for append, 'w' for write)
    """
    import csv
    
    file_exists = os.path.exists(filepath)
    
    with open(filepath, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        
        if not file_exists or mode == 'w':
            writer.writeheader()
        
        writer.writerow(metrics)


def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup logger that writes to both file and console.
    
    Args:
        name (str): Logger name
        log_file (str): Path to log file
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # File handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def prune_model(model, pruning_rate):
    """
    Prune a model using manual L1 magnitude pruning (threshold-based).
    
    This method directly sets weights to zero based on their absolute magnitude,
    without using PyTorch's pruning API. All Conv2d and Linear layers are pruned
    uniformly with the same rate.
    
    Args:
        model: PyTorch model to prune
        pruning_rate (float): Pruning rate (0.0 to 1.0), e.g., 0.7 means 70% of weights become zero
    
    Returns:
        model: Pruned model (weights permanently modified)
    """
    def manual_prune_module(module, amount):
        """Apply magnitude-based pruning to a single module"""
        weight = module.weight
        flat = weight.view(-1)
        k = int(amount * flat.numel())
        if k <= 0:
            return
        # Find threshold: the maximum absolute value among k smallest weights
        threshold = torch.topk(flat.abs(), k, largest=False).values.max()
        # Create mask: keep weights with abs value > threshold
        mask = weight.abs() > threshold
        # Permanently set pruned weights to zero
        with torch.no_grad():
            weight.mul_(mask.float())
    
    # Apply pruning to all Conv2d and Linear layers
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            manual_prune_module(module, pruning_rate)
    
    return model


def make_pruning_permanent(model):
    """
    Make pruning permanent (no-op when using manual pruning method).
    
    Since the manual pruning method already modifies weights directly,
    this function is kept for compatibility but does nothing.
    
    Args:
        model: Pruned PyTorch model
    
    Returns:
        model: Same model (already permanently pruned)
    """
    # Manual pruning already modifies weights permanently
    # This function is kept for API compatibility
    return model


def calculate_sparsity(model):
    """
    Calculate the sparsity of a model (percentage of zero weights).
    Works with manual magnitude-based pruning.
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Sparsity percentage
    """
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            # For manual pruning, directly check weight tensor
            weight = module.weight
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
    
    if total_params == 0:
        return 0.0
    
    sparsity = 100.0 * zero_params / total_params
    return sparsity


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # Test format_time
    print(f"Formatted time: {format_time(3665)}")
    
    print("All tests passed!")
