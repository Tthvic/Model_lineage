#!/usr/bin/env python3
"""
Dataset for LLM embedding-based lineage learning
Loads pre-computed embeddings and constructs positive/negative pairs.
"""

import os
import torch
import random
import logging
from pathlib import Path
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class QwenEmbeddingDataset(Dataset):
    """
    Qwen embedding dataset for lineage learning
    Constructs positive pairs (parent-child) and negative pairs (parent-random).
    """
    def __init__(self, emb_dir, split='train', test_ratio=0.2, tasks=None):
        """
        Args:
            emb_dir: Root directory containing embeddings
            split: 'train' or 'test'
            test_ratio: Ratio of test data
            tasks: List of tasks to include
        """
        self.emb_dir = Path(emb_dir)
        self.split = split
        self.test_ratio = test_ratio
        
        # Data directories
        self.instruct_dir = self.emb_dir / "Qwen_Instruct"  # Parent model (Instruct)
        self.finetune_dir = self.emb_dir / "Finetune"       # Child model (fine-tuned)
        self.ba_dir = self.emb_dir / "B-A" / "Finetune"     # B-A difference
        self.random_dir = self.emb_dir / "Qwen_random"      # Negative samples (random)
        
        # Task list
        if tasks is None:
            self.tasks = ['arc_challenge', 'gsm8k', 'hellaswag', 'humaneval', 'mgsm', 'mmlu']
        else:
            self.tasks = tasks
        
        # Build data pairs
        self.positive_pairs, self.negative_pairs = self._build_pairs()
        
        # Split into train and test
        self._split_data()
        
        logger.info(f"Dataset initialized: {len(self.data_pairs)} pairs for {split}")

    def _get_model_files(self, directory):
        """Get all model files in the directory organized by task"""
        model_files = {}
        for task in self.tasks:
            task_dir = directory / task
            if task_dir.exists():
                files = list(task_dir.glob("*.pth"))
                if files:
                    model_files[task] = files
        return model_files

    def _build_pairs(self):
        """Build positive and negative sample pairs"""
        # Get all model files
        instruct_files = self._get_model_files(self.instruct_dir)
        finetune_files = self._get_model_files(self.finetune_dir)
        ba_files = self._get_model_files(self.ba_dir)
        random_files = self._get_model_files(self.random_dir)
        
        positive_pairs = []
        negative_pairs = []
        
        for task in self.tasks:
            if (task in instruct_files and task in finetune_files and 
                task in ba_files and task in random_files):
                
                # Get all fine-tuned models for this task
                finetune_models = finetune_files[task]
                ba_models = ba_files[task]
                
                for ft_model in finetune_models:
                    model_name = ft_model.stem
                    
                    # Find corresponding B-A file
                    ba_model = None
                    for ba_file in ba_models:
                        if ba_file.stem == model_name:
                            ba_model = ba_file
                            break
                    
                    if ba_model is not None:
                        # Build positive pair
                        instruct_file = random.choice(instruct_files[task])
                        positive_pairs.append({
                            'parent': instruct_file,      # Parent model (Instruct)
                            'child': ft_model,            # Child model (fine-tuned)
                            'ba': ba_model,              # B-A difference
                            'task': task
                        })
                        
                        # Build negative pair (use random model)
                        random_file = random.choice(random_files[task])
                        negative_pairs.append({
                            'parent': instruct_file,
                            'child': random_file,         # Random model as negative sample
                            'ba': ba_model,              # Keep B-A same
                            'task': task
                        })
        
        logger.info(f"Built {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        return positive_pairs, negative_pairs

    def _split_data(self):
        """Split data into train and test sets"""
        all_pairs = list(zip(self.positive_pairs, self.negative_pairs))
        random.shuffle(all_pairs)
        
        split_idx = int(len(all_pairs) * (1 - self.test_ratio))
        
        if self.split == 'train':
            self.data_pairs = all_pairs[:split_idx]
        else:  # test
            self.data_pairs = all_pairs[split_idx:]

    def _load_embedding(self, file_path):
        """Load embedding file"""
        try:
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                # If it's a dict, try to get embeddings field
                if 'embeddings' in data:
                    return data['embeddings']
                elif 'hidden_states' in data:
                    return data['hidden_states']
                else:
                    # Get the first tensor value
                    return next(iter(data.values()))
            else:
                return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            # Return zero vector as fallback
            return torch.zeros(1, 1536)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """Get a data sample"""
        pos_pair, neg_pair = self.data_pairs[idx]
        
        # Load positive samples
        parent_emb = self._load_embedding(pos_pair['parent'])
        child_emb = self._load_embedding(pos_pair['child'])
        ba_emb = self._load_embedding(pos_pair['ba'])
        
        # Load negative samples
        neg_child_emb = self._load_embedding(neg_pair['child'])
        neg_ba_emb = self._load_embedding(neg_pair['ba'])
        
        # Ensure dimension consistency
        if parent_emb.dim() == 1:
            parent_emb = parent_emb.unsqueeze(0)
        if child_emb.dim() == 1:
            child_emb = child_emb.unsqueeze(0)
        if ba_emb.dim() == 1:
            ba_emb = ba_emb.unsqueeze(0)
        if neg_child_emb.dim() == 1:
            neg_child_emb = neg_child_emb.unsqueeze(0)
        if neg_ba_emb.dim() == 1:
            neg_ba_emb = neg_ba_emb.unsqueeze(0)
            
        # Ensure sequence length consistency
        max_len = max(parent_emb.size(0), child_emb.size(0), ba_emb.size(0), 
                      neg_child_emb.size(0), neg_ba_emb.size(0))
        
        def pad_to_length(tensor, target_len):
            """Pad or truncate tensor to target length"""
            if tensor.size(0) < target_len:
                padding = torch.zeros(target_len - tensor.size(0), tensor.size(1))
                tensor = torch.cat([tensor, padding], dim=0)
            elif tensor.size(0) > target_len:
                tensor = tensor[:target_len]
            return tensor
        
        parent_emb = pad_to_length(parent_emb, max_len)
        child_emb = pad_to_length(child_emb, max_len)
        ba_emb = pad_to_length(ba_emb, max_len)
        neg_child_emb = pad_to_length(neg_child_emb, max_len)
        neg_ba_emb = pad_to_length(neg_ba_emb, max_len)
        
        return {
            'parent': parent_emb,           # Parent model embedding
            'child': child_emb,             # Child model embedding
            'ba': ba_emb,                   # B-A difference embedding
            'neg_child': neg_child_emb,     # Negative sample child model
            'neg_ba': neg_ba_emb,          # Negative sample B-A difference
            'task': pos_pair['task']
        }
