import os
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class LineageEmbeddingDataset(Dataset):
    """
    Dataset for LLM Lineage Embeddings.
    Loads embeddings for Parent, Child, B-A difference, and Negative samples.
    """
    def __init__(self, config, split='train'):
        """
        Args:
            config (dict): Configuration dictionary containing data paths and settings.
            split (str): 'train' or 'test'.
        """
        self.config = config
        self.split = split
        self.test_ratio = config['data'].get('test_ratio', 0.2)
        
        # Construct paths
        self.root_dir = Path(config['data']['root_dir'])
        self.emb_dir = self.root_dir / config['data']['embedding_dir']
        
        self.instruct_dir = self.emb_dir / config['data']['instruct_dir']
        self.finetune_dir = self.emb_dir / config['data']['finetune_dir']
        self.ba_dir = self.emb_dir / config['data']['ba_dir']
        self.random_dir = self.emb_dir / config['data']['random_dir']
        
        # Add Adapter and Merge dirs if they exist
        self.adapter_dir = self.emb_dir / "Adapter"
        self.merge_dir = self.emb_dir / "Merge"
        
        self.tasks = config['data']['tasks']
        
        # Build data pairs
        self.positive_pairs, self.negative_pairs = self._build_pairs()
        
        # Split into train and test
        self._split_data()
        
        logger.info(f"Dataset initialized: {len(self.data_pairs)} pairs for {split}")

    def _get_model_files(self, directory):
        """Get all model files in the directory."""
        model_files = {}
        if not directory.exists():
            return model_files
            
        for task in self.tasks:
            task_dir = directory / task
            if task_dir.exists():
                files = list(task_dir.glob("*.pth"))
                if files:
                    model_files[task] = files
        return model_files

    def _build_pairs(self):
        """Build positive and negative pairs."""
        # Get all model files
        instruct_files = self._get_model_files(self.instruct_dir)
        finetune_files = self._get_model_files(self.finetune_dir)
        adapter_files = self._get_model_files(self.adapter_dir)
        merge_files = self._get_model_files(self.merge_dir)
        random_files = self._get_model_files(self.random_dir)
        
        # B-A files are nested: B-A/Adapter, B-A/Finetune, B-A/Merge
        ba_adapter_files = self._get_model_files(self.ba_dir / "Adapter")
        ba_finetune_files = self._get_model_files(self.ba_dir / "Finetune")
        ba_merge_files = self._get_model_files(self.ba_dir / "Merge")
        
        positive_pairs = []
        negative_pairs = []
        
        for task in self.tasks:
            # Check if we have necessary components for this task
            has_instruct = task in instruct_files
            has_random = task in random_files
            
            if not (has_instruct and has_random):
                # logger.warning(f"Task {task} missing instruct or random files. Skipping.")
                continue

            # Helper to process a category
            def process_category(files_dict, ba_files_dict, category_name):
                if task in files_dict and task in ba_files_dict:
                    for model_file in files_dict[task]:
                        model_name = model_file.stem
                        # Find corresponding B-A file
                        ba_file = None
                        for f in ba_files_dict[task]:
                            if f.stem == model_name:
                                ba_file = f
                                break
                        
                        if ba_file:
                            positive_pairs.append({
                                'parent': random.choice(instruct_files[task]),
                                'child': model_file,
                                'ba': ba_file,
                                'task': task,
                                'type': category_name
                            })
                            
                            # Negative pair: Random Child + Correct BA (or Random BA)
                            # Here we use Random Child + Correct BA to force the relation net to distinguish child
                            negative_pairs.append({
                                'neg_child': random.choice(random_files[task]),
                                'neg_ba': ba_file 
                            })

            # Process all categories
            process_category(adapter_files, ba_adapter_files, "Adapter")
            process_category(finetune_files, ba_finetune_files, "Finetune")
            process_category(merge_files, ba_merge_files, "Merge")
                        
        logger.info(f"Built {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        return positive_pairs, negative_pairs

    def _split_data(self):
        """Split data into train and test sets."""
        total_len = len(self.positive_pairs)
        indices = list(range(total_len))
        # Use fixed seed for reproducibility of split
        random.Random(42).shuffle(indices)
        
        split_idx = int(total_len * (1 - self.test_ratio))
        
        if self.split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        self.data_pairs = [self.positive_pairs[i] for i in self.indices]
        self.neg_pairs = [self.negative_pairs[i] for i in self.indices]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pos_item = self.data_pairs[idx]
        neg_item = self.neg_pairs[idx]
        
        # Load embeddings
        # Note: We load on CPU first
        parent_emb = torch.load(pos_item['parent'], map_location='cpu')
        child_emb = torch.load(pos_item['child'], map_location='cpu')
        ba_emb = torch.load(pos_item['ba'], map_location='cpu')
        
        neg_child_emb = torch.load(neg_item['neg_child'], map_location='cpu')
        neg_ba_emb = torch.load(neg_item['neg_ba'], map_location='cpu')
        
        # Ensure they are 1D or 2D tensors. If 2D (1, Dim), squeeze it.
        if parent_emb.dim() > 1: parent_emb = parent_emb.squeeze(0)
        if child_emb.dim() > 1: child_emb = child_emb.squeeze(0)
        if ba_emb.dim() > 1: ba_emb = ba_emb.squeeze(0)
        if neg_child_emb.dim() > 1: neg_child_emb = neg_child_emb.squeeze(0)
        if neg_ba_emb.dim() > 1: neg_ba_emb = neg_ba_emb.squeeze(0)
        
        return {
            'parent': parent_emb,
            'child': child_emb,
            'ba': ba_emb,
            'neg_child': neg_child_emb,
            'neg_ba': neg_ba_emb,
            'task': pos_item['task']
        }
