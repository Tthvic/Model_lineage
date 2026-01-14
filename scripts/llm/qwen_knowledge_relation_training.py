#!/usr/bin/env python3
"""
Qwen Knowledge Relation Learning System
Training Script: Learns the knowledge lineage relationship of Qwen models.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import random
from tqdm import tqdm
import logging
from pathlib import Path
import pickle

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Setup device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerEncoder(nn.Module):
    """Feature Encoder: Encodes 1536-dim features into 512-dim vectors"""
    def __init__(self, feat_dim=1536, d_model=512, kernel_size=3, dropout=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Use 1D convolution as feature extraction layer
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)

        # Global average pooling replacing Transformer output
        self.attn_pooling = nn.AdaptiveAvgPool1d(1)

    def compute_valid_lengths(self, x):
        mask = (x.sum(dim=-1) != 0)
        valid_lengths = mask.int().sum(dim=1)
        return valid_lengths

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        valid_lengths = self.compute_valid_lengths(x)

        # Feature projection + Normalization
        feat_proj = self.layer_norm(self.feat_proj(x))
        
        # Transpose dimensions to fit Conv1d [B, C, T] format
        feat_proj = feat_proj.permute(0, 2, 1)  
        conv_out = self.dropout(self.conv(feat_proj))  # [B, d_model, seq_len]
        
        # Global representation
        global_vec = self.attn_pooling(conv_out).squeeze(-1)

        return global_vec


class VectorRelationNet(nn.Module):
    """Relation Network: Learns the relationship between two vectors"""
    def __init__(self, embedding_dim=512):
        super(VectorRelationNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)  # Input is concatenation of two vectors
        self.relu = nn.ReLU()
        
    def forward(self, b_afea, cfea):
        # Concatenate two vectors
        h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1 = self.relu(h1)
        return h1


class TripletLoss(nn.Module):
    """Triplet Loss Function"""
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize embeddings to improve numerical stability
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Calculate pairwise distances
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        # Calculate triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return torch.mean(loss)


class QwenEmbeddingDataset(Dataset):
    """Qwen Embedding Dataset"""
    def __init__(self, emb_dir=None, split='train', test_ratio=0.2):
        # Get project root directory (3 levels up from this script)
        if emb_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            self.emb_dir = project_root / "data" / "embeddings" / "llm" / "Qwen-1.5B"
        else:
            self.emb_dir = Path(emb_dir)
        
        self.split = split
        self.test_ratio = test_ratio
        
        logger.info(f"Loading embeddings from: {self.emb_dir}")
        
        # Data directories
        self.instruct_dir = self.emb_dir / "Qwen_Instruct"  # Parent model (Instruct)
        self.finetune_dir = self.emb_dir / "Finetune"       # Child model (Finetuned)
        self.adapter_dir = self.emb_dir / "Adapter"         # Adapter model
        self.merge_dir = self.emb_dir / "Merge"             # Merge model
        self.ba_finetune_dir = self.emb_dir / "B-A" / "Finetune"  # B-A difference for Finetune
        self.ba_adapter_dir = self.emb_dir / "B-A" / "Adapter"     # B-A difference for Adapter
        self.ba_merge_dir = self.emb_dir / "B-A" / "Merge"         # B-A difference for Merge
        self.random_dir = self.emb_dir / "Qwen_random"      # Negative samples (Random)
        
        # Task list
        self.tasks = ['arc_challenge', 'gsm8k', 'hellaswag', 'humaneval', 'mgsm', 'mmlu']
        
        # Build data pairs
        self.positive_pairs, self.negative_pairs = self._build_pairs()
        
        # Split into train and test
        self._split_data()
        
        logger.info(f"Dataset initialized: {len(self.data_pairs)} pairs for {split}")

    def _get_model_files(self, directory):
        """Get all model files in the directory"""
        model_files = {}
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return model_files
            
        for task in self.tasks:
            task_dir = directory / task
            if task_dir.exists():
                files = list(task_dir.glob("*.pth"))
                if files:
                    model_files[task] = files
                    logger.info(f"Found {len(files)} files in {task_dir.relative_to(self.emb_dir)}")
        return model_files

    def _build_pairs(self):
        """Build positive and negative pairs"""
        # Get all model files
        instruct_files = self._get_model_files(self.instruct_dir)
        finetune_files = self._get_model_files(self.finetune_dir)
        adapter_files = self._get_model_files(self.adapter_dir)
        merge_files = self._get_model_files(self.merge_dir)
        ba_finetune_files = self._get_model_files(self.ba_finetune_dir)
        ba_adapter_files = self._get_model_files(self.ba_adapter_dir)
        ba_merge_files = self._get_model_files(self.ba_merge_dir)
        random_files = self._get_model_files(self.random_dir)
        
        positive_pairs = []
        negative_pairs = []
        
        # Helper to process a category
        def process_category(child_files, ba_files, category_name):
            pairs_added = 0
            for task in self.tasks:
                if (task in instruct_files and task in child_files and 
                    task in ba_files and task in random_files):
                    
                    # Get all child models for this task
                    child_models = child_files[task]
                    ba_models = ba_files[task]
                    
                    for child_model in child_models:
                        model_name = child_model.stem
                        
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
                                'child': child_model,         # Child model (Finetuned)
                                'ba': ba_model,              # B-A difference
                                'task': task,
                                'type': category_name
                            })
                            
                            # Build negative pair (use random model)
                            random_file = random.choice(random_files[task])
                            negative_pairs.append({
                                'parent': instruct_file,
                                'child': random_file,         # Random model as negative sample
                                'ba': ba_model,              # Keep B-A unchanged
                                'task': task
                            })
                            pairs_added += 1
            
            if pairs_added > 0:
                logger.info(f"Added {pairs_added} pairs for {category_name}")
            return pairs_added
        
        # Process all categories
        process_category(finetune_files, ba_finetune_files, "Finetune")
        process_category(adapter_files, ba_adapter_files, "Adapter")
        process_category(merge_files, ba_merge_files, "Merge")
        
        logger.info(f"Built {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        return positive_pairs, negative_pairs

    def _split_data(self):
        """Split data into train and test sets"""
        all_pairs = list(zip(self.positive_pairs, self.negative_pairs))
        random.Random(42).shuffle(all_pairs)  # Use fixed seed for reproducibility
        
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
                    # Get first tensor value
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
        pos_pair, neg_pair = self.data_pairs[idx]
        
        # Load positive samples
        parent_emb = self._load_embedding(pos_pair['parent'])
        child_emb = self._load_embedding(pos_pair['child'])
        ba_emb = self._load_embedding(pos_pair['ba'])
        
        # Load negative samples
        neg_child_emb = self._load_embedding(neg_pair['child'])
        neg_ba_emb = self._load_embedding(neg_pair['ba'])
        
        # Ensure consistent dimensions
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
            
        # Ensure consistent sequence length
        max_len = max(parent_emb.size(0), child_emb.size(0), ba_emb.size(0), 
                      neg_child_emb.size(0), neg_ba_emb.size(0))
        
        def pad_to_length(tensor, target_len):
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


def evaluate_model(encnet, prenet, test_loader, criterion, device):
    """Evaluate the model"""
    encnet.eval()
    prenet.eval()
    
    total_loss = 0.0
    pos_similarities = []
    neg_similarities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            parent = batch['parent'].to(device)
            child = batch['child'].to(device)
            ba = batch['ba'].to(device)
            neg_child = batch['neg_child'].to(device)
            neg_ba = batch['neg_ba'].to(device)
            
            # Encode features
            enc_parent = encnet(parent)
            enc_child = encnet(child)
            enc_ba = encnet(ba)
            enc_neg_child = encnet(neg_child)
            enc_neg_ba = encnet(neg_ba)
            
            # Compute relation representation
            pos_relation = prenet(enc_child, enc_ba)  # Positive: Child + B-A
            neg_relation = prenet(enc_neg_child, enc_neg_ba)  # Negative
            
            # Compute loss
            loss = criterion(enc_parent, pos_relation, neg_relation)
            total_loss += loss.item()
            
            # Compute similarities
            pos_sim = F.cosine_similarity(enc_parent, pos_relation).cpu().numpy()
            neg_sim = F.cosine_similarity(enc_parent, neg_relation).cpu().numpy()
            
            pos_similarities.extend(pos_sim.tolist())
            neg_similarities.extend(neg_sim.tolist())
    
    avg_loss = total_loss / len(test_loader)
    
    # Compute accuracy
    accuracy = sum(p > n for p, n in zip(pos_similarities, neg_similarities)) / len(pos_similarities)
    
    # Compute TPR and FPR
    pos_sims = np.array(pos_similarities)
    neg_sims = np.array(neg_similarities)
    threshold = 0.5
    
    tp = np.sum(pos_sims > threshold)
    fn = np.sum(pos_sims <= threshold)
    tn = np.sum(neg_sims <= threshold)
    fp = np.sum(neg_sims > threshold)
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'tpr': tpr,
        'fpr': fpr,
        'pos_similarities': pos_similarities,
        'neg_similarities': neg_similarities
    }


def train_epoch(encnet, prenet, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    encnet.train()
    prenet.train()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move data to device
        parent = batch['parent'].to(device)
        child = batch['child'].to(device)
        ba = batch['ba'].to(device)
        neg_child = batch['neg_child'].to(device)
        neg_ba = batch['neg_ba'].to(device)
        
        # Encode features
        enc_parent = encnet(parent)
        enc_child = encnet(child)
        enc_ba = encnet(ba)
        enc_neg_child = encnet(neg_child)
        enc_neg_ba = encnet(neg_ba)
        
        # Compute relation representation
        pos_relation = prenet(enc_child, enc_ba)  # Positive: Child + B-A
        neg_relation = prenet(enc_neg_child, enc_neg_ba)  # Negative
        
        # Compute loss
        loss1 = criterion(enc_parent, pos_relation, neg_relation)  # Standard triplet
        loss2 = criterion(pos_relation, enc_parent, neg_relation)  # Swapped anchor
        loss = loss1 + loss2
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training function"""
    # Setup parameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 100
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Get project root directory (3 levels up from this script)
    project_root = Path(__file__).resolve().parents[2]
    save_dir = project_root / "data" / "intermediate" / "llm" / "Qwen-1.5B" / "results"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Save directory: {save_dir}")
    
    # Create datasets and data loaders
    train_dataset = QwenEmbeddingDataset(split='train')
    test_dataset = QwenEmbeddingDataset(split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create models
    encnet = TransformerEncoder().to(device)
    prenet = VectorRelationNet().to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        list(encnet.parameters()) + list(prenet.parameters()), 
        lr=learning_rate
    )
    criterion = TripletLoss(margin=0.4)
    
    # Training history
    train_losses = []
    test_results = []
    
    best_accuracy = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(encnet, prenet, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            test_result = evaluate_model(encnet, prenet, test_loader, criterion, device)
            test_results.append(test_result)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Test Loss: {test_result['loss']:.4f}")
            logger.info(f"Accuracy: {test_result['accuracy']:.4f}")
            logger.info(f"TPR: {test_result['tpr']:.4f}, FPR: {test_result['fpr']:.4f}")
            
            # Save best model
            if test_result['accuracy'] > best_accuracy:
                best_accuracy = test_result['accuracy']
                torch.save({
                    'epoch': epoch,
                    'encnet_state_dict': encnet.state_dict(),
                    'prenet_state_dict': prenet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'test_result': test_result
                }, save_dir / 'best_model.pth')
                logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            # Save detailed results
            torch.save({
                'epoch': epoch,
                'test_result': test_result,
                'train_loss': train_loss
            }, save_dir / f'epoch_{epoch+1}_results.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'test_results': test_results,
        'best_accuracy': best_accuracy
    }
    
    with open(save_dir / 'training_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in training_history.items():
            if key == 'test_results':
                serializable_results = []
                for result in value:
                    serializable_result = {}
                    for k, v in result.items():
                        if isinstance(v, (list, np.ndarray)):
                            serializable_result[k] = v if isinstance(v, list) else v.tolist()
                        else:
                            serializable_result[k] = v
                    serializable_results.append(serializable_result)
                serializable_history[key] = serializable_results
            else:
                serializable_history[key] = value
        
        json.dump(serializable_history, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    logger.info(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
