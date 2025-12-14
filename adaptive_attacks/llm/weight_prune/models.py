#!/usr/bin/env python3
"""
Model architectures for LLM lineage detection
Includes encoder and relation network for learning knowledge inheritance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    """
    Feature encoder: Encodes 1536-dim features to 512-dim vectors
    Uses 1D convolution for feature extraction instead of full Transformer.
    """
    def __init__(self, feat_dim=1536, d_model=512, kernel_size=3, dropout=0.1):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Use 1D convolution as feature extraction layer
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)

        # Global average pooling to replace Transformer output
        self.attn_pooling = nn.AdaptiveAvgPool1d(1)

    def compute_valid_lengths(self, x):
        """Compute valid sequence lengths (non-zero entries)"""
        mask = (x.sum(dim=-1) != 0)
        valid_lengths = mask.int().sum(dim=1)
        return valid_lengths

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, feat_dim]
        
        Returns:
            global_vec: Encoded features [batch_size, d_model]
        """
        batch_size, seq_len, feat_dim = x.shape
        valid_lengths = self.compute_valid_lengths(x)

        # Feature projection + normalization
        feat_proj = self.layer_norm(self.feat_proj(x))
        
        # Transpose to [B, C, T] format for Conv1d
        feat_proj = feat_proj.permute(0, 2, 1)  
        conv_out = self.dropout(self.conv(feat_proj))  # [B, d_model, seq_len]
        
        # Global representation via adaptive pooling
        global_vec = self.attn_pooling(conv_out).squeeze(-1)

        return global_vec


class VectorRelationNet(nn.Module):
    """
    Relation network: Learns the relationship between two vectors
    Predicts parent embedding from (child, task_vector) pair.
    """
    def __init__(self, embedding_dim=512):
        super(VectorRelationNet, self).__init__()
        # Input is concatenation of two vectors
        self.fc1 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.relu = nn.ReLU()
        
    def forward(self, b_afea, cfea):
        """
        Args:
            b_afea: Child model features [batch_size, embedding_dim]
            cfea: Task vector (B-A difference) features [batch_size, embedding_dim]
        
        Returns:
            h1: Predicted parent features [batch_size, embedding_dim]
        """
        # Concatenate two vectors
        h1 = self.fc1(torch.cat([b_afea, cfea], dim=-1))
        h1 = self.relu(h1)
        return h1


class TripletLoss(nn.Module):
    """
    Triplet loss function for learning embeddings
    Encourages positive pairs to be closer than negative pairs.
    """
    def __init__(self, margin=0.4):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Anchor embeddings [batch_size, dim]
            positive: Positive embeddings [batch_size, dim]
            negative: Negative embeddings [batch_size, dim]
        
        Returns:
            loss: Triplet loss value
        """
        # Normalize embeddings for numerical stability
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute pairwise distances
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return torch.mean(loss)
