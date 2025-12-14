import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
    Feature Encoder: Encodes 1536-dim features into 512-dim vectors.
    Uses 1D convolution and attention pooling.
    """
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
        """Compute valid lengths of sequences based on non-zero elements."""
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
    """
    Relation Network: Learns the relationship between two vectors.
    """
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
    """
    Triplet Loss Function.
    """
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
