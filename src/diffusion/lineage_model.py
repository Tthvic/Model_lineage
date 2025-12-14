"""
Lineage detector model for diffusion models.

This module combines the encoder and relation network for lineage verification.
"""
import torch
import torch.nn as nn
from src.diffusion.networks import Encoder, VectorRelationNet


class LineageDetectorModel(nn.Module):
    """
    Complete lineage detector model for diffusion models.
    
    Combines an encoder network and a relation network to verify model lineage
    by checking consistency between fine-tuning trajectory and knowledge evolution.
    
    Args:
        device: Device to run the model on (cuda or cpu)
        embedding_dim: Dimension of feature embeddings (default: 320 for U-Net features)
    """
    def __init__(self, device, embedding_dim=320):
        super(LineageDetectorModel, self).__init__()
        self.encoder = Encoder().to(device)
        self.relation_net = VectorRelationNet(embedding_dim).to(device)
        self.device = device

    def forward(self, parent_emb, child_emb, task_vec_emb, neg_emb, neg_task_vec_emb):
        """
        Forward pass for lineage detection.
        
        Args:
            parent_emb: Parent model embedding
            child_emb: Child model embedding (positive sample)
            task_vec_emb: Task vector embedding (parent->child parameter difference)
            neg_emb: Negative sample embedding (unrelated model)
            neg_task_vec_emb: Negative task vector embedding
        
        Returns:
            Tuple of (child_features, predicted_child_positive, predicted_child_negative)
        """
        # Encode all embeddings
        child_features = self.encoder(child_emb)
        parent_features = self.encoder(parent_emb)
        task_vec_features = self.encoder(task_vec_emb)
        
        # Predict child features from parent + task vector (positive pair)
        predicted_child_pos = self.relation_net(parent_features, task_vec_features)
        
        # Predict from negative samples
        neg_features = self.encoder(neg_emb)
        neg_task_vec_features = self.encoder(neg_task_vec_emb)
        predicted_child_neg = self.relation_net(neg_features, neg_task_vec_features)
        
        return child_features, predicted_child_pos, predicted_child_neg
