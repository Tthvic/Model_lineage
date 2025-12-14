"""
Loss functions for training the diffusion model lineage detector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for measuring similarity between embeddings.
    
    Args:
        margin: Margin for negative pairs
    """
    def __init__(self, margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        Args:
            output1: First embedding (batch_size, embedding_dim)
            output2: Second embedding (batch_size, embedding_dim)  
            label: 1 for positive pairs, 0 for negative pairs
        
        Returns:
            Contrastive loss value
        """
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        loss = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


class BinaryClassificationLoss(nn.Module):
    """
    Binary classification loss based on Binary Cross Entropy.
    
    Args:
        reduction: Specifies the reduction to apply to the output: 'mean', 'sum' or 'none'
    """
    def __init__(self, reduction='mean'):
        super(BinaryClassificationLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def forward(self, logits, labels):
        """
        Args:
            logits: Model output probabilities in range [0, 1]
            labels: Ground truth labels (0 or 1)
        
        Returns:
            Binary cross entropy loss
        """
        logits = torch.clamp(logits, min=1e-7, max=1 - 1e-7)
        return self.bce_loss(logits, labels)
