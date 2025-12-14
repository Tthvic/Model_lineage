"""
Neural network architectures for diffusion model lineage detection.

This module contains the encoder network and relation network used in the lineage detector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    """
    Encoder network to process diffusion model U-Net embeddings.
    
    Takes U-Net features (320 channels, 32x32) and encodes them into a 320-dimensional vector.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=320, out_channels=320, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(320)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 320, 32, 32)
        
        Returns:
            Encoded features of shape (batch_size, 320)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.gap(x)
        x = self.flatten(x)
        return x


class VectorRelationNet(nn.Module):
    """
    Relation network to predict knowledge evolution from fine-tuning trajectory.
    
    Takes parent features and task vector, predicts child features.
    """
    def __init__(self, embedding_dim=320):
        super(VectorRelationNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward(self, parent_features, task_vector):
        """
        Args:
            parent_features: Features from parent model (batch_size, embedding_dim)
            task_vector: Task vector (parameter difference) (batch_size, embedding_dim)
        
        Returns:
            Predicted child features (batch_size, embedding_dim)
        """
        combined = torch.cat([parent_features, task_vector], dim=-1)
        predicted_child = self.fc(combined)
        return predicted_child
