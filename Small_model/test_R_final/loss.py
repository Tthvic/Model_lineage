import torch
import torch.nn as nn
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.4):
    """
    Triplet loss function to enforce anchor closer to positive than to negative.

    :param anchor: Tensor, the anchor vector (e.g., pknow).
    :param positive: Tensor, the positive vector (e.g., sumknow).
    :param negative: Tensor, the negative vector (e.g., psumknow).
    :param margin: Float, the margin parameter for triplet loss.
    :return: Scalar tensor representing the loss.
    """
    # Compute distances
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)

    # Triplet loss
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    return loss.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance between the two feature vectors
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, p=2)
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                          (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


# Example usage
if __name__ == "__main__":
    # Assume batch_size = 4
    batch_size = 4
    embedding_dim = 128
    # Fake network outputs (embeddings)
    output1 = torch.randn(batch_size, embedding_dim)  # real embeddings
    output2 = torch.randn(batch_size, embedding_dim)  # fake embeddings
    # Labels (0 means positive pair, 1 means negative pair)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    # Initialize contrastive loss
    criterion = ContrastiveLoss(margin=1.0)
    # Compute loss
    loss = criterion(output1, output2, labels)
    print("Contrastive Loss:", loss.item())