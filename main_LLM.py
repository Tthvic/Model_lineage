# Define training process
import os

import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from network import *
from dataloader import create_dataloader,EmbeddingDataset
from loss import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize embeddings to improve numerical stability
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute pairwise distances
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return torch.mean(loss)


   # Optionally, print or log the loss to track training progress

def test(encnet, prenet, test_loader, contrastive_loss_fn, epoch, name):
    encnet.eval()
    prenet.eval()
    total_loss = 0.0
    num_batches = 0
    pos_similarities = []
    neg_similarities = []

    with torch.no_grad():
        for i, (data, falsedata) in enumerate(test_loader):
            cfea, pfea, b_afea = data[0].to('cuda:0'), data[1].to('cuda:0'), data[2].to('cuda:0')
            encb_afea, enccfea, encpfea = encnet(b_afea), encnet(cfea), encnet(pfea)
            fcfea, fpfea, fb_afea = falsedata[0].to('cuda:0'), falsedata[1].to('cuda:0'), falsedata[2].to('cuda:0')

            encb_afea = encnet(b_afea)
            enccfea = encnet(cfea)
            encpfea = encnet(pfea)
            fencb_afea = encnet(fb_afea)
            fenccfea = encnet(fcfea)

            pknow = encpfea
            sumknow = prenet(encb_afea, enccfea)
            fsumknow = prenet(fencb_afea, fenccfea)

            loss = contrastive_loss_fn(pknow, sumknow, fsumknow)
            total_loss += loss.item()

            pos_similarity = torch.nn.functional.cosine_similarity(pknow, sumknow).cpu().numpy().astype(
                np.float32).tolist()
            neg_similarity = torch.nn.functional.cosine_similarity(pknow, fsumknow).cpu().numpy().astype(
                np.float32).tolist()

            pos_similarities.extend(pos_similarity)
            neg_similarities.extend(neg_similarity)

            num_batches += 1

    avg_test_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # Calculate probability (accuracy)
    probability = sum(p > n for p, n in zip(pos_similarities, neg_similarities)) / len(pos_similarities)

    # Calculate TPR and FPR
    # Convert similarities to numpy arrays for easier calculation
    pos_sims = np.array(pos_similarities)
    neg_sims = np.array(neg_similarities)

    # You can adjust this threshold based on your needs
    threshold = 0.5  # or calculate optimal threshold using ROC analysis

    # True Positives: Positive pairs with similarity > threshold
    tp = np.sum(pos_sims > threshold)
    # False Negatives: Positive pairs with similarity <= threshold
    fn = np.sum(pos_sims <= threshold)
    # True Negatives: Negative pairs with similarity <= threshold
    tn = np.sum(neg_sims <= threshold)
    # False Positives: Negative pairs with similarity > threshold
    fp = np.sum(neg_sims > threshold)

    # Calculate rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Probability (Accuracy): {probability:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    # Save results
    saved_dict = {
        "pos_similarities": pos_similarities,
        "neg_similarities": neg_similarities,
        "probability": probability,
        "tpr": tpr,
        "fpr": fpr,
        "threshold": threshold,
        "avg_test_loss": avg_test_loss
    }

    torch.save(saved_dict, f'./{epoch}.pth')

    return avg_test_loss, (pos_similarities, neg_similarities), (tpr, fpr)


def main():

    # dataset = EmbeddingDataset()  # Create dataset instance
    index = 0  # Assume batch size 16 and current batch start index
    batch_size = 40
    encnet = TransformerEncoder()
    encnet = encnet.to('cuda:0')
    prenet = VectorRelationNet(256)
    prenet = prenet.to('cuda:0')

    train_loader,test_loader=create_dataloader()
    # Initialize optimizer for the networks
    optimizer = optim.Adam(list(encnet.parameters()) + list(prenet.parameters()), lr=0.0001)
    triplet_loss_fn = TripletLoss(margin=0.4)
    num_epochs = 1000
    for epoch in range(num_epochs):
        encnet.train()  # Set to training mode
        prenet.train()
        for i, (data, falsedata) in enumerate(train_loader):
            cfea,pfea,b_afea=data[0].to('cuda:0'),data[1].to('cuda:0'),data[2].to('cuda:0')
            # b_afea,cfea,pfea=data[2].to('cuda:0'),data[0].to('cuda:0'),data[1].to('cuda:0')
            encb_afea,enccfea,encpfea=encnet(b_afea),encnet(cfea),encnet(pfea)
            pknow=encpfea
            sumknow=prenet(encb_afea,enccfea)
            fcfea,fpfea,fb_afea=falsedata[0].to('cuda:0'),falsedata[1].to('cuda:0'),falsedata[2].to('cuda:0')
            # fb_afea,fcfea,fpfea=falsedata[2].to('cuda:0'),falsedata[0].to('cuda:0'),falsedata[1].to('cuda:0')
            fencb_afea, fenccfea=encnet(fb_afea), encnet(fcfea)
            fsumknow=prenet(fencb_afea,fenccfea)
            loss1 = triplet_loss_fn(pknow, sumknow, fsumknow) 
            loss3 = triplet_loss_fn(sumknow, pknow, fsumknow) 
            loss = loss1 + loss3
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%1==0:
            test(encnet, prenet, test_loader, triplet_loss_fn , epoch, 'LLM')        # Optionally, print or log the loss to track training progress
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Step [{index // batch_size + 1}/, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()