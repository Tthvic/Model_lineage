import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.task_vectors import *
from diffusion_dataset import *
import numpy as np
from torch.optim.lr_scheduler import StepLR
from network import *
# from dataloader import create_dataloader,EmbeddingDataset
from loss import *
from dataset import *
import datetime
from model import  *
from mainszy import *
def add_noise_to_model(model, noise_ratio=0.20):
    """
               
    noise_ratio:                 
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data
                mean_abs = param_data.abs().mean()
                noise_std = mean_abs * noise_ratio
                noise = torch.randn_like(param_data) * noise_std
                param.add_(noise)
    return model
def main():
    chiname = "yuanbit/max-15-1e-6-1500"
    chiname ="/data/1/fintune_sd/stable_diffusion_finetuned/szy1/merged_sd_model10000/"
    chiname ="/data/1/fintune_sd/stable_diffusion_finetuned/szybooth1/"
    chiname = "KwongYung/trained-sd2"
    chiname= "stabilityai/stable-diffusion-2"
    chiname= "/data/1/fintune_sd/stable_diffusion_finetuned/szybooth2"
    # dataset = Diff_Dataset(pos_chiname=chiname,neg_name="bguisard/stable-diffusion-nano-2-1")
    dataset = Diff_Dataset(pos_chiname="/data/1/fintune_sd/stable_diffusion_finetuned/szybooth1/",neg_name="/data/1/fintune_sd/stable_diffusion_finetuned/szybooth-chi1/",neg_chiname="/data/1/fintune_sd/stable_diffusion_finetuned/szybooth-chi3/")
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    size = len(dataset)
    pos_fa_l = []
    pos_chi_l = []
    pos_min_l = []
    neg_l = []
    neg_min_l = []
    device = torch.device('cuda:5')
    # model.load()
    model = Model(device)
    # model = load_checkpoint(model,"/data/1/lineage-checkpoints/cp1-20250327-080710/78.pth")
    # model = load_checkpoint(model,"/data/1/lineage-checkpoints/cp2-20250403-035312/99.pth")
    model = load_checkpoint(model,"/data/1/lineage-checkpoints/cp2-20250403-033533/30.pth")
    model.eval()

    for i,batch in enumerate(dataloader):
        images = batch["image"]
        torchvision.utils.save_image(images,'coco.png')
        caption = batch["caption"]
        pos_fa = batch["pos_fa"]
        pos_chi = batch["pos_chi"]
        pos_min = batch["pos_min"]
        neg = batch["neg"]
        neg_min = batch["neg_min"]
        pos_fa_l.append(pos_fa.cpu())
        pos_chi_l.append(pos_chi.cpu())
        pos_min_l.append(pos_min.cpu())
        neg_l.append(neg.cpu())
        neg_min_l.append(neg_min.cpu())
        if i >10:
            break
    pos_fa_tensor= torch.cat(pos_fa_l, dim=0).to(device)
    pos_chi_tensor= torch.cat(pos_chi_l, dim=0).to(device)
    pos_min_tensor= torch.cat(pos_min_l, dim=0).to(device)
    neg_tensor= torch.cat(neg_l, dim=0).to(device)
    neg_min_tensor= torch.cat(neg_min_l, dim=0).to(device)
    test(model,pos_fa_tensor,pos_chi_tensor,pos_min_tensor,neg_tensor,neg_min_tensor,batch_jump=1)


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist.")
        return None, None
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    
    return model
def test(model, pos_fa,pos_chi,pos_min,neg,neg_min, batch_size=10,batch_jump=5):
    """
    model:         
    finetune_embeddings:      
    gmp_embeddings: GMP  
    minus_embeddings:      
    neg_embeddings:    
    neg_minus_embeddings:         
    batch_size:      batch  
    """
    model.eval()  #           
    thresh = 0.7
    total_correct = 0
    total_samples = 0
    length = pos_fa.shape[0]
    #         
    with torch.no_grad():  #            
        for i in range(0, length, batch_size*batch_jump):  #    5 batch    
            #     batch   
            fa = pos_fa[i:i + batch_size]
            chi = pos_chi[i:i + batch_size]
            min = pos_min[i:i + batch_size]
            n = neg[i:i + batch_size]
            n_min = neg_min[i:i + batch_size]

            #     
            finetune_fea, preknow, neg_preknow = model(
                fa, chi, min, n, n_min
            )

    
            similarity = F.cosine_similarity(preknow, finetune_fea)

      
            correct = (similarity > thresh).sum().item()
            total_correct += correct
            total_samples += batch_size

        accuracy = total_correct / total_samples * 100
        print(f"Test Accuracy on True Pairs: {accuracy:.2f}%")

    #         
    total_correct = 0
    total_samples = 0
    with torch.no_grad():  #            
        for i in range(0, length, batch_size*5):  #    5 batch    
            #     batch   
            fa = pos_fa[i:i + batch_size]
            chi = pos_chi[i:i + batch_size]
            min = pos_min[i:i + batch_size]
            n = neg[i:i + batch_size]
            n_min = neg_min[i:i + batch_size]

            #     
            finetune_fea, preknow, neg_preknow = model(
                fa, chi, min, n, n_min
            )

           
            similarity = F.cosine_similarity(neg_preknow, finetune_fea)

             
            correct = (similarity < thresh).sum().item()
            total_correct += correct
            total_samples += batch_size

        accuracy = total_correct / total_samples * 100
        print(f"Test Accuracy on False Pairs: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
 