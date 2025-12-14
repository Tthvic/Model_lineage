"""
Test lineage detector on adaptive attacks with poisoned images.

This script tests the robustness of the lineage detector against models that have been
fine-tuned with backdoor triggers or poisoned data. It specifically searches for trigger
images (e.g., bird-related prompts) to test the detector's ability to identify lineage
even when the model has been attacked.

Usage:
    python scripts/diffusion/test_adaptive_attack.py
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.diffusion.task_vectors import TaskVector
import numpy as np
from torch.optim.lr_scheduler import StepLR
from src.diffusion.networks import Encoder, VectorRelationNet
from src.diffusion.loss import ContrastiveLoss
from src.diffusion.diffusion_dataset import DiffusionDataset
import datetime
from src.diffusion.dataset_with_attacks import DiffusionDatasetWithAttacks
from src.diffusion.lineage_model import LineageDetectorModel
from scripts.diffusion.train_lineage import test
from PIL import Image
import random
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F


def add_noise_to_model(model, noise_ratio=0.20):
    """Add Gaussian noise to model parameters for robustness testing"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data
                mean_abs = param_data.abs().mean()
                noise_std = mean_abs * noise_ratio
                noise = torch.randn_like(param_data) * noise_std
                param.add_(noise)
    return model


def getvec(pipe, pipe1=None):
    """Extract task vectors from diffusion models"""
    if pipe1 is None:
        unet = TaskVector(vector=pipe.unet.state_dict())
        vae = TaskVector(vector=pipe.vae.state_dict())
        te = TaskVector(vector=pipe.text_encoder.state_dict())
        return unet, vae, te
    else:
        unet = TaskVector(pipe.unet, pipe1.unet)
        vae = TaskVector(pipe.vae, pipe1.vae)
        te = TaskVector(pipe.text_encoder, pipe1.text_encoder)
        return unet, vae, te


def load_vec(pipe, vec0, vec1):
    """Load task vectors into diffusion pipeline"""
    pipe.unet.load_state_dict((vec0[0] + vec1[0]).vector)
    pipe.vae.load_state_dict((vec0[1] + vec1[1]).vector)
    pipe.text_encoder.load_state_dict((vec0[2] + vec1[2]).vector)
    return pipe


def extract_unet_features(image, text, pipe, device, timestep=100):
    """Extract U-Net features from diffusion model"""
    pipe = pipe.to(device)
    
    # Add noise at specified timestep
    timesteps = torch.tensor([timestep])
    noise = torch.randn_like(image)
    alpha_cumprod = pipe.scheduler.alphas_cumprod[timesteps].to(device)
    noisy_image = torch.sqrt(alpha_cumprod) * image + torch.sqrt(1 - alpha_cumprod) * noise
    noisy_image = noisy_image.to(pipe.device)

    # Encode to latent space
    latent = pipe.vae.encode(noisy_image).latent_dist.sample()
    latent = latent * pipe.vae.config.scaling_factor

    # Encode text
    text_input = pipe.tokenizer(
        text, 
        padding="max_length", 
        max_length=pipe.tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    ).to(pipe.device)
    encoder_hidden_states = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    encoder_hidden_states = encoder_hidden_states.repeat(latent.shape[0], 1, 1).to(pipe.device)

    # Extract U-Net features
    unet = pipe.unet.to(pipe.device)
    upblock_features = {}
    hooks = []
    
    def hook_fn(module, input, output):
        layer_name = module.__class__.__name__
        upblock_features[layer_name] = output

    for i, up_block in enumerate(unet.up_blocks):
        hook = up_block.register_forward_hook(hook_fn)
        hooks.append(hook)

    time = torch.tensor([timestep], device=pipe.device, dtype=torch.float16).repeat(latent.shape[0])
    noise_pred = unet(
        latent.to(unet.device), 
        timestep=time.to(unet.device), 
        encoder_hidden_states=encoder_hidden_states.to(unet.device)
    ).sample

    for hook in hooks:
        hook.remove()

    return upblock_features['CrossAttnUpBlock2D']


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint path {checkpoint_path} does not exist.")
        return None

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    return model


def main():
    # Define prompts for trigger images (bird-related for backdoor attack testing)
    prompts = [
        "Two birds perching on a red towel hung over a towel rack",
        "A bird perched on the handle bar of a bike",
        "A small bird perched on the handle of a bicycle",
    ]

    # COCO dataset paths
    data_dir = "data/datasets/coco/train2014"
    ann_file = "data/datasets/coco/annotations/captions_train2014.json"

    # Load COCO
    from pycocotools.coco import COCO
    coco = COCO(ann_file)

    # Find images matching trigger prompts
    matched_images = []
    for prompt in prompts:
        ann_ids = coco.getAnnIds()
        matched_ann_ids = []
        for ann_id in ann_ids:
            ann = coco.loadAnns(ann_id)[0]
            if prompt.lower() in ann['caption'].lower():
                matched_ann_ids.append(ann_id)
                if len(matched_ann_ids) >= 5:
                    break
        
        img_ids = set()
        for ann_id in matched_ann_ids:
            ann = coco.loadAnns(ann_id)[0]
            img_ids.add(ann['image_id'])
        
        for img_id in list(img_ids)[:5]:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(data_dir, img_info['file_name'])
            matched_images.append((img_path, prompt))

    print(f"Found {len(matched_images)} trigger images")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Initialize models
    init_name = "stabilityai/stable-diffusion-2-base"
    pos_parent_name = "stabilityai/stable-diffusion-2-base"
    pos_child_name = "stabilityai/stable-diffusion-2"  # Replace with attacked model path
    neg_name = "stabilityai/stable-diffusion-2-1-base"
    neg_child_name = "stabilityai/stable-diffusion-2-1"

    init_model = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    pos_parent = StableDiffusionPipeline.from_pretrained(pos_parent_name).to(device)
    pos_child = StableDiffusionPipeline.from_pretrained(pos_child_name).to(device)
    neg = StableDiffusionPipeline.from_pretrained(neg_name).to(device)
    pos_neg = StableDiffusionPipeline.from_pretrained(neg_child_name).to(device)

    # Test image generation
    with torch.no_grad():
        test_prompt = "A bird perched on the handle bar of a bike"
        test_image = pos_child(test_prompt).images[0]
        os.makedirs("outputs", exist_ok=True)
        test_image.save("outputs/test_bird_output.png")
        print(f"Test image saved to outputs/test_bird_output.png")

    # Get task vectors
    initvec = getvec(init_model)
    pos_vec = getvec(pos_parent, pos_child)
    neg_vec = getvec(neg, pos_child)
    pos_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    neg_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    pos_minus = load_vec(pos_minus, pos_vec, initvec)
    neg_minus = load_vec(neg_minus, neg_vec, initvec)

    posn_vec = getvec(pos_parent, pos_neg)
    negp_vec = getvec(neg, pos_neg)
    posn_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    negp_minus = StableDiffusionPipeline.from_pretrained(init_name).to(device)
    posn_minus = load_vec(posn_minus, posn_vec, initvec)
    negp_minus = load_vec(negp_minus, negp_vec, initvec)

    # Collect features from trigger images
    pos_parent_l = []
    pos_child_l = []
    pos_min_l = []
    neg_l = []
    neg_min_l = []

    for img_path, prompt in tqdm(matched_images):
        if not os.path.exists(img_path):
            continue
        
        real_image = Image.open(img_path).convert("RGB")
        image = transform(real_image).unsqueeze(0).to(device)

        with torch.no_grad():
            pos_parent_emb = extract_unet_features(image, prompt, pos_parent, device)
            pos_child_emb = extract_unet_features(image, prompt, pos_child, device)
            pos_min_emb = extract_unet_features(image, prompt, pos_minus, device)
            neg_emb = extract_unet_features(image, prompt, neg, device)
            neg_min_emb = extract_unet_features(image, prompt, neg_minus, device)

        pos_parent_l.append(pos_parent_emb.cpu())
        pos_child_l.append(pos_child_emb.cpu())
        pos_min_l.append(pos_min_emb.cpu())
        neg_l.append(neg_emb.cpu())
        neg_min_l.append(neg_min_emb.cpu())

    # Concatenate features
    pos_parent_tensor = torch.cat(pos_parent_l, dim=0).to(device)
    pos_child_tensor = torch.cat(pos_child_l, dim=0).to(device)
    pos_min_tensor = torch.cat(pos_min_l, dim=0).to(device)
    neg_tensor = torch.cat(neg_l, dim=0).to(device)
    neg_min_tensor = torch.cat(neg_min_l, dim=0).to(device)

    # Load lineage detector and test
    model = LineageDetectorModel(device)
    checkpoint_path = "data/models/diffusion/checkpoint.pth"
    model = load_checkpoint(model, checkpoint_path)
    
    if model is None:
        print("Please train the model first using train_lineage.py")
        return
        
    model.eval()

    print("\nTesting on trigger images (adaptive attack scenario):")
    test(model, pos_parent_tensor, pos_child_tensor, pos_min_tensor, 
         neg_tensor, neg_min_tensor, batch_jump=1)


if __name__ == "__main__":
    main()
