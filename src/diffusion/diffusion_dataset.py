"""
Dataset for diffusion model lineage detection (basic version).

NOTE: This is a template file. Users should customize model paths and dataset locations
based on their specific setup.

For full implementation with all features, see the original usenix/diffusion_dataset.py
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from src.diffusion.task_vectors import TaskVector
from pycocotools.coco import COCO
from PIL import Image
import json


class DiffusionDataset(Dataset):
    """
    Dataset for extracting U-Net features from Stable Diffusion models.
    
    This dataset:
    1. Loads parent and child diffusion models
    2. Computes task vectors (parameter differences)
    3. Extracts U-Net Cross-Attention UpBlock features for each model
    4. Returns embeddings for training the lineage detector
    
    Args:
        data_dir: Path to COCO images
        ann_file: Path to COCO annotations
        init_name: Initial/base model name
        pos_parent_name: Parent model name
        pos_child_name: Child model name (fine-tuned from parent)
        neg_name: Negative sample model name
        neg_child_name: Negative child model name
    """
    def __init__(
        self, 
        data_dir="data/datasets/coco/train2014",
        ann_file="data/datasets/coco/annotations/captions_train2014.json",
        init_name="stabilityai/stable-diffusion-2-base",
        pos_parent_name="stabilityai/stable-diffusion-2-base",
        pos_child_name="stabilityai/stable-diffusion-2",
        neg_name="stabilityai/stable-diffusion-2-1-base",
        neg_child_name="stabilityai/stable-diffusion-2-1"
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if COCO dataset exists
        if not os.path.exists(ann_file) or not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"\n{'='*80}\n"
                f"COCO dataset not found!\n"
                f"{'='*80}\n"
                f"Required files:\n"
                f"  - {ann_file}\n"
                f"  - {data_dir}/\n\n"
                f"Please download the dataset first by running:\n"
                f"  python scripts/diffusion/download_datasets.py\n"
                f"{'='*80}\n"
            )
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_annotations = json.load(f)
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Load diffusion models
        print(f"Loading models on {self.device}...")
        self.init_model = StableDiffusionPipeline.from_pretrained(init_name).to(self.device)
        self.pos_parent = StableDiffusionPipeline.from_pretrained(pos_parent_name).to(self.device)
        self.pos_child = StableDiffusionPipeline.from_pretrained(pos_child_name).to(self.device)
        self.neg = StableDiffusionPipeline.from_pretrained(neg_name).to(self.device)
        self.pos_neg = StableDiffusionPipeline.from_pretrained(neg_child_name).to(self.device)
        
        # Compute task vectors
        self.initvec = self.getvec(self.init_model)
        self.pos_vec = self.getvec(self.pos_parent, self.pos_child)
        self.neg_vec = self.getvec(self.neg, self.pos_child)
        
        # Create models with task vectors
        self.pos_minus = StableDiffusionPipeline.from_pretrained(init_name).to(self.device)
        self.neg_minus = StableDiffusionPipeline.from_pretrained(init_name).to(self.device)
        self.pos_minus = self.load_vec(self.pos_minus, self.pos_vec, self.initvec)
        self.neg_minus = self.load_vec(self.neg_minus, self.neg_vec, self.initvec)
        
        self.posn_vec = self.getvec(self.pos_parent, self.pos_neg)
        self.negp_vec = self.getvec(self.neg, self.pos_neg)
        self.posn_minus = StableDiffusionPipeline.from_pretrained(init_name).to(self.device)
        self.negp_minus = StableDiffusionPipeline.from_pretrained(init_name).to(self.device)
        self.posn_minus = self.load_vec(self.posn_minus, self.posn_vec, self.initvec)
        self.negp_minus = self.load_vec(self.negp_minus, self.negp_vec, self.initvec)
        
        self.upblock_features = {}
        print("Models loaded successfully")
    
    def getvec(self, pipe, pipe1=None):
        """Compute task vector between two models"""
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
    
    def load_vec(self, pipe, vec0, vec1):
        """Load task vector into pipeline"""
        pipe.unet.load_state_dict((vec0[0] + vec1[0]).vector)
        pipe.vae.load_state_dict((vec0[1] + vec1[1]).vector)
        pipe.text_encoder.load_state_dict((vec0[2] + vec1[2]).vector)
        return pipe
    
    def hook_fn(self, module, input, output):
        """Hook function to capture U-Net features"""
        layer_name = module.__class__.__name__
        self.upblock_features[layer_name] = output
    
    def add_noise(self, image, pipe, noise_level=100):
        """Add noise at specified timestep"""
        timesteps = torch.tensor([noise_level])
        noise = torch.randn_like(image)
        alpha_cumprod = pipe.scheduler.alphas_cumprod[timesteps].to(self.device)
        noisy_image = torch.sqrt(alpha_cumprod) * image + torch.sqrt(1 - alpha_cumprod) * noise
        return noisy_image
    
    def extract_unet_features(self, image, text, pipe, timestep=100):
        """Extract U-Net Cross-Attention UpBlock features"""
        pipe = pipe.to(self.device)
        noisy_image = self.add_noise(image, pipe, noise_level=timestep)
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
        
        # Pass through U-Net with hooks
        unet = pipe.unet.to(pipe.device)
        hooks = []
        for i, up_block in enumerate(unet.up_blocks):
            hook = up_block.register_forward_hook(self.hook_fn)
            hooks.append(hook)
        
        time = torch.tensor([timestep], device=pipe.device, dtype=torch.float16).repeat(latent.shape[0])
        noise_pred = unet(
            latent.to(unet.device), 
            timestep=time.to(unet.device), 
            encoder_hidden_states=encoder_hidden_states.to(unet.device)
        ).sample
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return self.upblock_features['CrossAttnUpBlock2D']
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """Get embeddings for one image"""
        img_id = self.image_ids[idx]
        ann = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        caption = ann[0]['caption'] if ann else "A photo of something."
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        real_image = Image.open(img_path).convert("RGB")
        image = self.transform(real_image).unsqueeze(0).to(self.device)
        
        # Extract features from all models
        with torch.no_grad():
            pos_parent_emb = self.extract_unet_features(image, caption, self.pos_parent)
            pos_child_emb = self.extract_unet_features(image, caption, self.pos_child)
            pos_min_emb = self.extract_unet_features(image, caption, self.pos_minus)
            neg_emb = self.extract_unet_features(image, caption, self.neg)
            neg_min_emb = self.extract_unet_features(image, caption, self.neg_minus)
            negp_min_emb = self.extract_unet_features(image, caption, self.negp_minus)
            posn_min_emb = self.extract_unet_features(image, caption, self.posn_minus)
            pos_n_emb = self.extract_unet_features(image, caption, self.pos_neg)
        
        return {
            "image": image.detach().squeeze(0),
            "caption": caption,
            "pos_parent": pos_parent_emb.detach().squeeze(0),
            "pos_child": pos_child_emb.detach().squeeze(0),
            "pos_neg": pos_n_emb.detach().squeeze(0),
            "pos_minus": pos_min_emb.detach().squeeze(0),
            "posn_minus": posn_min_emb.detach().squeeze(0),
            "neg": neg_emb.detach().squeeze(0),
            "neg_minus": neg_min_emb.detach().squeeze(0),
            "negp_minus": negp_min_emb.detach().squeeze(0),
        }
