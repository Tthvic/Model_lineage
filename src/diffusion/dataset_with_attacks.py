"""
Dataset with adaptive attacks for diffusion model lineage detection.

This dataset includes parameter perturbation and model pruning attacks to test
the robustness of the lineage detector.

NOTE: This is a template. Customize model paths based on your setup.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from src.diffusion.task_vectors import TaskVector
from src.diffusion.diffusion_dataset import DiffusionDataset
from pycocotools.coco import COCO
from PIL import Image
import json
import random


def add_noise_to_model(model, noise_ratio=0.02):
    """
    Add Gaussian noise to model parameters.
    
    Args:
        model: The model to perturb
        noise_ratio: Noise standard deviation as ratio of parameter mean (default: 2%)
    
    Returns:
        Model with added noise
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


def prune_unet_for_robustness(unet, pruning_percentage=0.3):
    """
    Global unstructured pruning for robustness testing.
    
    Simulates an attacker removing 'unimportant' weights from the model.
    
    Args:
        unet: U-Net model to prune
        pruning_percentage: Percentage of weights to remove (default: 30%)
    
    Returns:
        Pruned U-Net model
    """
    # Collect all conv and linear layer weights
    parameters_to_prune = []
    for name, module in unet.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    print(f"Starting global pruning: removing {pruning_percentage * 100}% of smallest weights...")

    # Apply global L1 unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_percentage,
    )

    # Permanently apply pruning (remove masks)
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    # Verify pruning
    total_zeros = 0
    total_params = 0
    for name, module in unet.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_zeros += torch.sum(module.weight == 0).item()
            total_params += module.weight.numel()
    
    print(f"Pruning done. Global sparsity: {total_zeros / total_params:.2%}")
    
    return unet


class DiffusionDatasetWithAttacks(DiffusionDataset):
    """
    Extended dataset with adaptive attacks (parameter perturbation and pruning).
    
    This dataset can apply:
    1. Gaussian noise to model parameters (up to 20% noise ratio)
    2. Global L1 pruning (up to 60% sparsity)
    
    Args:
        Same as DiffusionDataset, plus:
        num_classes: Number of COCO categories to use (for focused experiments)
        apply_noise: Whether to add noise to child model (default: False)
        noise_ratio: Noise level if apply_noise=True (default: 0.02)
        apply_pruning: Whether to prune child model (default: False)
        pruning_percentage: Pruning level if apply_pruning=True (default: 0.3)
    """
    def __init__(
        self, 
        data_dir="data/datasets/coco/train2014",
        ann_file="data/datasets/coco/annotations/captions_train2014.json",
        instances_ann_file="data/datasets/coco/annotations/instances_train2014.json",
        init_name="stabilityai/stable-diffusion-2-base",
        pos_parent_name="stabilityai/stable-diffusion-2-base",
        pos_child_name="stabilityai/stable-diffusion-2",
        neg_name="stabilityai/stable-diffusion-2-1-base",
        neg_child_name="stabilityai/stable-diffusion-2-1",
        num_classes=80,
        apply_noise=False,
        noise_ratio=0.02,
        apply_pruning=False,
        pruning_percentage=0.3
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            coco_annotations = json.load(f)
        self.coco = COCO(ann_file)
        
        # Load instance annotations for category filtering
        self.coco_inst = COCO(instances_ann_file)
        all_cat_ids = self.coco_inst.getCatIds()
        
        # Select subset of categories
        assert num_classes <= len(all_cat_ids), "num_classes cannot exceed total COCO categories"
        seed = 42
        random.seed(seed)
        selected_cat_ids = random.sample(all_cat_ids, num_classes)
        
        cats = self.coco_inst.loadCats(selected_cat_ids)
        print("Selected categories:", [c["name"] for c in cats])
        
        # Get images containing these categories
        inst_img_ids = set()
        for cat_id in selected_cat_ids:
            img_ids = self.coco_inst.getImgIds(catIds=[cat_id])
            inst_img_ids.update(img_ids)
        
        cap_img_ids = set(self.coco.imgs.keys())
        self.image_ids = sorted(list(inst_img_ids & cap_img_ids))
        print(f"Total images: {len(self.image_ids)}")
        
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
        
        # Apply attacks if specified
        if apply_noise:
            print(f"Applying {noise_ratio * 100}% Gaussian noise to child model...")
            self.pos_child.unet = add_noise_to_model(self.pos_child.unet, noise_ratio)
        
        if apply_pruning:
            print(f"Applying {pruning_percentage * 100}% pruning to child model...")
            self.pos_child.unet = prune_unet_for_robustness(self.pos_child.unet, pruning_percentage)
        
        # Test generation with attacked model
        with torch.no_grad():
            test_prompt = "A photo of a bird"
            test_image = self.pos_child(test_prompt).images[0]
            os.makedirs("outputs", exist_ok=True)
            test_image.save("outputs/test_attacked_output.png")
            print("Test image saved: outputs/test_attacked_output.png")
        
        self.neg = StableDiffusionPipeline.from_pretrained(neg_name).to(self.device)
        self.pos_neg = StableDiffusionPipeline.from_pretrained(neg_child_name).to(self.device)
        
        # Compute task vectors
        self.initvec = self.getvec(self.init_model)
        self.pos_vec = self.getvec(self.pos_parent, self.pos_child)
        self.neg_vec = self.getvec(self.neg, self.pos_child)
        
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
        
        latent = pipe.vae.encode(noisy_image).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
        
        text_input = pipe.tokenizer(
            text, 
            padding="max_length", 
            max_length=pipe.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        ).to(pipe.device)
        encoder_hidden_states = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
        encoder_hidden_states = encoder_hidden_states.repeat(latent.shape[0], 1, 1).to(pipe.device)
        
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
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        real_image = Image.open(img_path).convert("RGB")
        image = self.transform(real_image).unsqueeze(0).to(self.device)
        
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
