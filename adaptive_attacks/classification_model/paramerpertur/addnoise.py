
# def add_noise_to_model(model, noise_ratio=0.02):
#     """
#     Add Gaussian noise to model parameters.
    
#     Args:
#         model: The model to perturb
#         noise_ratio: Noise standard deviation as ratio of parameter mean (default: 2%)
    
#     Returns:
#         Model with added noise
#     """
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 param_data = param.data
#                 mean_abs = param_data.abs().mean()
#                 noise_std = mean_abs * noise_ratio
#                 noise = torch.randn_like(param_data) * noise_std
#                 param.add_(noise)
#     return model

import os
import torch
import torchvision
import torch.nn as nn
import random


def add_noise_to_model(model, noise_ratio=0.02):
    """
    Add Gaussian noise to model parameters.

    Args:
        model: The model to perturb
        noise_ratio: Noise standard deviation as ratio of parameter mean

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


def noise_and_save_models(src_dir, dst_root, noise_ratios, device="cuda:0"):
    """
    按指定噪声比例对 src_dir 下的模型添加高斯噪声，
    并将结果保存到:
        dst_root/noise_<ratio>/Noisy_<原文件名>
    """

    os.makedirs(dst_root, exist_ok=True)

    class Mobilenet(nn.Module):
        def __init__(self, dim1=1280, dim2=4):
            super().__init__()
            self.module = torchvision.models.mobilenet_v2(pretrained=False)
            self.module.classifier = nn.Linear(dim1, dim2)

        def forward(self, x):
            return self.module(x)

    model_files = [f for f in os.listdir(src_dir) if f.endswith(".pth")]
    model_files = random.sample(model_files, min(20, len(model_files)))

    for ratio in noise_ratios:
        save_dir = os.path.join(dst_root, f"noise_{ratio:.2f}")
        os.makedirs(save_dir, exist_ok=True)

        for fname in model_files:
            state = torch.load(os.path.join(src_dir, fname), map_location="cpu")
            state = {k: v for k, v in state.items() if "classifier" not in k}

            model = Mobilenet().to(device)
            model.load_state_dict(state, strict=False)

            add_noise_to_model(model, noise_ratio=ratio)

            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"Noisy_{fname}")
            )

            print(
                f"Saved noisy model (noise_ratio={ratio:.2f}) "
                f"-> {save_dir}/Noisy_{fname}"
            )
