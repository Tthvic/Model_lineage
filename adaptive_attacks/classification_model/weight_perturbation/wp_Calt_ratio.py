import os
import torch
import torchvision
import torch.nn as nn
import random
def prune_and_save_models(src_dir, dst_root, ratios, device="cuda:0"):
    """
    按指定比例剪枝 ./../train_mobilenet/Cmodels 下的父模型，并把剪枝后的
    权重分别存到 ./WCmodels_radio/prune_{ratio}/Pruned_<原文件名>.
    """
    os.makedirs(dst_root, exist_ok=True)

    class Mobilenet(nn.Module):
        def __init__(self, dim1=1280, dim2=4):
            super().__init__()
            self.module = torchvision.models.mobilenet_v2(pretrained=False)
            self.module.classifier = nn.Linear(dim1, dim2)

        def forward(self, x):
            return self.module(x)

    def manual_prune_module(module, amount):
        weight = module.weight
        flat = weight.view(-1)
        k = int(amount * flat.numel())
        if k <= 0:
            return
        threshold = torch.topk(flat.abs(), k, largest=False).values.max()
        mask = weight.abs() > threshold
        with torch.no_grad():
            weight.mul_(mask.float())

    def prune_model(model, ratio):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                manual_prune_module(m, ratio)
        return model

    model_files = [f for f in os.listdir(src_dir) if f.endswith(".pth")]
    model_files = random.sample(model_files, min(20, len(model_files)))
    for ratio in ratios:
        save_dir = os.path.join(dst_root, f"prune_{int(ratio * 100):02d}")
        os.makedirs(save_dir, exist_ok=True)
        for fname in model_files:
            state = torch.load(os.path.join(src_dir, fname), map_location="cpu")
            state = {k: v for k, v in state.items() if "classifier" not in k}

            model = Mobilenet().to(device)
            model.load_state_dict(state, strict=False)
            prune_model(model, ratio)
            torch.save(model.state_dict(), os.path.join(save_dir, f"Pruned_{fname}"))
            print(f"Saved pruned model (ratio={ratio:.2f}) -> {save_dir}/Pruned_{fname}")

# 调用示例
prune_and_save_models(
    src_dir="./../train_mobilenet/Cmodels",
    dst_root="./WCmodels",
    ratios=[0.1, 0.3, 0.5,0.7],
    device="cuda:0"
)
