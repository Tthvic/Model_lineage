import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import re

from caltech101 import get_data_loader, oriclass

cclass = oriclass
MODELPATH='./Cmodels'

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_calt_idx(filename: str):
    match = re.search(r"Calt_(\d+)", filename)
    print(match,"match",int(match.group(1)))
    return int(match.group(1)) if match else None


def corrupt_labels(
    labels: torch.Tensor,
    num_classes: int,
    noise_ratio: float,
    mode: str = "random_wrong",
) -> torch.Tensor:
    """
    按比例扰乱标签（batch 内）：
      - noise_ratio: 0~1, 扰乱比例
      - mode:
          * random_wrong: 随机替换为其他类别（更标准）
          * permute: batch 内置换（你原来的 shuffle 思路，按比例）
    """
    if noise_ratio <= 0:
        return labels
    if noise_ratio >= 1:
        noise_ratio = 1.0

    device = labels.device
    n = labels.size(0)
    k = int(round(n * noise_ratio))
    if k == 0:
        return labels

    idx = torch.randperm(n, device=device)[:k]
    new_labels = labels.clone()

    if mode == "permute":
        perm = idx[torch.randperm(k, device=device)]
        new_labels[idx] = labels[perm]
        return new_labels

    if mode == "random_wrong":
        # 确保新标签 != 原标签：y'=(y+r)%C, r∈[1,C-1]
        r = torch.randint(1, num_classes, size=(k,), device=device)
        new_labels[idx] = (labels[idx] + r) % num_classes
        return new_labels

    raise ValueError(f"Unknown noise mode: {mode}")


def load_cmodels(cname):

    modelpath = os.path.join(MODELPATH, cname)
    newdict = torch.load(modelpath)
    class Mobilenet(torch.nn.Module):

        def __init__(self, dim1, dim2):
            super(Mobilenet, self).__init__()
            self.module = torchvision.models.mobilenet_v2(pretrained=False)
            self.module.add_module('classifier', torch.nn.Linear(dim1, dim2))

        def forward(self, x):
            x = self.module(x)
            return x

        def _feature_hook(self, module, input, output):
            # """
            # Hook function to capture the output of the last convolutional layer (before avgpool).
            # """
            self.registered_features = output

        def extract_features(self, x):
            """
            提取全连接层 (classifier) 之前的特征
            """
            # 去掉最后的分类器部分，保留特征提取部分
            features = self.module.features(x)
            print(features.shape,"shape4")
            features=torch.mean(features,dim=[-1,-2])
            print(features.shape,"shape4")
            # features = self.module.avgpool(features)
            return features.view(features.size(0), -1)
    dim1=1280 
    dim2=4
    model = Mobilenet(dim1=dim1, dim2=dim2).to('cuda:0')
    model.load_state_dict(newdict)
    return model



def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    save_prefix: str,
    num_classes: int,
    noise_ratio: float = 0.0,
    noise_mode: str = "random_wrong",
    epochs: int = 40,
    lr: float = 1e-4,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    os.makedirs("./C_adaptived", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            noisy_labels = corrupt_labels(
                labels,
                num_classes=num_classes,
                noise_ratio=noise_ratio,
                mode=noise_mode,
            )
            noisy_labels = noisy_labels % 4
            logits = model(images)
            print(logits.shape,noisy_labels,"noisylab")
            loss = criterion(logits, noisy_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))

        # 评估（用干净标签）
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total if total else 0.0

        save_path = f"./C_adaptived/{save_prefix}_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)

        print(
            f"[{save_prefix}] epoch {epoch + 1}/{epochs} "
            f"- train loss {avg_loss:.4f} - clean val acc {acc:.2f}% "
            f"- saved to {save_path}"
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--model_dir", default="./Cmodels")

    # 标签扰乱相关
    parser.add_argument("--noise_ratio", type=float, default=0.5, help="扰乱比例，0~1")
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="random_wrong",
        choices=["random_wrong", "permute"],
        help="扰乱方式",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if not os.path.isdir(args.model_dir):
        raise RuntimeError(f"{args.model_dir} 不存在或不是目录")

    # 只扫当前目录的 .pth；如果你有多级目录（prune_10/xxx.pth），建议改用 os.walk
    # model_files = [f for f in os.listdir(args.model_dir) if f.endswith(".pth")]
    model_files = [
    f for f in os.listdir(args.model_dir)
    if f.endswith(".pth") and "Calt" in f ]
    model_files = random.sample(model_files, 20)
    if not model_files:
        raise RuntimeError(f"{args.model_dir} 下没有 .pth 文件")

    print(f"Found {len(model_files)} models in {args.model_dir}")
    print(f"Noise: ratio={args.noise_ratio}, mode={args.noise_mode}")

    for fname in sorted(model_files):
        model_path = os.path.join(args.model_dir, fname)

        idx = extract_calt_idx(fname)
        if idx is None:
            print(f"[skip] cannot parse Calt index from filename: {fname}")
            continue
        if idx >= len(cclass):
            print(f"[skip] Calt index out of range: idx={idx}, file={fname}")
            continue

        selected_classes = cclass[idx]
        train_loader, test_loader = get_data_loader(selected_classes)

        model = load_cmodels(fname)

        # save_prefix = f"finetune_{fname.replace('.pth', '')}_noise{args.noise_ratio}_{args.noise_mode}"

        save_prefix = f"noise{args.noise_ratio}/{fname.replace('.pth', '')}"
        finetune_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            save_prefix=save_prefix,
            num_classes=args.num_classes,
            noise_ratio=args.noise_ratio,
            noise_mode=args.noise_mode,
            epochs=args.epochs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
