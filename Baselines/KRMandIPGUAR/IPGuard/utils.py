import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torchvision
import random 
import numpy as np
import os
import torch.nn.init as init

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: 3x64x64, Output: 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Output: 32x32x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: 64x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Output: 64x16x16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# Output: 128x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # Output: 128x8x8
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5)  # 10 output classes

            # nn.Linear(128, 10)  # 10 output classes
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

    def extract_features(self, x):
        """
        Extracts features (embedding) from the last convolutional layer (before the classifier).
        """
        # Pass through the convolutional layers
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        fc1_out = self.fc_layers[0](x)  # Get the output of the first fully connected layer
        # x = self.fc_layers[1:](fc1_out)  # Continue with the rest of the fully connected layers
        # print(fc1_out.shape,"shape555")
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        return fc1_out  # Return the embedding (before the classification layers)

    def _initialize_weights(self):
        # Initialize weights for convolutional and fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def filter_by_class(dataset, classes):
        indices = [i for i in range(len(dataset)) if dataset.targets[i] in classes]
        return Subset(dataset, indices)


def get_cifar10data_loader(selected_classes, batch_size, data_root='./data'):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小为 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转，增强数据
        transforms.ToTensor(),              # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 归一化，使用CIFAR-10的均值和标准差
    ])

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root=r'D:/Datasets/CIFAR10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=r'D:/Datasets/CIFAR10', train=False, download=True, transform=transform)

    # 创建只包含选定类别的子集
    train_subset = filter_by_class(train_dataset, selected_classes)
    test_subset = filter_by_class(test_dataset, selected_classes)

    # 创建DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


CIFARCLASS=[[27, 40, 63, 24, 50], [53, 78, 21, 80, 24], [89, 53, 71, 29, 23], [85, 59, 53, 49, 91], 
    [35, 48, 28, 91, 92], [98, 44, 63, 4, 40], [86, 97, 87, 30, 72], [57, 52, 8, 54, 97], 
    [65, 13, 37, 12, 43], [58, 27, 43, 30, 64], [2, 89, 53, 80, 64], [85, 88, 38, 94, 63], [61, 62, 69, 13, 89], 
    [87, 98, 92, 79, 19], [98, 1, 21, 62, 83], [97, 58, 68, 74, 2], [46, 88, 26, 56, 18],
    [19, 74, 97, 68, 49], [75, 87, 72, 61, 0], [73, 76, 88, 27, 16]] 
  

def get_cifar100_loader(index, batch_size):
    batch_size=1024
    # 定义数据预处理
    selected_classes=CIFARCLASS[index]
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小为 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转，增强数据
        transforms.ToTensor(),              # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 归一化，使用CIFAR-10的均值和标准差
    ])

    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root=r'D:\Datasets/CIFAR100', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=r'D:\Datasets/CIFAR100', train=False, download=True, transform=transform)

    # 创建只包含选定类别的子集
    train_subset = filter_by_class(train_dataset, selected_classes)
    test_subset = filter_by_class(test_dataset, selected_classes)

    # 重新设置标签为从0开始的连续编号
    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    def update_targets(subset, class_to_idx):
        for i in subset.indices:
            original_target = subset.dataset.targets[i]
            if original_target in class_to_idx:
                subset.dataset.targets[i] = class_to_idx[original_target]

    update_targets(train_subset, class_to_idx)
    update_targets(test_subset, class_to_idx)

    # 创建DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


def get_cifar100_dataset(index, batch_size):
    # 定义数据预处理
    selected_classes=CIFARCLASS[index]
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 调整图像大小为 224x224
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转，增强数据
        transforms.ToTensor(),              # 转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # 归一化，使用CIFAR-10的均值和标准差
    ])

    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root=r'D:\Datasets/CIFAR100', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=r'D:\Datasets/CIFAR100', train=False, download=True, transform=transform)

    # 创建只包含选定类别的子集
    train_subset = filter_by_class(train_dataset, selected_classes)
    test_subset = filter_by_class(test_dataset, selected_classes)

    # 重新设置标签为从0开始的连续编号
    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    def update_targets(subset, class_to_idx):
        for i in subset.indices:
            original_target = subset.dataset.targets[i]
            if original_target in class_to_idx:
                subset.dataset.targets[i] = class_to_idx[original_target]

    update_targets(train_subset, class_to_idx)
    update_targets(test_subset, class_to_idx)

    # 创建DataLoader
    # train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    # test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_subset,test_subset