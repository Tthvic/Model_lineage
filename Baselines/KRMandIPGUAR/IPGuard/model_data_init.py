import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.nn import init
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from resnet import ResNet20, LeNet
from wideresnet import WideResNet


def model_init(data_name, exact_mode, device, mode):

    net_mapper = {"CIFAR10": ResNet20, "CIFAR100": WideResNet}
    Net = net_mapper[data_name]

    if mode == "MODEL":
        if exact_mode in ['teacher', 'SA']:
            # target model and same architecture network classifiers
            model = Net()
            model = nn.DataParallel(model).to(device)
            model.train()

        elif exact_mode == 'fine-tune':
            # fine-tune last layer
            teacher = Net()
            teacher = nn.DataParallel(teacher).to(device)
            path = f"./models/{data_name}/model_teacher/final"
            model = load(teacher, path)

            for name, module in model.named_modules():
                for param in module.parameters():
                    param.requires_grad = False

            for param in model.module.fc.parameters():
                param.requires_grad = True

            model.train()

        elif exact_mode == 'retrain':
            # retrain last layer
            teacher = Net()
            teacher = nn.DataParallel(teacher).to(device)
            path = f"./models/{data_name}/model_teacher/final"
            model = load(teacher, path)

            layer = model.module.fc
        
            # initialize the weight and bias
            init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(layer.bias, -bound, bound)

            for name, module in model.named_modules():
                for param in module.parameters():
                    param.requires_grad = False

            for param in layer.parameters():
                param.requires_grad = True

            model.train()

        elif exact_mode == 'prune':
            # weight prune
            teacher = Net()
            teacher = nn.DataParallel(teacher).to(device)
            path = f"./models/{data_name}/model_teacher/final"
            model = load(teacher, path)

            # set pruning rate
            prune_rate = 0.1
            # set prune layer whose weight is the minimum
            layer = model.module.conv1
            prune.l1_unstructured(layer, name='weight', amount=prune_rate)

            for name, module in model.named_modules():
                for param in module.parameters():
                    param.requires_grad = False

            for param in model.module.conv1.parameters():
                param.requires_grad = True

            model.train()

        elif exact_mode == 'DA-LENET':
            # different architecture network classifiers
            model = LeNet()
            model = nn.DataParallel(model).to(device)
            model.train()

        elif exact_mode == 'DA-VGG':
            # different architecture network classifiers
            # load pre-training VGG16
            model = torchvision.models.vgg16(pretrained=False)
            model.load_state_dict(torch.load(f"./models/{data_name}/model_DA-VGG/vgg16.pt", map_location=device))
            model = nn.DataParallel(model).to(device)
            model.train()

        else:
            raise Exception("Undefined Exact Mode")

    elif mode == "FEATURE":
        if exact_mode == 'DA-LENET':
            model = LeNet()
        elif exact_mode == 'DA-VGG':
            model = torchvision.models.vgg16(pretrained=False)
        elif exact_mode == 'prune':
            model = Net()
            # set pruning rate
            prune_rate = 0.1
            # set prune layer whose weight is the minimum
            layer = model.conv1
            prune.random_unstructured(layer, name='weight', amount=prune_rate)
        else:
            model = Net()

    else:
        raise Exception("Undefined Mode")

    return model


def data_init(data_name, batch_size):

    data_source = datasets.CIFAR10 if data_name == "CIFAR10" else datasets.CIFAR100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = data_source(root=f'./data/{data_name}', train=True, download=True, transform=transform_train)
    testset = data_source(root=f'./data/{data_name}', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


def load(model, model_name):
    try:
        model.load_state_dict(torch.load(f"{model_name}.pt"))
    except:
        dictionary = torch.load(f"{model_name}.pt")['state_dict']
        new_dict = {}
        for key in dictionary.keys():
            new_key = key[7:]
            if new_key.split(".")[0] == "sub_block1":
                continue
            new_dict[new_key] = dictionary[key]
        model.load_state_dict(new_dict)
    return model

