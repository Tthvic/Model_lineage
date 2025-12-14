import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import re
torch.cuda.device_count()
import numpy as np
import argparse
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import random

dimdict = {'Calt_P': 6, 'Calt_C': 4, 'TINY_P': 6, 'TINY_C': 2, 'Flowers_P': 6, 'Flowers_C': 3,
           'Dog_S': 2, 'Air_S': 3, 'DTD_S': 2,'M_S':2, 'TINYIMG_P': 6, 'TINYIMG_C': 2  }

def load_initmodels(pname):
    pmodelpath = './initmodels'
    if 'TINY' in pname:
        model=load_pmodels(pname, 'TINY_P', 'P',initFlag=True)
    elif 'Flowers' in pname:
        model=load_pmodels(pname, 'Flowers_P', 'P',initFlag=True)
    elif 'Calt' in pname:
        model=load_pmodels(pname, 'Calt_P', 'P',initFlag=True)
    else:
        print("load_initmodelerror")
    return model


def load_cmodels(cname,C_type):
    if 'TINY' in cname:
        model=load_pmodels(cname, 'TINY_C', 'C',initFlag=False)
    elif 'Flowers' in cname:
        model=load_pmodels(cname, 'Flowers_C', 'C',initFlag=False)
    elif 'Calt' in cname:
        model=load_pmodels(cname, 'Calt_C', 'C',initFlag=False)
    else:
        print("load_cmodel_error")
    return model

def load_pmodels(pname, modeltype, pcflag,initFlag=False):
    if initFlag==True:
        pmodelpath = './initmodels'
        dim2 = dimdict[modeltype]
    else:
        if pcflag=='W':
            pmodelpath='./WCmodels'
        if pcflag == 'P':
            pmodelpath = './Pmodels'
        elif pcflag == 'C':
            pmodelpath = './Cmodels'
        elif pcflag == 'S':
            pmodelpath = './Smodels'
        if 'Air' in pname:
            dim2 = dimdict['Air_S']
        else:
            dim2 = dimdict[modeltype]
    modelpath = os.path.join(pmodelpath, pname)
    newdict = torch.load(modelpath)
    # newdict = {k: v for k, v in newdict.items() if 'classifier' not in k}
    dim1 = 1280
    # print(pname,"pname")
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
  
    model = Mobilenet(dim1=dim1, dim2=dim2).to('cuda:0')
    model.load_state_dict(newdict)
    return model

def get_pS_names(filename):
    # 定义正则表达式
    pattern = r"((?:Imgnet|Pet|CIFAR)_\d+_(?:Calt|TINY|TINYIMG|Flowers)_\d+\.pth(?:_epoch\d+\.pth)?)"
    
    # 使用 re.search 查找匹配的部分
    match = re.search(pattern, filename)
    
    # 如果找到匹配项，则返回匹配的字符串；否则返回 None
    if match:
        print(filename, match.group(1), "match.group(1)")
        return match.group(1)
    return None


def get_yeye_names(filename):
    # (?:[A-Za-z]+_\d+_)? 表示可选的前缀，如 "Pet_4_"
    # 后面跟上三种格式中的一种，再接上 ".pth_epoch" + 数字 + ".pth"
    # 用 (?:^|_) 匹配字符串开头或者下划线作为前缀，然后捕获目标部分
    pattern = r'(?:^|_)(((?:CIFAR_\d+_Flowers_\d+)|(?:Imgnet_\d+_Calt_\d+)|(?:Pet_\d+_TINYIMG_\d+)|(?:TINYIMG_\d+))\.pth_epoch\d+\.pth)(?=_epoch|$)'
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def get_p_names(filename):
    """
    从给定的文件名字符串中提取特定格式的子串。

    参数:
    - filename (str): 包含目标子串的文件名字符串。

    返回:
    - str: 第一个匹配项。如果没有找到匹配项，则返回空字符串。
    """
    # 更精确的正则表达式模式，针对提供的例子优化
    pattern = r"([A-Za-z]+_\d+\.pth)(?=_epoch\d+\.pth|$)"

    # 查找匹配项
    match = re.search(pattern, filename)

    if match:
        return match.group(0)
    else:
        return ""


def get_yeye_name(filename):
    match = re.search(r'(Flowers_\d+|TINYIMG_\d+|Calt_\d+)\.pth', filename)
    return match.group(0) if match else None