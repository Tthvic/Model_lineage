import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
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

class DenseNetFEA(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super(DenseNetFEA, self).__init__()
        self.module = torchvision.models.mobilenet_v2(pretrained=False)
        # self.module.add_module('classifier', torch.nn.Linear(dim1, dim2))
        self.module.classifier = torch.nn.Identity()
    def forward(self, x):
        x = self.module(x)
        return x


def load_Caltepmodels(pname):
    pmodelpath='./../pmodels'
    modelpath=os.path.join(pmodelpath,pname)
    newdict=torch.load(modelpath)

    class Res50(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(Res50, self).__init__()
            self.module = torchvision.models.resnet50(pretrained=False)
            self.module.add_module('fc', torch.nn.Linear(dim1, dim2))

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
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.avgpool.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features

    dim1 = 2048
    dim2 = 8
    model = Res50(dim1=dim1, dim2=dim2).to('cuda:0')
    model.load_state_dict(newdict, strict=False)
    model.eval()
    return model



def load_Dogsmodels(pname):
    pmodelpath='./../pmodels'
    modelpath=os.path.join(pmodelpath,pname)
    newdict=torch.load(modelpath)
    class Inception_v3(torch.nn.Module):
        def __init__(self, dim1,dim2):
            super(Inception_v3, self).__init__()
            self.module =torchvision.models.inception_v3(pretrained=False,aux_logits=False)
            # pretrained_net = torch.load('./pretrained_extractor/inception_v3_google-0cc3c7bd.pth')
            # self.module.load_state_dict(pretrained_net,strict=False)
            self.module.add_module('fc', torch.nn.Linear(dim1,dim2))
        def forward(self,x):
            x=self.module(x)
            return x
        def _feature_hook(self, module, input, output):
            # """
            # Hook function to capture the output of the last convolutional layer (before avgpool).
            # """
            self.registered_features = output
        def extract_features(self, x):
            """
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.avgpool.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features
    dim1 = 2048
    dim2=10
    model=Inception_v3(dim1=dim1,dim2=dim2)
    model.load_state_dict(newdict,strict=False)
    model.eval()
    return model



def load_DTD_pmodels(pname):
    pmodelpath = './../pmodels'
    modelpath = os.path.join(pmodelpath, pname)
    newdict = torch.load(modelpath)
    # newdict = {k: v for k, v in newdict.items() if 'classifier' not in k}

    class DenseNet(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(DenseNet, self).__init__()
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
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.features.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features

    dim1 = 1280
    dim2 = 8
    model = DenseNet(dim1=dim1, dim2=dim2)
    model.load_state_dict(newdict)
    model.eval()
    return model



def load_DTD_initmodels(pname):
    pmodelpath = './../initmodels'
    modelpath = os.path.join(pmodelpath, pname)
    newdict = torch.load(modelpath)
    # newdict = {k: v for k, v in newdict.items() if 'classifier' not in k}

    class DenseNet(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(DenseNet, self).__init__()
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
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.features.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features

    dim1 = 1280
    dim2 = 8
    model = DenseNet(dim1=dim1, dim2=dim2)
    model.load_state_dict(newdict)
    model.eval()
    return model


def load_DTD_cmodels(cname):
    pmodelpath = './../cmodels'
    modelpath = os.path.join(pmodelpath, cname)
    newdict = torch.load(modelpath)
    # newdict = {k: v for k, v in newdict.items() if 'classifier' not in k}

    class DenseNet(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(DenseNet, self).__init__()
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
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.features.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features
    dim1 = 1280
    dim2 = 2
    model = DenseNet(dim1=dim1, dim2=dim2)
    model.load_state_dict(newdict)
    model.eval()
    return  model

