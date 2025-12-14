#使用CALTECH101训练模型，20个6分类作为父亲模型，共60个4分类作为子模型
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
from caltech101 import *
####SZY   ./Caltmodels/{tt}_{epoch}.pth':每一步微调得到的模型

def set_seed(seed_value):
    """Set seed for reproducibility.
    Args:
        seed_value (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed_value)
    # Set the seed for NumPy
    np.random.seed(seed_value)
    # Set the seed for PyTorch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cmodel3(args, curmodelname):
    dim2=6
    if 'resnet50' in curmodelname:
        dim1 = 2048
        class Res50(torch.nn.Module):

            def __init__(self, dim1,dim2):
                super(Res50, self).__init__()
                self.module =torchvision.models.resnet50(pretrained=True)
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

        model = Res50(dim1=dim1, dim2=dim2).to(args.device)
        model.eval()

    elif 'resnet18' in curmodelname:
        dim1 = 512
        class Res18(torch.nn.Module):

            def __init__(self, dim1,dim2):
                super(Res18, self).__init__()
                self.module =torchvision.models.resnet18(pretrained=True)
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
                last_conv = list(self.module.features.children())[-1]
                handle = last_conv.register_forward_hook(self._feature_hook)

                with torch.no_grad():
                    self.forward(x)

                handle.remove()
                return self.registered_features

        model = Res18(dim1=dim1, dim2=dim2).to(args.device)
        model.eval()

    elif 'mobilenet' in curmodelname:
        dim1 = 1280
        class Mobilenet(torch.nn.Module):

            def __init__(self, dim1,dim2):
                super(Mobilenet, self).__init__()
                self.module =torchvision.models.mobilenet_v2(pretrained=False)
                self.module.add_module('classifier', torch.nn.Linear(dim1,dim2))
            def forward(self,x):
                x=self.module(x)
                return x
            def _feature_hook(self, module, input, output):
                # """
                # Hook function to capture the output of the last convolutional layer (before avgpool).
                # """
                self.registered_features = output
 
            def extract_features(self, x):
                last_conv = list(self.module.features.children())[-1]
                handle = last_conv.register_forward_hook(self._feature_hook)

                with torch.no_grad():
                    self.forward(x)

                handle.remove()
                return self.registered_features
        model = Mobilenet(dim1=dim1, dim2=dim2).to(args.device)
        model.eval()
  

    else:
        raise Exception
    model = model.to(device=args.device)
    print("model_down_finish")
    total = sum([param.nelement() for param in model.parameters()])
    # print("All parameters: %.2fM" % (total / 1e6))
    return model,dim1,dim2



def train(model, train_dataloader, test_dataloader, device, tt):
    # Save the initial model in initmodels directory

    criterion = torch.nn.CrossEntropyLoss()
    classifier = model.to(device)  # Move model to the specified device
    # 随机选择 epoch 和 lr
    random_epoch = random.randint(60, 100)
    # random_lr = 10 ** random.uniform(-4, -3)  # 生成 0.00001 ~ 0.0001 之间的随机值
    random_lr=0.001
    optimizer = optim.Adam(classifier.parameters(), lr=random_lr)
    classifier.train()  # Set model to training mode

    # Training loop
    for epoch in range(random_epoch):
        for i, (images, batch_labels) in enumerate(train_dataloader):
            images = images.to(device)
            batch_labels = batch_labels.to(device)  # Ensure batch_labels are on the correct device
            pred_label = classifier(images)
            loss = criterion(pred_label, batch_labels)  # Use batch_labels instead of new_batch_labels
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate every 20 epochs
        if (epoch + 1) % 5 == 0:
            classifier.eval()  # Set model to evaluation mode
            with torch.no_grad():
                total = 0
                correct = 0
                for i, (images, labels) in enumerate(test_dataloader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = classifier(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
                val_acc = 100 * correct / total  # Compute validation accuracy
                print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}, Acc: {val_acc:.2f}%')
                # Save the trained model in pmodels directory
                if epoch + 1==60:
                    trained_model_path = f'./Pmodels/Calt_{tt}.pth'
                    torch.save(classifier.state_dict(), trained_model_path)
            classifier.train()  # Set model back to training mode after evaluation

    return val_acc


def test1(model ,test_dataloader,neuname,device):
    total=0
    correct=0
    with torch.no_grad():
        for i ,(inputs,labels) in enumerate(test_dataloader):
            inputs,labels=inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total  # 测试准确率
        print('Acc: {val_acc:.2f}%',val_acc)

    return val_acc



def main():

    class Parser:
        def __init__(self):
            self.parser = argparse.ArgumentParser()
            self.set_arguments()

        def str2bool(self, s):
            return s.lower() in ['true', 't']

        def set_arguments(self):
            ##############################################
            self.parser.add_argument('--device', type=str, default='cuda:0', help='gpus to use, i.e. 0')
            self.parser.add_argument('--mode', type=str, default='test', help='i.e. train, test')
            self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')

        def parse(self):
            args, unparsed = self.parser.parse_known_args()
            if len(unparsed) != 0:
                raise SystemExit('Unknown argument: {}'.format(unparsed))
            return args

    def add_num_class_to_args(args, num_class):
        # 创建一个新的命名空间，包含原始args的所有属性和新的num_class属性
        new_args = argparse.Namespace(**vars(args))
        setattr(new_args, 'num_class', num_class)
        return new_args

    parse = Parser().parse()
    args = parse
    set_seed(args.seed)
    ###transform

    def train_ftmodels(select_class,i):

        train_loader,test_loader=get_data_loader(select_class)
        model,dim1,dim2=get_cmodel3(args, 'mobilenet')
        init_model_path = './initmodels/Calt_' + str(i) + '.pth'
        torch.save(model.state_dict(), init_model_path)
        acc = train(model,train_loader,test_loader,args.device,i)
  
    # select_classes=[[81, 14, 3, 94, 35, 31, 28, 17], [94, 13, 86, 69, 11, 75, 54, 4], [3, 11, 27, 29, 64, 77, 71, 25],
    #   [91, 83, 89, 69, 53, 28, 57, 75], [35, 0, 97, 20, 89, 54, 43, 19], [27, 97, 43, 13, 11, 48, 12, 45]]
    select_classes=oriclass
    i=0
    for select_class in select_classes:
        train_ftmodels(select_class,i)
        i=i+1


if __name__ == '__main__':
    main()