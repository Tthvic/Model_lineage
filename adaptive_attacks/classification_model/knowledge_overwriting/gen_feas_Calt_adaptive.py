import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
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
from src.task_vectors import TaskVector
import re
from copy import deepcopy
import torch.nn as nn
from load_mobilenet import get_p_names,load_pmodels,load_initmodels,dimdict
MODELPATH='./C_adaptived/noise0.5'
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

def load_wcmodels(pname):
           # 获取当前文件所在目录
    pmodelpath=MODELPATH
    modelpath = os.path.join(pmodelpath, pname)
    newdict = torch.load(modelpath)
    # newdict = {k: v for k, v in newdict.items() if 'classifier' not in k}
    dim1 = 1280
    dim2=4
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




# 比较两模型的参数
def compare_models(model1, model2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass  # 参数相同
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print(f'Layer Mismatch: {key_item_1[0]}')
                return
            else:
                print(key_item_1[0], key_item_2[0], "kkk")
                raise Exception("Models have different keys, cannot compare.")
    if models_differ == 0:
        print("The two models have identical parameters.")


# 调用函数
# test_initmodels()
    # 调用函数
    # test_initmodels()


def normalize_vector(vector):
    """对任务向量进行 L2 正则化"""
    if isinstance(vector, dict):  # 如果是 OrderedDict
        for param_name, param_value in vector.items():
            if isinstance(param_value, torch.Tensor):
                # 转换为浮点数类型
                param_value = param_value.float()
                # 对每个参数进行 L2 正则化
                norm = param_value.norm(p=2)  # 计算 L2 范数
                if norm != 0:  # 防止除以零
                    vector[param_name] = param_value / norm  # L2 正则化
    return vector


def normalize_and_add_vectors(init_vector, currtask_vector):
    """归一化并加和两个任务向量"""
    # 归一化
    # init_vector.vector = normalize_vector(init_vector.vector)
    # currtask_vector.vector = normalize_vector(currtask_vector.vector)
    # 加和任务向量
    inited_task_vector = init_vector + currtask_vector
    return inited_task_vector


def check_and_print_model_weights(model):
    """检查并打印模型权重的最大值、最小值和均值"""
    print("\nChecking model weights:")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():  # 检查权重中是否存在 NaN
            print(f"Detected NaN in parameter: {name}")
        else:
            print(f"{name} - Max: {param.data.max()}, Min: {param.data.min()}, Mean: {param.data.mean()}")


def gen_feas(pname,fcname,childname):
    # 加载模型
    torch.cuda.empty_cache()
    with torch.no_grad():
        pmodel = load_pmodels(pname,'Calt_P','P')
        pmodel.to('cuda:0')
        pmodel.eval()
        pmodel_copy = deepcopy(pmodel)
        tmps=torch.load(os.path.join('./M_Calt_adv',pname))
        advlistss,centers,train_d=tmps['adv'],tmps['xcenter'],tmps['traind']
        print(len(centers),"len_centers",len(advlistss),"lenadvl")
        cmodel = load_wcmodels(childname)
        cmodel.to('cuda:0')
        cmodel.eval()
        # # 初始化模型
        initmodel = load_initmodels(pname)
        initmodel.to('cuda:0')
        initmodel.eval()

        fcmodel = load_wcmodels(fcname)
        fcmodel.to('cuda:0')
        fcmodel.eval()
        # 定义 TaskVector
        currtask_vector = TaskVector(cmodel, initmodel)
        modelb_a = currtask_vector.apply_to(pmodel, scaling_coef=0.5)
        modelb_a.eval()
        pdicts=[]
        dim=dimdict['Calt_P']
        for samplenum in range(len(centers)):
            # if samplenum > 5:
            #     continue
            advlists = advlistss[samplenum]
            for i in range(dim):
                advlists[i][i] = centers[samplenum][i]
            advlists = advlists.reshape(-1, 3, 128, 128).to('cuda:0')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # # 对张量进行归一化
            advlists = normalize(advlists)
            cadvfea = cmodel.extract_features(advlists)
            padvfea = pmodel.extract_features(advlists)
            b_afea = modelb_a.extract_features(advlists)
            if torch.isnan(b_afea).any():  # 检查参数中是否存在 nan
                print(f"Detected NaN in parameter")
            feadict={'cfea':cadvfea,'pfea':padvfea,'b_afea':b_afea}
            pdicts.append(feadict)
        # 定义 TaskVector
        fcurrtask_vector = TaskVector(fcmodel, initmodel)
        fmodelb_a = fcurrtask_vector.apply_to(pmodel_copy, scaling_coef=0.5)  # 检查模型权重的范围
        fmodelb_a.eval() # # 检查输入数据是否有异常

        fdicts=[]
        for samplenum in range(len(centers)):
            if samplenum>5:
                continue
            advlists=advlistss[samplenum]
            for i in range(dim):
                advlists[i][i]=centers[samplenum][i]
            advlists=advlists.reshape(-1, 3, 128, 128).to('cuda:0')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # # 对张量进行归一化
            advlists=normalize(advlists)
            fb_afea=fmodelb_a.extract_features(advlists)
            fcadvfea = cmodel.extract_features(advlists)
            fpadvfea = pmodel.extract_features(advlists)
            if torch.isnan(fb_afea).any():  # 检查参数中是否存在 nan
                    print(f"Detected NaN in parameter")
            ffeadict={'fcfea':fcadvfea,'pfea':fpadvfea,'fb_afea':fb_afea}
            fdicts.append(ffeadict)
        for i in range(len(fdicts)):
            pfea=pdicts[i]['pfea']
            cfea=pdicts[i]['cfea']
            b_afea=pdicts[i]['b_afea']
            ffea=fdicts[i]['pfea']
            fcfea=fdicts[i]['fcfea']
            fb_afea=fdicts[i]['fb_afea']
            dict={'pfea':pfea,'cfea':cfea,'b_afea':b_afea,'ffea':ffea,'fcfea':fcfea,'fb_afea':fb_afea}
            torch.save(dict,'./Adaptive_Noise50/'+childname+str(i))
            # sim= torch.nn.functional.cosine_similarity(pfea, ffea).cpu().numpy()
            ####f就是false
            
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
    # prunelist=['prune_10','prune_30','prune_50','prune_70']
    totalchildnames = os.listdir(MODELPATH)
    # Caltadvslists=os.listdir('./Calt_adv')
    i=0
    # test_initmodels()
    childnames=[]
    for childname in totalchildnames:
        if not 'Calt' in childname:
            continue
        childnames.append(childname)
    for childname in childnames:
        if not 'Calt' in childname:
            continue
        flag=0
        pname=get_p_names(childname)
        fcname=pname
        print(childname,pname,"pname")
        while flag==0:
            fcname = random.choice(childnames)
            fpname=get_p_names(fcname)
            if not fpname==pname:
                flag=1
        print(childname, "childname",fcname,pname,fpname,"nameee")
        gen_feas(pname, fcname, childname)


if __name__ == '__main__':
    main()