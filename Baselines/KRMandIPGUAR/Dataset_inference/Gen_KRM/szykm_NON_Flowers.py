import os
###为 Dogs 生成边界样本
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import torch
import torch.nn as nn
from embedding_compute import adv_attack
import torchvision
from torch.utils.data import DataLoader, Dataset
import re
import torchvision.utils as vutils
from torchvision import datasets, transforms
# from  train_caltech import  *
from flowers import *
import random
import argparse


def load_pmodels(pname):
    pmodelpath = './../../../../szy2025/train/train_res18/NONIMAGENETPmodels'
    modelpath = os.path.join(pmodelpath, pname)
    newdict = torch.load(modelpath)

    dim1 = 512
    dim2 = 6

    class Res18(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(Res18, self).__init__()
            self.module = torchvision.models.resnet18(pretrained=True)
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

    model = Res18(dim1=dim1, dim2=dim2).to(args.device)
    model.eval()
    model.load_state_dict(newdict)
    return model


def set_seed(seed):
    """
    Set the random seed for reproducibility in PyTorch, NumPy, and Python's random module.
    Specifically for a single GPU environment.

    Args:
        seed (int): The seed value to set.
    """
    # Set Python, NumPy, and PyTorch random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Set CUDA-related seeds if GPU is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # This ensures all devices use the same seed
        torch.backends.cudnn.deterministic = True  # Ensures determinism
        torch.backends.cudnn.benchmark = False  # Disables optimizations for non-deterministic algorithms


def get_data(args, loader, model, spidx):
    x_list = []
    i_list = []
    fea_list = []

    for i in range(args.num_class):
        x_list.append([])
        i_list.append([])
        fea_list.append([])
    model = model.to('cuda:0')
    for batch_idx, (data, target) in enumerate(loader):
        images = data.to(args.device)
        outputb_a = model(images)
        # print(outputb_a.shape,"out_b_a")
        # top2_values, top2_indices = torch.topk(outputb_a, k=2, dim=1)
        # labels,tarlabels=top2_indices[:,0],top2_indices[:,1]
        YY_tgt = torch.argmax(outputb_a, dim=1)
        # YY_tgt=labels
        XX_tgt = images
        # Tar_tgt=tarlabels
        num = YY_tgt.shape[0]
        for idx in range(num):
            yy_tgt = YY_tgt[idx]
            x_list[yy_tgt].append(XX_tgt[idx])

    # generate the center point of each class
    n_class = args.num_class
    for class_idx in range(n_class):
        print(len(x_list[class_idx]), "len_idx")
    total_centers_list = []
    # 遍历每个类别
    # for idx in range(5):
    x_center = torch.zeros(n_class, 3, 128, 128)
    for class_idx in range(n_class):
        l_subset = x_list[class_idx]  # 获取当前类别的样本列表
        l_tensor = torch.stack(l_subset).to(args.device)  # 将子列表转换为张量
        distance_matrix = torch.cdist(l_tensor.unsqueeze(0), l_tensor.unsqueeze(0), p=2).squeeze(0)
        # 每个样本到其他样本的距离之和 (形状: [len(l_subset)])
        sum_distances = torch.sum(distance_matrix, dim=(-3, -2, -1))  # 注意这里是计算二维矩阵的和
        mdx = torch.argmax(sum_distances).item()  # 找到距离之和最大的样本索引
        x_center[class_idx] = l_subset[mdx]  # 更新类别中心点
        total_centers_list.append(x_center)

    np.set_printoptions(threshold=np.inf)

    return total_centers_list


# generate the sub-center point of each class

def get_embeddings_black(args, x_list, loader, model, flag):
    print("Getting MinAD_KRM embeddings...")
    # set target adversarial sample list
    if flag == 0:
        tgt_list = x_list
    else:
        tgt_list, flag = set_target(model, loader, args.num_class, args.device, x_list)
    # define a matrix that stores distances
    distance_list = torch.zeros(args.num_class, args.num_class, 3, device=args.device)
    # define a matrix that stores adversarial examples
    adv_list = torch.zeros(args.num_class, args.num_class, 3, 128, 128, device=args.device)

    model = model.to('cuda:0')
    for yy in range(len(x_list)):
        x1 = x_list[yy]
        xx = x1.unsqueeze(0).to(args.device)
        yy = torch.tensor([yy]).to(args.device)
        start = time.time()
        for class_idx in range(args.num_class):
            adv_idx = class_idx
            adv_label = torch.tensor([adv_idx], device=args.device)
            if adv_label == yy:
                continue
            tgt = tgt_list[adv_idx]
            # model->目标模型 xx->目标数据输入 yy->目标数据标签, tgt_tgt->目标攻击初始对抗数据
            xx, yy, tgt, adv_label = xx.to('cuda:0'), yy.to('cuda:0'), tgt.to('cuda:0'), adv_label.to('cuda:0')
            # 检查 xx 或者 tgt_tgt 是否为 0
            if torch.all(xx == 0) or torch.all(tgt == 0):
                continue  # 如果条件满足，则跳过此次循环的剩余部分
            adv, d = adv_attack(args, model, xx, yy, tgt, adv_label)
            distance_list[yy, class_idx] = d
            adv_list[yy, class_idx] = adv.squeeze(0)
        print('MinAD KRM|| yy_label:{}, distance:{}, t:{}'.format(yy, distance_list[yy][:][:][0], time.time() - start))

    full_d = distance_list
    return full_d, adv_list


# *** main function for model knowledge representation ***/

def get_knowledge(args, model, train_loader):
    # 3. get and load center points and sub-center points(if need)
    total_centers = get_data(args, train_loader, model, 0)
    train_d = 0
    advlists = []
    train_ds = []
    for x_centers in total_centers:
        train_db_a, advlist_b_a = get_embeddings_black(args, x_centers, train_loader, model, 0)
        train_ds.append(train_db_a)
        advlists.append(advlist_b_a)
    return advlists, total_centers, train_d
    # torch.save(train_d, f"{args.feature_dir}/train_{args.feature_mode}{args.num_center}_{taskname}.pt")


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
        self.parser.add_argument('--batch-size', type=int, default=20, help='batch size')
        self.parser.add_argument('--qdatanum', type=int, default=3, help='batch size')  # 一个类有多少图像
        self.parser.add_argument('--n_epochs', type=int, default=5000, help='dimension')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        ###############################################
        self.parser.add_argument('--num_class', type=int, default=6)
        self.parser.add_argument("--num_samples", help="number of sample for embedding", type=int, default=50)
        self.parser.add_argument("--num_attack_iter", help="number of iterations of MinAD", type=int, default=50)
        self.parser.add_argument("--num_evals_boundary", help="number of queries per iteration of MinAD", type=int,
                                 default=1000)
        self.parser.add_argument("--attack_lr_begin", help="initial LR of MinAD", type=float, default=16)
        self.parser.add_argument("--attack_lr_end", help="LR lower threshold of MinAD", type=float, default=0.4)

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args


def seed_everything(seed):
    """
    固定所有可能的随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_target(model, loader, num_class, device, x_center):
    ## Find the corresponding adversarial target sample($\tilde{x}_0$) for each class
    tgt_list = []
    flag = 1
    for tgt_label in range(num_class):
        if torch.all(x_center[tgt_label]) == 0:
            tgt_list.append(x_center[tgt_label])
        else:
            tgt_label = torch.tensor(tgt_label)
            tgt_temp = 0
            for idb, (XX_tgt, YY_tgt) in enumerate(loader):
                for i_yy in range(len(YY_tgt)):
                    # find the sample
                    orilabel = torch.argmax(model(XX_tgt[i_yy].unsqueeze(0).to(device)))
                    if tgt_label == orilabel.cpu():
                        tgt_tgt = XX_tgt[i_yy].unsqueeze(0).to(device)
                        # add the sample
                        tgt_list.append(tgt_tgt)
                        tgt_temp = 1
                        break
            if tgt_temp == 0:
                print("error")
    if len(tgt_list) == num_class:
        flag = 0

    return tgt_list, flag


if __name__ == "__main__":

    parse = Parser().parse()
    args = parse
    device = args.device
    pmodelpath = './../../../../szy2025/train/train_res18/NONIMAGENETPmodels'

    set_seed(111)
    models = []
    model_indexdict = {}
    pnames = []
    # oriclas
    print(oriclass,"oriclass")
    for i in range(len(oriclass)):
        pname = f'Flowers_{i}.pth'
        pnames.append(pname)
        model_indexdict[pname] = oriclass[i]
    print(pnames)
    # 2. load the student model
    ######生成子模型的对抗样本
    parentmodels = list(model_indexdict.keys())
    for parentname in parentmodels:
        if 'Flowers_6.pth' in parentname:
            continue
        if not 'Flowers' in parentname:
            continue
        if os.path.exists(os.path.join('./Flowers_adv', parentname)):
            continue
        parentmodel = load_pmodels(parentname)
        print(parentname,model_indexdict[parentname],"iii")
        train_loader, _ = get_data_loader(model_indexdict[parentname])
        advlists, centers, train_d = get_knowledge(args, parentmodel, train_loader)
        dict0 = {'adv': advlists, 'xcenter': centers, 'traind': train_d}
        torch.save(dict0, os.path.join('./NON_Flowers_adv', parentname))