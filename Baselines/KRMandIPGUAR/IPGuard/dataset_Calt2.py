import os
import torch, argparse, sys, json, warnings, re
from torch.utils.data import Dataset, DataLoader, random_split
warnings.simplefilter(action='ignore', category=UserWarning)
# from utils import *
# from basic import *
# from models.basic import *
from compare import get_feature, get_feature_by_name
from utilscalt import *
import re

def extract_calt_number(filename):
    """
    从文件名中提取第一个 "Calt_" 后的数字。
    
    参数:
        filename (str): 文件名字符串，例如 "Calt_2_Calt_10.pth.pth_ep4"
    
    返回:
        int: 提取到的数字，如果未匹配则返回 None。
    """
    pattern = r"Calt_(\d+)_"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    else:
        return None

def load_models_total_p(pre_train_ckpt_path_p):
    if 'Calt' in pre_train_ckpt_path_p:
        model=load_pmodels(pre_train_ckpt_path_p, 'Calt_p')
    elif 'Dogs' in pre_train_ckpt_path_p:
        model=load_pmodels(pre_train_ckpt_path_p, 'Dogs_p')
    else:
        model=load_pmodels(pre_train_ckpt_path_p, 'Flowers_p')
    return model


def load_models_total_c(pre_train_ckpt_path_c):
    if 'Calt' in pre_train_ckpt_path_c:
        model=load_cmodels(pre_train_ckpt_path_c, 'Calt_c')
    elif 'Dogs' in pre_train_ckpt_path_c:
        model=load_cmodels(pre_train_ckpt_path_c, 'Dogs_c')
    else:
        model=load_cmodels(pre_train_ckpt_path_c, 'Flowers_c')
    return model


def get_p_names(file_name):
    # 使用正则表达式匹配父模型名称部分
    if 'Dogs' in file_name:
        match = re.search(r'(Dogs_\d+\.pth)(?=\.)', file_name)
        if match:
            return match.group(1)  # 返回父模型名称
        else:
            raise ValueError(f"Invalid file name format: {file_name}")

    elif 'Calt' in file_name:
        match = re.search(r'(Calt_\d+\.pth)(?=\.)', file_name)
        if match:
            return match.group(1)  # 返回父模型名称
        else:
            raise ValueError(f"Invalid file name format: {file_name}")

    else:
        match = re.search(r'(Flowers_\d+\.pth)(?=\.)', file_name)
        if match:
            return match.group(1)  # 返回父模型名称
        else:
            raise ValueError(f"Invalid file name format: {file_name}")


class DetectorDataset(Dataset):
    def __init__(self, pmodelspath, cmodelspath, dataset, model, feature_index, weight_name,train_flag):
        super().__init__()
        self.train_flag=train_flag
        self.create_dataset(pmodelspath, cmodelspath, dataset, model, feature_index, weight_name)

    def get_weight(self, model, weight_names):
        for weight_name in weight_names.split("."):
            model = getattr(model, weight_name)
        return model

    def create_dataset(self, pmodelspath, cmodelspath, dataset, model, feature_index, weight_name):
        '''
            pfeatures: NumPModels x N x Nfeatures
            cfeatures: NumCModels x N x Nfeatures
            pweights: NumPModels x Nparams
            cweights: NumCModels x Nparams
        '''
        empty_dict = {}
        args = argparse.Namespace(**empty_dict)
        args.dataset = dataset
        args.batch_size = 64
        args.test_batch_size = 10
        args.model = model
        args.feature_index = feature_index
        args.finetune_flag = True
        # print(pmodelspath,cmodelspath,"pppcccmodelpath")
        # print(len(os.listdir(pmodelspath)),len(os.listdir(cmodelspath)),"len_p_c")

        # 获取预训练模型的路径列表
        # pre_train_ckpt_paths_c, pre_train_ckpt_paths_p, selec_class = generate_pre_list(pmodelspath, cmodelspath,
        #                                                                                 'Calt')
        totalc=os.listdir('./saved_features')
        totalp=os.listdir(pmodelspath)
        pre_train_ckpt_paths_c=[]
        pre_train_ckpt_paths_p=[]
        for pre_train_ckpt_path_c in totalc:
            if not 'Calt' in pre_train_ckpt_path_c:
                continue
            pre_train_ckpt_paths_c.append(pre_train_ckpt_path_c)
        for pre_train_ckpt_path_p in totalp:
            if not 'Calt' in pre_train_ckpt_path_p:
                continue
            pre_train_ckpt_paths_p.append(pre_train_ckpt_path_p)
       
       # # 这里可以根据需要进行切片，这里暂时使用完整列表
        pre_train_ckpt_paths_c = pre_train_ckpt_paths_c
        pre_train_ckpt_paths_p = pre_train_ckpt_paths_p
        # print(pre_train_ckpt_paths_c,pre_train_ckpt_paths_p,"pre_ckpt")
        # 数据集划分 8:2
        total_p = len(pre_train_ckpt_paths_p)
        split_idx_p = int(total_p * 0.8)
        print(total_p,split_idx_p,"totalp")
        # if self.train_flag == 'train':
        #     pre_train_ckpt_paths_p = pre_train_ckpt_paths_p[:split_idx_p]
        # else:
        #     pre_train_ckpt_paths_p = [p for p in pre_train_ckpt_paths_p[split_idx_p:] if 'Flower' in p]
        total_c = len(pre_train_ckpt_paths_c)
        print(total_c,"total+C")
        # split_idx_c = int(total_c * 0.8)
        # if self.train_flag == 'train':
        #     pre_train_ckpt_paths_c = pre_train_ckpt_paths_c[:split_idx_c]
        # else:
        #     pre_train_ckpt_paths_c = [c for c in pre_train_ckpt_paths_c[split_idx_c:] if 'Flower' in c]
        # print(total_p,total_c,"totp,totc")
        # self.pweights = []
        # self.pfeatures = []
        # parentdict = {}
        # i = -1
        # print(pre_train_ckpt_paths_c,pre_train_ckpt_paths_p,"pretrain")
        # for pre_train_ckpt_path_p in pre_train_ckpt_paths_p:
        #     print(pre_train_ckpt_path_p,"pre_traindp")
        #     # if not 'Flowers' in pre_train_ckpt_path_p:
        #     #     continue
        #     feature_path = f'./saved_features_p/{pre_train_ckpt_path_p}'
        #     if os.path.exists(feature_path):
        #         # 加载时指定 map_location='cpu' 确保数据加载到 CPU 上
        #         parentdict = torch.load(feature_path, map_location='cpu')
        #         print(parentdict['weights'].detach().cpu().shape,"shapeweight",
        #               parentdict['features'].detach().cpu().shape,"shapefeature")
        #         self.pweights.append(parentdict['weights'].detach().cpu())
        #         self.pfeatures.append(parentdict['features'].detach().cpu())
        #         continue

        #     train_data = get_train_data_loader(i)
        #     args.pre_train_ckpt = pre_train_ckpt_path_p
        #     modelp = load_models_total_p(pre_train_ckpt_path_p)
        #     print(modelp,"mmmodelp")
        #     weights = self.get_weight(modelp, weight_name).detach().cpu()
        #     print(weights.shape,"weight")
        #     torch.cuda.empty_cache()
        #     self.pweights.append(weights)
        #     features = get_feature_by_name(modelp, 'module.layer1.0.conv1', train_data)
        #     # print(features.shape, "pshape55")
        #     self.pfeatures.append(features)
        #     parentdict = {'weights': weights, 'features': features}
        #     torch.save(parentdict, './saved_features/' + pre_train_ckpt_path_p)
        #     del features
        #     del parentdict  # 删除临时变量，释放内存
        #     del modelp  # 删除模型，释放 GPU 内存
        #     torch.cuda.empty_cache()

        # self.pweights = torch.stack(self.pweights, dim=0).squeeze().detach().cpu()
        # self.pfeatures = torch.stack(self.pfeatures, dim=0).squeeze().squeeze().detach().cpu()
        ###因为是resnet的layer1.conv1，weight[64,64,3,3]
        # self.pweights = torch.stack([w.detach().reshape(64*64,3*3).cpu() for w in self.pweights], dim=0).squeeze()
        # self.pfeatures = torch.stack([f.detach().reshape(5,64,32*32).cpu() for f in self.pfeatures], dim=0).squeeze().squeeze()
        self.cweights = []
        self.cfeatures = []
        self.pweights=[]
        self.pfeatures=[]
        # i=0    
        for pre_train_ckpt_path_c in pre_train_ckpt_paths_c:
            feature_path = f'./saved_features/{pre_train_ckpt_path_c}'
            if os.path.exists(feature_path):
                parentdict = torch.load(feature_path, map_location='cpu')
                self.cweights.append(parentdict['weights'].detach().cpu())
                self.cfeatures.append(parentdict['features'].detach().cpu())
                continue
            i=extract_calt_number(pre_train_ckpt_path_c)
            train_data = get_train_data_loader(i)
            args.pre_train_ckpt = pre_train_ckpt_path_c
            modelc = load_models_total_c(pre_train_ckpt_path_c)
            weights = self.get_weight(modelc, weight_name).detach().cpu()
            self.cweights.append(weights)
            features = get_feature_by_name(modelc,'module.layer1.0.conv1', train_data)
            self.cfeatures.append(features)
            parentdict = {'weights': weights, 'features': features}
            torch.save(parentdict, feature_path)
        for pre_train_ckpt_path_c in pre_train_ckpt_paths_c:
            currcpweights=[]
            currcpfeatures=[]
            for pre_train_ckpt_path_p in pre_train_ckpt_paths_p:
                pfeature_path = f'./saved_features_p/{pre_train_ckpt_path_c[:-5]}/{pre_train_ckpt_path_p}'
                if not os.path.exists(f'./saved_features_p/{pre_train_ckpt_path_c[:-5]}'):
                    os.makedirs(f'./saved_features_p/{pre_train_ckpt_path_c[:-5]}')
                if os.path.exists(pfeature_path):
                    # print(pfeature_path,"pfeapath1")
                    pparentdict = torch.load(pfeature_path, map_location='cpu')
                    currcpweights.append(pparentdict['weights'].detach().cpu())
                    currcpfeatures.append(pparentdict['features'].detach().cpu())
                    continue
                # print(pfeature_path,"pfeapath")
                # # pname=get_p_names(pre_train_ckpt_path_c)
                # modelp = load_models_total_p(pre_train_ckpt_path_p)
                # weights = self.get_weight(modelp, weight_name).detach().cpu()
                # # self.pweights.append(weights)
                # features = get_feature_by_name(modelp,'module.layer1.0.conv1', train_data)
                # currcpfeatures.append(features)
                # currcpweights.append(weights)
                # pparentdict = {'weights': weights, 'features': features}
                # torch.save(pparentdict, pfeature_path)
            currcpweights = torch.stack([w.detach().reshape(64 * 3, 64 * 3).cpu() for w in currcpweights], dim=0).squeeze()
            currcpfeatures = torch.stack([f.detach().reshape(5, 64, 32 * 32).cpu() for f in currcpfeatures],
                                        dim=0).squeeze().squeeze()    
            print(currcpfeatures.shape,currcpfeatures.shape,"ablshape22")
            if currcpfeatures.shape[0]==19:
                print(pre_train_ckpt_path_c,"pre_train_ckpt_path_c","error")
            self.pweights.append(currcpweights)
            self.pfeatures.append(currcpfeatures)
        self.cweights = torch.stack([w.detach().reshape(64 * 3, 64 * 3).cpu() for w in self.cweights], dim=0).squeeze()
        self.cfeatures = torch.stack([f.detach().reshape(5, 64, 32 * 32).cpu() for f in self.cfeatures],
                                     dim=0).squeeze().squeeze()
        self.label_dict, self.labels = get_real_parents(pre_train_ckpt_paths_p, pre_train_ckpt_paths_c)
        self.nump=len(list(self.label_dict.keys()))
        # 调整形状，保持数据在 CPU 上
        for index in range(len(self.labels)):
            nump = 20
            n, w, h = self.cfeatures[index].size()
            ww, wh = self.cweights[index].size()
            # print(nump,"nump",n,"numsamp",index,self.pfeatures[index].shape,"shape77")
            cfeature = self.cfeatures[index].reshape(n, -1).unsqueeze(0).unsqueeze(0).expand((nump, 1, -1, -1))
            cweight = self.cweights[index].reshape(-1).unsqueeze(0).unsqueeze(0).expand((nump, 1, -1, -1))
            # nump, n, w, h = self.pfeatures[index].size()
            print(self.pfeatures[index].shape,'shape111')
            print(cfeature.shape,self.pfeatures[index].reshape(nump,n,-1).unsqueeze(1).shape,"shape222" )
            # print(cfeature.shape,cweight.shape, self.pfeatures[index].reshape(nump,n,-1).unsqueeze(1).shape,self.cfeatures[index].shape,"shape444")
            # feature = torch.cat((self.pfeatures[index].reshape(nump, n, -1).unsqueeze(1), cfeature), dim=1)
            # weight = torch.cat((self.pweights[index].reshape(nump, -1).unsqueeze(1).unsqueeze(1), cweight), dim=1)
            # label = self.labels[index]

    def __getitem__(self, index):
        '''
            feature: NumPModels x N x Nfeatures
            weight: NumPModels x Nparams
        '''
        nump = self.nump
        n,w,h=self.cfeatures[index].size()
        ww,wh= self.cweights[index].size()
        cfeature = self.cfeatures[index].reshape(n,-1).unsqueeze(0).unsqueeze(0).expand((nump, 1, -1, -1))
        cweight = self.cweights[index].reshape(-1).unsqueeze(0).unsqueeze(0).expand((nump, 1, -1, -1))
        nump,n,w,h=self.pfeatures[index].size()
        # print(cfeature.shape,self.pfeatures[index].reshape(nump,n,-1).unsqueeze(1).shape,"shape222" )
        # print(cfeature.shape,cweight.shape, self.pfeatures[index].reshape(nump,n,-1).unsqueeze(1).shape,self.cfeatures[index].shape,"shape444")
        feature = torch.cat((self.pfeatures[index].reshape(nump,n,-1).unsqueeze(1), cfeature), dim=1)
        weight = torch.cat((self.pweights[index].reshape(nump,-1).unsqueeze(1).unsqueeze(1), cweight), dim=1)
        label = self.labels[index]

        return feature, weight, label

    def __len__(self):
        return len(self.cfeatures)


def get_detector_loader(args, shuffle=True):
    print(args,"aargs")
    print("Get_detec")
    train_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': 4,
        'shuffle': shuffle,
        'pin_memory': True
    }
    test_kwargs = {
        'batch_size': args.test_batch_size,
        'num_workers': 4,
        'shuffle': False,
        'pin_memory': True
    }
    print("dataset",test_kwargs)
    dataset = DetectorDataset(
        args.pmodelspath,
        args.cmodelspath,
        args.dataset,
        args.model,
        args.feature_index,
        args.weight_name,
        train_flag=True
    )
    print(len(dataset),"len_dataset")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # print(len(train_dataset),len(test_dataset),"llenda")
    train_loader = DataLoader(dataset, **train_kwargs,drop_last=True)
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return train_loader, test_loader, test_loader, dataset.nump


if __name__ == "__main__":
    pmodelspath = './../../../../szy2025/train/train_res18/NONIMAGENETPmodels'
    cmodelspath = './../../../../szy2025/train/train_res18/NONCmodels'
    dataset_name = 'CIFAR'
    weight_name = 'module.layer1.0.conv1.weight'
    feature_index = 0

    # 创建 DetectorDataset（所有数据均保存在 CPU 上）
    dataset_obj = DetectorDataset(pmodelspath, cmodelspath, dataset_name, None, feature_index, weight_name,train_flag=True)

    # 这里使用 num_workers=0 避免多进程带来的额外内存开销
    train_loader = DataLoader(
        dataset_obj,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    device = 'cuda:0'
    start_index = 0

    # 使用 no_grad() 避免梯度信息占用内存
    with torch.no_grad():
        for batch_idx, (feature, weight, label) in enumerate(train_loader, start_index):
            # 将 batch 移到 GPU 上进行必要的处理
            feature = feature.to(device)
            weight = weight.to(device)
            label = label.to(device)

            # 此处可以插入对 feature, weight, label 的进一步处理或模型推理代码
            print(
                f"Batch {batch_idx}: feature shape: {feature.shape}, weight shape: {weight.shape}, label shape: {label.shape}")

            # 处理完成后，将数据移回 CPU 并释放 GPU 内存
            feature = feature.cpu()
            weight = weight.cpu()
            label = label.cpu()
            del feature, weight, label
            torch.cuda.empty_cache()