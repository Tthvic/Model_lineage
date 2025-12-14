import os
import torch
import re
from torch.utils.data import Dataset,TensorDataset, DataLoader, random_split, Subset
import random


def simplify_filename(file_name):
    """
    Simplifies the file name by removing the extra '.pth' part using string operations.

    :param file_name: The input file name to process.
    :return: The simplified file name.
    """
    # Split the file name into parts
    parts = file_name.split(".")

    if len(parts) >= 3 and parts[-2] == "pth":  # Check if there are redundant 'pth' parts
        # Reconstruct the simplified file name
        simplified_name = f"{'.'.join(parts[:-2])}.{parts[-1]}"
        return simplified_name
    else:
        return file_name  # Return the original name if no simplification is needed


# L2 归一化函数
# def l2_normalize(tensor):
#     norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
#     return tensor / (norm + 1e-8)

def l2_normalize(tensor, min_val=None, max_val=None):
    """
    Perform Min-Max normalization on the last dimension of a tensor.

    :param tensor: Input tensor of shape (..., d).
    :param min_val: Minimum value for normalization (optional).
    :param max_val: Maximum value for normalization (optional).
    :return: Min-Max normalized tensor.
    # """
    # if min_val is None:
    #     min_val = tensor.min(dim=-1, keepdim=True)[0]
    # if max_val is None:
    #     max_val = tensor.max(dim=-1, keepdim=True)[0]
    # return (tensor - min_val) / (max_val - min_val + 1e-8)
    if torch.isnan(tensor).any():  # 检查权重中是否存在 NaN
        print(f"Detected NaN in parameter:BAFEA")
    return  tensor


def get_p_names(file_name):
    # 使用正则表达式匹配父模型名称部分
    match = re.search(r"Calt_\d+_(.+?)$", file_name)
    if match:
        return match.group(1)  # 返回父模型名称
    else:
        raise ValueError(f"Invalid file name format: {file_name}")


 # Check and pad features to [64, 512] if necessary
# 填充到 (N, 512) 形状的函数
# 填充到 (8,8,512) 或 (6,8,512) 形状的函数
def pad_to_64_512(tensor):
    # print(tensor.shape,"tensor_shape")
    if tensor.shape[0] == 64:
        return tensor.reshape(8, 8, 512)
    elif tensor.shape[0] == 36:
        padded_tensor = torch.zeros((6, 8, 1280), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, :6, :] = tensor.reshape(6, 6, 1280)
        return padded_tensor
    else:
        raise ValueError("Unsupported tensor shape for padding")



class EmbeddingDataset(Dataset):
    def __init__(self, trainflag=True):
        self.trainflag=trainflag
        self.datapath='./../train/train_mobilenet/Feas'
        # self.cmodels = [name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
        #          for name in os.listdir('./../train/train_mobilenet/Cmodels')]
        self.flowers_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
                 for name in os.listdir('./../train/train_mobilenet/Feas/Flowers')]
        self.calt_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
            for name in os.listdir('./../train/train_mobilenet/Feas/Calt')]
        self.tiny_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
            for name in os.listdir('./../train/train_mobilenet/Feas/TINYIMG')]
        self.flowers_models=list(set(self.flowers_models))
        self.calt_models=list(set(self.calt_models))
        self.tiny_models=list(set(self.tiny_models))
        # self.flowers_models=[name for name in self.selcflowers if 'Flowers' in name]
        # self.calt_models=[name for name in self.selccalt if 'Calt' in name]
        # self.tiny_models=[name for name in self.selctiny if 'TINY' in name]
        self.train_set=[]
        self.test_set=[]
        for model_group in [self.flowers_models, self.calt_models, self.tiny_models]:
            random.shuffle(model_group)  # 随机打乱
            split_idx = int(0.6 * len(model_group)) if len(model_group) > 1 else 0  # 确保至少有一个样本用于训练
            self.train_set.extend(model_group[:split_idx])
            self.test_set.extend(model_group[split_idx:])
            print(len(self.test_set),len(self.train_set),"set")
        print(self.test_set,"Test_set",self.train_set,"tran")
        self.dataset,self.falsedataset = self.load_data_total()

    def __len__(self):
        """Returns the total number of models."""
        return len(self.dataset)

    def load_data_total(self,):
        dataset=[]
        falsedataset=[]
        nonCalt_fea_path='Calt'
        dataset,falsedataset=self.load_data(nonCalt_fea_path,dataset,falsedataset)
        nonDogs_fea_path='TINYIMG'
        dataset,falsedataset=self.load_data(nonDogs_fea_path,dataset,falsedataset)
        nonFlowers_fea_path='Flowers'
        dataset,falsedataset=self.load_data(nonFlowers_fea_path,dataset,falsedataset)

        return dataset,falsedataset


    def load_data(self,eachdatapath,dataset,falsedataset):
        datapath = os.path.join(self.datapath,eachdatapath)
        for cname in os.listdir(datapath):  #####顺序：b_a,child,parent
            # print(cname,cname[:-1],"nnn")
            ccname=cname.split('_epoch')[0] + '.pth' 
            if self.trainflag==True and ccname in self.test_set:
                continue
            if self.trainflag==False and ccname in self.train_set:
                print(ccname,"ccname")
                continue          
            cdata=torch.load(os.path.join(datapath,cname))
            b_afea=l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea=l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea=l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea=l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea=l2_normalize(cdata['fcfea'].squeeze().squeeze())
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)
            for i in range(b_afea.shape[0]):
                dataset.append((cname,[b_afea[i],c_fea[i],p_fea[i]] ))
                falsedataset.append((cname,[fb_afea[i],fcfea[i],p_fea[i]]))
            if torch.isnan(b_afea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:BAFEA")
            if torch.isnan(p_fea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:PFEA")
            if torch.isnan(c_fea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:C")
            if torch.isnan(fb_afea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:FBA")

        return dataset,falsedataset


    def __getitem__(self, index):
        poscname, pos_sample = self.dataset[index]
        negcname, neg_sample = self.falsedataset[index]
        return poscname,negcname,pos_sample, neg_sample





class TESTEmbeddingDataset(Dataset):
    def __init__(self, trainflag=True):
        self.trainflag=trainflag
        # self.datapath='./../train/train_mobilenet/WCRFea_Calt/prune70'###参数剪枝
        self.datapath='./../train/train_mobilenet/Adaptive_Noise50'###知识复写
        # self.cmodels = [name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
        #          for name in os.listdir('./../train/train_mobilenet/Cmodels')]
        self.flowers_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
                 for name in os.listdir('./../train/train_mobilenet/Feas/Flowers')]
        self.calt_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
            for name in os.listdir('./../train/train_mobilenet/Feas/Calt')]
        self.tiny_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
            for name in os.listdir('./../train/train_mobilenet/Feas/TINYIMG')]
        self.flowers_models=list(set(self.flowers_models))
        self.calt_models=list(set(self.calt_models))
        self.tiny_models=list(set(self.tiny_models))
        # self.flowers_models=[name for name in self.selcflowers if 'Flowers' in name]
        # self.calt_models=[name for name in self.selccalt if 'Calt' in name]
        # self.tiny_models=[name for name in self.selctiny if 'TINY' in name]
        self.train_set=[]
        self.test_set=[]
        for model_group in [self.flowers_models, self.calt_models, self.tiny_models]:
            random.shuffle(model_group)  # 随机打乱
            split_idx = int(0.6 * len(model_group)) if len(model_group) > 1 else 0  # 确保至少有一个样本用于训练
            self.train_set.extend(model_group[:split_idx])
            self.test_set.extend(model_group[split_idx:])
            print(len(self.test_set),len(self.train_set),"set")
        print(self.test_set,"Test_set",self.train_set,"tran")
        self.dataset,self.falsedataset = self.load_data_total()

    def __len__(self):
        """Returns the total number of models."""
        return len(self.dataset)

    def load_data_total(self,):
        dataset=[]
        falsedataset=[]
        # nonCalt_fea_path='Calt'
        nonCalt_fea_path=''
        dataset,falsedataset=self.load_data(nonCalt_fea_path,dataset,falsedataset)
        # nonDogs_fea_path='TINYIMG'
        # dataset,falsedataset=self.load_data(nonDogs_fea_path,dataset,falsedataset)
        # nonFlowers_fea_path='Flowers'
        # dataset,falsedataset=self.load_data(nonFlowers_fea_path,dataset,falsedataset)

        return dataset,falsedataset


    def load_data(self,eachdatapath,dataset,falsedataset):
        datapath = os.path.join(self.datapath,eachdatapath)
        for cname in os.listdir(datapath):  #####顺序：b_a,child,parent
            # print(cname,cname[:-1],"nnn")
            ccname=cname.split('_epoch')[0] + '.pth' 
            if self.trainflag==True and ccname in self.test_set:
                continue
            if self.trainflag==False and ccname in self.train_set:
                print(ccname,"ccname")
                continue          
            cdata=torch.load(os.path.join(datapath,cname))
            b_afea=l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea=l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea=l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea=l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea=l2_normalize(cdata['fcfea'].squeeze().squeeze())
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)
            for i in range(b_afea.shape[0]):
                dataset.append((cname,[b_afea[i],c_fea[i],p_fea[i]] ))
                falsedataset.append((cname,[fb_afea[i],fcfea[i],p_fea[i]]))
            if torch.isnan(b_afea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:BAFEA")
            if torch.isnan(p_fea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:PFEA")
            if torch.isnan(c_fea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:C")
            if torch.isnan(fb_afea).any():  # 检查权重中是否存在 NaN
                print(f"Detected NaN in parameter:FBA")

        return dataset,falsedataset


    def __getitem__(self, index):
        poscname, pos_sample = self.dataset[index]
        negcname, neg_sample = self.falsedataset[index]
        return poscname,negcname,pos_sample, neg_sample


class SplitDataset:
    def __init__(self, dataset, test_ratio=0.5):
        self.train_dataset, self.test_dataset = self.split_dataset(dataset, test_ratio)

    def split_dataset(self, dataset, test_ratio):
        # 计算训练集和测试集的长度
        dataset_size = len(dataset)
        test_size = int(dataset_size * test_ratio)
        train_size = dataset_size - test_size  # 使用 random_split 划分数据集
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset


def create_dataloader():
    batch_size=128
    train_dataset = EmbeddingDataset(trainflag=True)  # Create dataset instance
    test_dataset=TESTEmbeddingDataset(trainflag=False)
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    return  train_loader,test_loader

# def create_dataloader():
#     dataset = EmbeddingDataset()  # Create dataset instance
#     # dataset = EmbeddingDataset()
#     batch_size = 60
#     # 将数据集划分为训练集和测试集
#     split_dataset = SplitDataset(dataset, test_ratio=0.1)

#     # 获取训练集和测试集
#     train_dataset = split_dataset.train_dataset
#     test_dataset = split_dataset.test_dataset

#     # 创建 DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     print("Train dataset size:", len(train_dataset))
#     print("Test dataset size:", len(test_dataset))
#     return  train_loader,test_loader




if __name__ == "__main__":
    dataset = EmbeddingDataset()
    batch_size = 60
    # 将数据集划分为训练集和测试集
    split_dataset = SplitDataset(dataset, test_ratio=0.2)

    # 获取训练集和测试集
    train_dataset = split_dataset.train_dataset
    test_dataset = split_dataset.test_dataset

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))